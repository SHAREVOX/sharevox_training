import os
import random
import json
import argparse

import torch
import torch.nn.functional as F
import yaml
from scipy.interpolate import interp1d

from text import accent_to_id

from scipy.io.wavfile import read as load_wav
from sklearn.preprocessing import StandardScaler
from librosa.util import normalize
import numpy as np
import pyworld as pw
from tqdm import tqdm

from stft import TacotronSTFT, get_mel_from_wav

from typing import Tuple, List, TypedDict


class PreProcessPath(TypedDict):
    data_path: str
    text_data_path: str
    preprocessed_path: str


class PreProcessAudio(TypedDict):
    sampling_rate: int
    max_wav_value: float


class PreProcessSTFT(TypedDict):
    filter_length: int
    hop_length: int
    win_length: int


class PreProcessMel(TypedDict):
    n_mel_channels: int
    mel_fmin: int
    mel_fmax: int


class PreProcessConfig(TypedDict):
    path: PreProcessPath
    val_size: int
    audio: PreProcessAudio
    stft: PreProcessSTFT
    mel: PreProcessMel


def get_wav(config: PreProcessConfig, speaker: str, basename: str) -> np.ndarray:
    in_dir = config["path"]["data_path"]
    max_wav_value = config["audio"]["max_wav_value"]
    sampling_rate = config["audio"]["sampling_rate"]
    filter_length = config["stft"]["filter_length"]
    hop_length = config["stft"]["hop_length"]

    wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))

    # Read and trim wav files
    data: Tuple[int, np.ndarray] = load_wav(wav_path)
    sr, wav = data
    assert sampling_rate == sr, f"sampling rate is invalid (required: {sampling_rate}, actually: {sr}, file: {wav_path})"
    wav = wav / max_wav_value
    wav = normalize(wav) * 0.95

    # padding
    wav_tensor = torch.FloatTensor(wav)
    wav_tensor = wav_tensor.unsqueeze(0)
    wav_tensor = F.pad(
        wav_tensor.unsqueeze(1),
        (int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2)),
        mode='reflect'
    )

    wav = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    return wav


class Preprocessor:
    def __init__(self, config: PreProcessConfig):
        self.config = config
        self.in_dir = config["path"]["data_path"]
        self.text_dir = config["path"]["text_data_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["val_size"]
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.max_wav_value = config["audio"]["max_wav_value"]
        self.hop_length = config["stft"]["hop_length"]

        self.STFT = TacotronSTFT(
            config["stft"]["filter_length"],
            config["stft"]["hop_length"],
            config["stft"]["win_length"],
            config["mel"]["n_mel_channels"],
            config["audio"]["sampling_rate"],
            config["mel"]["mel_fmin"],
            config["mel"]["mel_fmax"],
        )

    def build_from_path(self) -> List[str]:
        os.makedirs((os.path.join(self.out_dir, "wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "accent")), exist_ok=True)

        print("Processing Data ...")
        out: List[str] = []
        n_frames = 0
        pitch_scaler = StandardScaler()

        # Compute pitch, duration, and mel-spectrogram
        speakers = {}
        dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.in_dir, x)), os.listdir(self.in_dir)))
        for i, speaker in enumerate(tqdm(dirs, desc="Dir", position=0)):
            speakers[speaker] = i
            wavs = list(filter(lambda p: ".wav" in p, os.listdir(os.path.join(self.in_dir, speaker))))
            for wav_name in tqdm(wavs, desc="File", position=1):
                basename = wav_name.split(".")[0]
                pitch, n = self.process_utterance(speaker, basename)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

                n_frames += n

            # phoneme
            phoneme_path = os.path.join(self.text_dir, speaker, "phoneme.csv")
            with open(phoneme_path) as f:
                phonemes = f.read().split("\n")
                # 前後の無音区間のポーズをつけ足す
                out += ["|".join([text.split(",")[0], speaker, "pau " + text.split(",")[1] + " pau"]) for text in phonemes]

            # accent
            accent_path = os.path.join(self.text_dir, speaker, "accent.csv")
            with open(accent_path) as f:
                accents = f.read().split("\n")
                basenames = [text.split(",")[0] for text in accents]
                # 前後の無音区間のアクセント区切りをつけ足す
                accents = ["# " + text.split(",")[1] + " #" for text in accents]

            for j, accent in enumerate(accents):
                basename = basenames[j]
                accent_seq = np.array([accent_to_id[a] for a in accent.split(" ")])
                accent_filename = f"{speaker}-accent-{basename}.npy"
                np.save(os.path.join(self.out_dir, "accent", accent_filename), accent_seq)

        print("Computing statistic quantities ...")
        pitch_mean = pitch_scaler.mean_[0]
        pitch_std = pitch_scaler.scale_[0]

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker: str, basename: str) -> Tuple[np.ndarray, np.ndarray]:
        wav = get_wav(self.config, speaker, basename)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        if np.sum(pitch != 0) <= 1:
            raise RuntimeError()

        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch: np.ndarray = interp_fn(np.arange(0, len(pitch)))

        # Compute mel-scale spectrogram
        mel_spectrogram = get_mel_from_wav(wav, self.STFT)
        assert pitch.size == mel_spectrogram.shape[1], "pitch length != mel spec length"

        # Save files
        wav_filename = "{}-wav-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "wav", wav_filename), wav)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            self.remove_outlier(pitch),
            mel_spectrogram.shape[1],
        )

    def remove_outlier(self, values: np.ndarray) -> np.ndarray:
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir: str, mean: float, std: float) -> Tuple[np.generic, np.generic]:
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in tqdm(os.listdir(in_dir), desc="Normalizing"):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config["preprocess"])
    preprocessor.build_from_path()
