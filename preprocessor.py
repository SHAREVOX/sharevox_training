import os
import random
import json
import argparse

import librosa
import resampy
import torch
import yaml
from scipy.interpolate import interp1d
from scipy.stats import betabinom

from text import _accent_to_id as accent_to_id

from sklearn.preprocessing import StandardScaler
from librosa.util import normalize
import numpy as np
import pyworld as pw
from tqdm import tqdm as tqdm_console
from tqdm.notebook import tqdm as tqdm_notebook

from utils.mel_processing import spectrogram_torch as spectrogram

from typing import Tuple, List, TypedDict


class PreProcessPath(TypedDict):
    data_path: str
    text_data_path: str
    preprocessed_path: str


class PreProcessAudio(TypedDict):
    sampling_rate: int
    max_wav_value: float
    trim_top_db: int

class PreProcessSTFT(TypedDict):
    filter_length: int
    hop_length: int
    win_length: int


class PreProcessMel(TypedDict):
    n_mel_channels: int
    n_mel_channels_loss: int
    mel_fmin: int
    mel_fmax: int
    mel_fmax_loss: int


class PreProcessConfig(TypedDict):
    path: PreProcessPath
    val_size: int
    audio: PreProcessAudio
    stft: PreProcessSTFT
    mel: PreProcessMel


def load_wav(full_path: str, sr: int, filter_length: int, hop_length: int, trim_top_db: int) -> Tuple[np.ndarray, int]:
    # not resampling in librosa
    data, sampling_rate = librosa.load(full_path, sr=None)
    # resampling by outside of librosa and showing warning
    if sampling_rate != sr:
        tqdm_console.write(f"sampling rate is different(required: {sr}, actually: {sampling_rate}, file: {full_path}), auto converted by script.")
        data = resampy.resample(data, sampling_rate, sr, filter="kaiser_best")

    _, index = librosa.effects.trim(data, top_db=trim_top_db, frame_length=filter_length, hop_length=hop_length)
    pau_duration = sampling_rate // 5  # 0.2s
    start = index[0] - pau_duration
    end = index[1] + pau_duration
    if start < 0:
        start = 0
    if end > len(data) - 1:
        end = len(data) - 1
    data = data[start:end]
    return data, (end - start) // hop_length


def get_wav(config: PreProcessConfig, speaker: str, basename: str) -> Tuple[np.ndarray, int]:
    in_dir = config["path"]["data_path"]
    sampling_rate = config["audio"]["sampling_rate"]
    trim_top_db = config["audio"]["trim_top_db"]
    filter_length = config["stft"]["filter_length"]
    hop_length = config["stft"]["hop_length"]

    wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))

    # Read and trim wav files
    wav, duration = load_wav(wav_path, sampling_rate, filter_length, hop_length, trim_top_db)
    wav = normalize(wav) * 0.95
    wav = wav.astype(np.float32)

    return wav, duration


class Preprocessor:
    def __init__(self, config: PreProcessConfig, tqdm_mode = "console"):
        self.config = config
        self.in_dir = config["path"]["data_path"]
        self.text_dir = config["path"]["text_data_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["val_size"]
        self.sampling_rate = config["audio"]["sampling_rate"]
        self.max_wav_value = config["audio"]["max_wav_value"]
        self.hop_length = config["stft"]["hop_length"]
        assert tqdm_mode in ["console", "notebook"], "tqdm mode must be console or notebook"
        self.tqdm = tqdm_notebook if tqdm_mode == "notebook" else tqdm_console

    def build_from_path(self) -> List[str]:
        os.makedirs((os.path.join(self.out_dir, "wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "spec")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "accent")), exist_ok=True)

        print("Processing Data ...")
        out: List[str] = []
        n_frames = 0
        pitch_scaler = StandardScaler()

        # Compute pitch, duration, and mel-spectrogram
        speakers = {}
        dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.in_dir, x)), os.listdir(self.in_dir)))
        for i, speaker in enumerate(self.tqdm(dirs, desc="Dir", position=0)):
            speakers[speaker] = i

            # phoneme
            phoneme_path = os.path.join(self.text_dir, speaker, "phoneme.csv")
            if not os.path.isfile(phoneme_path):
                phoneme_path = os.path.join(self.text_dir, "phoneme.csv")
            with open(phoneme_path) as f:
                phonemes = filter(lambda x: bool(x), f.read().split("\n"))
                # 前後の無音区間のポーズをつけ足す
                out += ["|".join([text.split(",")[0], speaker, "pau " + text.split(",")[1] + " pau"]) for text in phonemes]

            # accent
            accent_path = os.path.join(self.text_dir, speaker, "accent.csv")
            if not os.path.isfile(accent_path):
                accent_path = os.path.join(self.text_dir, "accent.csv")
            with open(accent_path) as f:
                accents = list(filter(lambda x: bool(x), f.read().split("\n")))
                basenames = [text.split(",")[0] for text in accents]
                # 前後の無音区間のアクセント区切りをつけ足す
                accents = ["# " + text.split(",")[1] + " #" for text in accents]

            wavs = list(filter(lambda p: ".wav" in p, os.listdir(os.path.join(self.in_dir, speaker))))
            for wav_name in self.tqdm(wavs, desc="File", position=1):
                basename = wav_name.split(".")[0]
                filter_out = list(filter(lambda d: basename in d, out))
                if len(filter_out) == 0:
                    continue
                phonemes = filter_out[0].split("|")[2].split(" ")
                pitch, n = self.process_utterance(speaker, basename, phonemes)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

                n_frames += n

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
            for m in out:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker: str, basename: str, phonemes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        wav, duration = get_wav(self.config, speaker, basename)

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
        pitch = pitch[:duration]

        # Compute mel-scale spectrogram
        wav_torch = torch.FloatTensor(wav).unsqueeze(0)
        spec = spectrogram(
            y=wav_torch,
            n_fft=self.config["stft"]["filter_length"],
            hop_size=self.config["stft"]["hop_length"],
            win_size=self.config["stft"]["win_length"],
        )
        spec = torch.squeeze(spec, 0).numpy().astype(np.float32)

        assert pitch.size == spec.shape[1], f"pitch length != spec length ({pitch.size} != {spec.shape[1]})"

        # Save files
        wav_filename = "{}-wav-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "wav", wav_filename), wav)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        spec_filename = "{}-spec-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "spec", spec_filename),
            spec.T,
        )

        return (
            self.remove_outlier(pitch),
            spec.shape[1],
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
        for filename in self.tqdm(os.listdir(in_dir), desc="Normalizing"):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    parser.add_argument("-t", "--tqdm_mode", type=str, help="tqdm mode (console or notebook)", default="console")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config["preprocess"], tqdm_mode=args.tqdm_mode)
    preprocessor.build_from_path()
