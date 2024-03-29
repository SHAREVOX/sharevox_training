import os
import random
import json
import argparse
import yaml
from scipy.interpolate import interp1d

from text import accent_to_id

import tgt
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


def get_alignment(config: PreProcessConfig, tier: tgt.IntervalTier) -> Tuple[List[str], List[int], float, float]:
    sil_phones = ["sil", "sp", "spn"]
    sampling_rate = config["audio"]["sampling_rate"]
    hop_length = config["stft"]["hop_length"]

    phones = []
    durations = []
    start_time = 0.
    end_time = 0.
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trim leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s

        if p not in sil_phones:
            # For ordinary phones
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            # For silent phones
            phones.append(p)

        durations.append(
            int(
                np.round(e * sampling_rate / hop_length)
                - np.round(s * sampling_rate / hop_length)
            )
        )

    # Trim tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]

    return phones, durations, start_time, end_time


def get_tgt_and_wav(config: PreProcessConfig, speaker: str, basename: str) -> Tuple[str, List[int], np.ndarray]:
    in_dir = config["path"]["data_path"]
    out_dir = config["path"]["preprocessed_path"]
    max_wav_value = config["audio"]["max_wav_value"]
    sampling_rate = config["audio"]["sampling_rate"]

    wav_path = os.path.join(in_dir, speaker, "{}.wav".format(basename))
    tg_path = os.path.join(
        out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
    )

    # Get alignments
    textgrid = tgt.io.read_textgrid(tg_path)
    phones_tier = textgrid.get_tier_by_name("phoneme")
    phone, duration, start, end = get_alignment(config, phones_tier)
    text = " ".join(phone)
    if start >= end:
        raise RuntimeError()

    # Read and trim wav files
    data: Tuple[int, np.ndarray] = load_wav(wav_path)
    sr, wav = data
    assert sampling_rate == sr, f"sampling rate is invalid (required: {sampling_rate}, actually: {sr}, file: {wav_path})"
    wav = wav / max_wav_value
    wav = normalize(wav) * 0.95
    wav = wav[
        int(sampling_rate * start): int(sampling_rate * end)
    ].astype(np.float32)

    return text, duration, wav


class Preprocessor:
    def __init__(self, config: PreProcessConfig):
        self.config = config
        self.in_dir = config["path"]["data_path"]
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
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "accent")), exist_ok=True)

        print("Processing Data ...")
        out: List[str] = []
        n_frames = 0
        pitch_scaler = StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        dirs = list(filter(lambda x: os.path.isdir(os.path.join(self.in_dir, x)), os.listdir(self.in_dir)))
        for i, speaker in enumerate(tqdm(dirs, desc="Dir", position=0)):
            speakers[speaker] = i
            wavs = list(filter(lambda p: ".wav" in p, os.listdir(os.path.join(self.in_dir, speaker))))
            for wav_name in tqdm(wavs, desc="File", position=1):
                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, n = ret
                    out.append(info)
                else:
                    raise Exception("TextGrid not found")

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))

                n_frames += n

            # accent
            accent_path = os.path.join(self.in_dir, speaker, "accent.csv")
            with open(accent_path) as f:
                accents = f.read().split("\n")
                basenames = [text.split(",")[0] for text in accents]
                accents = [text.split(",")[1] for text in accents]

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

    def process_utterance(self, speaker: str, basename: str) -> Tuple[str, np.ndarray, np.ndarray]:
        text, duration, wav = get_tgt_and_wav(self.config, speaker, basename)

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
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

        # Phoneme-level average
        pos = 0
        for i, d in enumerate(duration):
            if d > 0:
                pitch[i] = np.mean(pitch[pos : pos + d])
            else:
                pitch[i] = 0
            pos += d
        pitch = pitch[: len(duration)]

        # Compute mel-scale spectrogram
        mel_spectrogram = get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]

        # Save files
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

        return (
            "|".join([basename, speaker, text]),
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
