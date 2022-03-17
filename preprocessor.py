import os
import random
import json
import argparse
import yaml


from text import accent_to_id

import tgt
from scipy.io.wavfile import read as load_wav
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

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

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
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phones_tier = textgrid.get_tier_by_name("phoneme")
        phone, duration, start, end = self.get_alignment(phones_tier)
        text = " ".join(phone)
        if start >= end:
            raise RuntimeError()

        # Read and trim wav files
        data: Tuple[int, np.ndarray] = load_wav(wav_path)
        sr, wav = data
        wav = wav / self.max_wav_value
        wav = normalize(wav) * 0.95
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

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

    def get_alignment(self, tier: tgt.IntervalTier) -> Tuple[List[str], List[int], float, float]:
        sil_phones = ["sil", "sp", "spn"]

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
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time

    def remove_outlier(self, values: np.ndarray) -> np.ndarray:
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config["preprocess"])
    preprocessor.build_from_path()
