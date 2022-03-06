import json
import os

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from preprocessor import PreProcessConfig
from text import phoneme_to_id
from utils.pad import pad_1D, pad_2D

from typing import TypedDict, List, Tuple, Dict


class TrainConfigPath(TypedDict):
    ckpt_path: str
    log_path: str
    result_path: str


class TrainConfigOptimizer(TypedDict):
    batch_size: int
    betas: List[float]
    eps: float
    weight_decay: float
    grad_clip_thresh: float
    grad_acc_step: int
    warm_up_step: int
    anneal_steps: List[int]
    anneal_rate: float


class TrainConfigStep(TypedDict):
    total_step: int
    log_step: int
    synth_step: int
    val_step: int
    save_step: int


class TrainConfig(TypedDict):
    path: TrainConfigPath
    optimizer: TrainConfigOptimizer
    step: TrainConfigStep


class DatasetItem(TypedDict):
    id: str
    speaker: int
    text: np.ndarray
    raw_text: str
    accent: np.ndarray
    mel: np.ndarray
    pitch: np.ndarray
    energy: np.ndarray
    duration: np.ndarray


ReProcessedItem = Tuple[
    List[str],
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.int64,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.int64,
    np.ndarray,
    np.ndarray,
    np.ndarray
]


class Dataset(TorchDataset):
    def __init__(
        self,
        filename: str,
        preprocess_config: PreProcessConfig,
        train_config: TrainConfig,
        sort: bool = False,
        drop_last: bool = False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map: Dict[str, int] = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> DatasetItem:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array([phoneme_to_id[p] for p in self.text[idx].split(" ")])
        accent_path = os.path.join(
            self.preprocessed_path,
            "accent",
            "{}-accent-{}.npy".format(speaker, basename),
        )
        accent = np.load(accent_path)[:len(phone) + 1]
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        sample = DatasetItem(
            id=basename,
            speaker=speaker_id,
            text=phone,
            raw_text=raw_text,
            accent=accent,
            mel=mel,
            pitch=pitch,
            energy=energy,
            duration=duration
        )

        return sample

    def process_meta(
        self, filename: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name: List[str] = []
            speaker: List[str] = []
            text: List[str] = []
            raw_text: List[str] = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data: List[DatasetItem], idxs: List[int]) -> ReProcessedItem:
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        accents = [data[idx]["accent"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        accents = pad_1D(accents)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        max_text_len: np.int64 = max(text_lens)
        max_mel_len: np.int64 = max(mel_lens)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max_text_len,
            accents,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            energies,
            durations,
        )

    def collate_fn(self, data: List[DatasetItem]) -> List[ReProcessedItem]:
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output: List[ReProcessedItem] = []
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    preprocess_config = yaml.load(
        open("./config/default.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/default.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
