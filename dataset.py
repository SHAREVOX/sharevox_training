import json
import os

import torch
import numpy as np
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from preprocessor import PreProcessConfig
from text import _symbol_to_id as phoneme_to_id
from utils.pad import pad_1D, pad_2D #, pad_3D

from typing import TypedDict, List, Tuple, Dict, Optional


class TrainConfigOptimizer(TypedDict):
    batch_size: int
    betas: Tuple[float, float]
    learning_rate: float
    lr_decay: float
    eps: float


class TrainConfigStep(TypedDict):
    total_epoch: int
    log_step: int
    synth_step: int
    val_step: int
    save_step: int
    feat_learn_start: int


class TrainConfigLossBalance(TypedDict):
    mel: float
    kl: float
    align: float
    reg: float
    gen: float
    fm: float


class TrainConfig(TypedDict):
    optimizer: TrainConfigOptimizer
    step: TrainConfigStep
    loss_balance: TrainConfigLossBalance
    seed: int
    fp16_run: bool


class DatasetItem(TypedDict):
    id: str
    speaker: int
    text: np.ndarray
    accent: np.ndarray
    wav: np.ndarray
    spec: np.ndarray
    pitch: np.ndarray


ReProcessedItem = Tuple[
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.int64,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.int64,
    np.ndarray,
]

ReProcessedTextItem = Tuple[
    List[str],
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.int64,
    np.ndarray
]


class Dataset(TorchDataset):
    def __init__(
        self,
        filename: str,
        preprocess_config: PreProcessConfig,
        train_config: TrainConfig
    ):
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        self.basename, self.speaker, self.text, self.lengths = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map: Dict[str, int] = json.load(f)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, idx: int) -> DatasetItem:
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        phone = np.array([phoneme_to_id[p] for p in self.text[idx].split(" ")])
        accent_path = os.path.join(
            self.preprocessed_path,
            "accent",
            "{}-accent-{}.npy".format(speaker, basename),
        )
        accent = np.load(accent_path)[:len(phone) + 1]
        wav_path = os.path.join(
            self.preprocessed_path,
            "wav",
            "{}-wav-{}.npy".format(speaker, basename),
        )
        wav = np.load(wav_path)
        spec_path = os.path.join(
            self.preprocessed_path,
            "spec",
            "{}-spec-{}.npy".format(speaker, basename),
        )
        spec = np.load(spec_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)

        sample = DatasetItem(
            id=basename,
            speaker=speaker_id,
            text=phone,
            accent=accent,
            wav=wav,
            spec=spec,
            pitch=pitch,
        )

        return sample

    def process_meta(
        self, filename: str
    ) -> Tuple[List[str], List[str], List[str], List[int]]:
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name: List[str] = []
            speaker: List[str] = []
            text: List[str] = []
            length: List[int] = []
            for line in f.readlines():
                n, s, t = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                pitch_path = os.path.join(
                    self.preprocessed_path,
                    "pitch",
                    "{}-pitch-{}.npy".format(s, n),
                )
                pitch = np.load(pitch_path)
                length.append(pitch.shape[0])
            return name, speaker, text, length

    def reprocess(self, data: List[DatasetItem], idxs: List[int]) -> ReProcessedItem:
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        accents = [data[idx]["accent"] for idx in idxs]
        wavs = [data[idx]["wav"] for idx in idxs]
        specs = [data[idx]["spec"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        spec_lens = np.array([spec.shape[0] for spec in specs])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        accents = pad_1D(accents)
        wavs = pad_1D(wavs)
        specs = pad_2D(specs)
        pitches = pad_1D(pitches)

        max_text_len: np.int64 = max(text_lens)
        max_spec_len: np.int64 = max(spec_lens)

        return (
            ids,
            speakers,
            texts,
            text_lens,
            max_text_len,
            accents,
            wavs,
            specs,
            spec_lens,
            max_spec_len,
            pitches,
        )

    def collate_fn(self, data: List[DatasetItem]) -> ReProcessedItem:
        idx_arr = np.argsort(
            np.array([d["spec"].shape[0] for d in data]),
            axis=0
        )[::-1]

        return self.reprocess(data, idx_arr.tolist())


class DistributedBucketSampler(DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        boundaries: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    # Test
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
        "train.txt", preprocess_config, train_config
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config
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
