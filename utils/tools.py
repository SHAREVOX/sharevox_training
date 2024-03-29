from typing import overload, List, Tuple, Any, Literal

import numpy as np
import torch
from torch import nn, Tensor, device as TorchDevice, LongTensor

from dataset import ReProcessedItem, ReProcessedTextItem

ReProcessedItemTorch = Tuple[
    List[str],
    Tensor,
    Tensor,
    LongTensor,
    np.int64,
    Tensor,
    Tensor,
    LongTensor,
    np.int64,
    Tensor,
    Tensor
]

ReProcessedTextItemTorch = Tuple[
    List[str],
    Tensor,
    Tensor,
    LongTensor,
    np.int64,
    Tensor
]


@overload
def to_device(data: ReProcessedItem, device: TorchDevice) -> ReProcessedItemTorch:
    pass


@overload
def to_device(data: ReProcessedTextItem, device: TorchDevice) -> ReProcessedTextItemTorch:
    pass


def to_device(data: Any, device: TorchDevice):
    if len(data) == 11:
        (
            ids,
            speakers,
            phonemes,
            phoneme_lens,
            max_phoneme_len,
            accents,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            durations,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        phonemes = torch.from_numpy(phonemes).long().to(device)
        phoneme_lens = torch.from_numpy(phoneme_lens).long().to(device)
        accents = torch.from_numpy(accents).long().to(device)
        mels = torch.from_numpy(mels).float().to(device)
        mel_lens = torch.from_numpy(mel_lens).long().to(device)
        pitches = torch.from_numpy(pitches).float().to(device)
        durations = torch.from_numpy(durations).long().to(device)

        return (
            ids,
            speakers,
            phonemes,
            phoneme_lens,
            max_phoneme_len,
            accents,
            mels,
            mel_lens,
            max_mel_len,
            pitches,
            durations,
        )

    if len(data) == 6:
        (ids, speakers, phonemes, phoneme_lens, max_phoneme_len, accents) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        phonemes = torch.from_numpy(phonemes).long().to(device)
        phoneme_lens = torch.from_numpy(phoneme_lens).to(device)
        accents = torch.from_numpy(accents).long().to(device)

        return ids, speakers, phonemes, phoneme_lens, max_phoneme_len, accents


ActivationType = Literal["hardtanh", "tanh", "relu", "selu", "swish"]


def get_activation(act: ActivationType) -> nn.Module:
    """Return activation function."""
    # Lazy load to avoid unused import
    from modules.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()
