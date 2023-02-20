from typing import List, Tuple, Literal, Iterator, Optional

import numpy as np
import torch
from torch import nn, Tensor, device as TorchDevice, LongTensor

from dataset import ReProcessedItem

ReProcessedItemTorch = Tuple[
    List[str],
    Tensor,
    Tensor,
    LongTensor,
    np.int64,
    LongTensor,
    Tensor,
    Tensor,
    Tensor,
    LongTensor,
    np.int64,
    Tensor
]


def to_device(data: ReProcessedItem, device: TorchDevice) -> ReProcessedItemTorch:
    (
        ids,
        speakers,
        phonemes,
        phoneme_lens,
        max_phoneme_len,
        moras,
        accents,
        wavs,
        specs,
        spec_lens,
        max_spec_len,
        pitches,
    ) = data

    speakers = torch.autograd.Variable(torch.from_numpy(speakers).long().to(device, non_blocking=True))
    phonemes = torch.autograd.Variable(torch.from_numpy(phonemes).long().to(device, non_blocking=True))
    phoneme_lens = torch.autograd.Variable(torch.from_numpy(phoneme_lens).long().to(device, non_blocking=True))
    moras = torch.autograd.Variable(torch.from_numpy(moras).long().to(device, non_blocking=True))
    accents = torch.autograd.Variable(torch.from_numpy(accents).long().to(device, non_blocking=True))
    wavs = torch.autograd.Variable(torch.from_numpy(wavs).float().to(device, non_blocking=True))
    specs = torch.autograd.Variable(torch.from_numpy(specs).float().to(device, non_blocking=True))
    spec_lens = torch.autograd.Variable(torch.from_numpy(spec_lens).long().to(device, non_blocking=True))
    pitches = torch.autograd.Variable(torch.from_numpy(pitches).float().to(device, non_blocking=True))

    return (
        ids,
        speakers,
        phonemes,
        phoneme_lens,
        max_phoneme_len,
        moras,
        accents,
        wavs,
        specs,
        spec_lens,
        max_spec_len,
        pitches,
    )


ActivationType = Literal["hardtanh", "tanh", "relu", "selu", "swish"]


def get_activation(act: ActivationType) -> nn.Module:
    """Return activation function."""
    # Lazy load to avoid unused import
    from models.conformer.swish import Swish

    activation_funcs = {
        "hardtanh": nn.Hardtanh,
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "swish": Swish,
    }

    return activation_funcs[act]()


def clip_grad_value_(parameters: Iterator[nn.Parameter], clip_value: Optional[int] = None, norm_type: int = 2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
