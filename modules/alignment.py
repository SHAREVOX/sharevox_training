# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import jit, prange
from torch import Tensor


class AlignmentModule(nn.Module):
    """Alignment Learning Framework proposed for parallel TTS models in:

    https://arxiv.org/abs/2108.10447

    """

    def __init__(self, adim: int, odim: int, temperature: float):
        super().__init__()
        self.temperature = temperature

        self.text_proj = nn.Sequential(
            nn.Conv1d(
                adim,
                adim,
                kernel_size=3,
                bias=True,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv1d(
                adim,
                adim,
                kernel_size=1,
                bias=True,
                padding=0
            ),
        )

        self.feat_proj = nn.Sequential(
            nn.Conv1d(
                odim,
                adim,
                kernel_size=3,
                bias=True,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                adim,
                adim,
                kernel_size=3,
                bias=True,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                adim,
                adim,
                kernel_size=1,
                bias=True,
                padding=0,
            ),
        )

        self.text_spk_proj = nn.Linear(adim, adim, bias=False)
        self.feat_spk_proj = nn.Linear(adim, odim, bias=False)


    def forward(self, texts: Tensor, feats: Tensor, x_masks: Tensor, attn_prior: Tensor, speaker_embed: Tensor):
        """Calculate alignment loss.

        Args:
            text (Tensor): Batched text embedding (B, T_text, adim).
            feats (Tensor): Batched acoustic feature (B, T_feats, odim).
            x_masks (Tensor): Mask tensor (B, T_text).
            attn_prior (Tensor): prior for attention matrix.
            speaker_embed (Tensor): speaker embedding for multi-speaker scheme.

        Returns:
            attn (Tensor):attention mask.
            attn_logprob (Tensor): log-prob attention mask.

        """
        texts = texts.transpose(1, 2)
        texts = texts + self.text_spk_proj(speaker_embed.unsqueeze(1).expand(
            -1, texts.shape[-1], -1
        )).transpose(1, 2)

        feats = feats.transpose(1, 2)
        feats = feats + self.feat_spk_proj(speaker_embed.unsqueeze(1).expand(
            -1, feats.shape[-1], -1
        )).transpose(1, 2)

        texts_enc = self.text_proj(texts)
        feats_enc = self.feat_proj(feats)

        # Simplistic Gaussian Isotopic Attention
        attn = (feats_enc[:, :, :, None] - texts_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            # print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = F.log_softmax(attn, dim=-1) + torch.log(attn_prior[:, None] + 1e-8)
            # print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if x_masks is not None:
            attn.data.masked_fill_(x_masks.permute(0, 2, 1).unsqueeze(2), -float("inf"))

        attn = F.softmax(attn, dim=-1)
        return attn, attn_logprob


@jit(nopython=True)
def mas_width1(attn_map: np.ndarray) -> np.ndarray:
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True, parallel=True)
def b_mas(b_attn_map: np.ndarray, in_lens: np.ndarray, out_lens: np.ndarray, width: int = 1) -> np.ndarray:
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


def binarize_attention_parallel(attn: Tensor, in_lens: Tensor, out_lens: Tensor) -> Tensor:
    """For training purposes only. Binarizes attention with MAS.
    These will no longer recieve a gradient.
    Args:
        attn: B x 1 x max_mel_len x max_text_len
    """
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.device)


@jit(nopython=True)
def _average_by_duration(ds, xs, text_lengths, feats_lengths):
    B = ds.shape[0]
    xs_avg = np.zeros_like(ds)
    ds = ds.astype(np.int32)
    for b in range(B):
        t_text = text_lengths[b]
        t_feats = feats_lengths[b]
        d = ds[b, :t_text]
        d_cumsum = d.cumsum()
        d_cumsum = [0] + list(d_cumsum)
        x = xs[b, :t_feats]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])):
            if len(x[start:end]) != 0:
                xs_avg[b, n] = x[start:end].mean()
            else:
                xs_avg[b, n] = 0
    return xs_avg


def average_by_duration(ds, xs, text_lengths, feats_lengths):
    """Average frame-level features into token-level according to durations

    Args:
        ds (Tensor): Batched token duration (B, T_text).
        xs (Tensor): Batched feature sequences to be averaged (B, T_feats).
        text_lengths (Tensor): Text length tensor (B,).
        feats_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched feature averaged according to the token duration (B, T_text).

    """
    device = ds.device
    args = [ds, xs, text_lengths, feats_lengths]
    args = [arg.detach().cpu().numpy() for arg in args]
    xs_avg = _average_by_duration(*args)
    xs_avg = torch.from_numpy(xs_avg).to(device)
    return xs_avg
