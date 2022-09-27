# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Function to get random segments."""

from typing import Tuple

import torch
from torch import Tensor


def get_random_segments(x: Tensor, x_lengths: Tensor, segment_size: int) -> Tuple[Tensor, Tensor]:
    """Get random segments.
    Args:
        x (Tensor): Input tensor (B, C, T).
        x_lengths (Tensor): Length tensor (B,).
        segment_size (int): Segment size.
    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
        Tensor: Start index tensor (B,).
    """
    b, c, t = x.size()
    max_start_idx = x_lengths - segment_size
    start_idxs = (torch.rand([b]).to(x.device) * max_start_idx).to(
        dtype=torch.long,
    )
    segments = get_segments(x, start_idxs, segment_size)
    return segments, start_idxs


def get_segments(x: Tensor, start_idxs: Tensor, segment_size: int) -> Tensor:
    """Get segments.
    Args:
        x (Tensor): Input tensor (B, C, T).
        start_idxs (Tensor): Start index tensor (B,).
        segment_size (int): Segment size.
    Returns:
        Tensor: Segmented tensor (B, C, segment_size).
    """
    b, c, t = x.size()
    segments = x.new_zeros(b, c, segment_size)
    for i, start_idx in enumerate(start_idxs):
        segments[i] = x[i, :, start_idx : start_idx + segment_size]
    return segments
