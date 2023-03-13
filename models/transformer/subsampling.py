#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""
from typing import Optional, Tuple

from torch import nn, Tensor

from models.transformer.embedding import PositionalEncoding


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (nn.Module): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x: Tensor, x_mask: Optional[Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        """Subsample x.
        Args:
            x (Tensor): Input tensor (#batch, time, idim).
            x_mask (Tensor): Input mask (#batch, 1, time).
        Returns:
            Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

    def __getitem__(self, key: int) -> nn.Module:
        """Get item.
        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.
        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
