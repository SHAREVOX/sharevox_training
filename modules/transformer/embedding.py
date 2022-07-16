# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positional Encoding Module."""

import math

import torch
from torch import nn, Tensor

from typing import Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, d_model: int, dropout_rate: float):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.
        Args:
            x (Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        pe = torch.zeros(x.shape[1], self.d_model).to(device=x.device)
        # pe = torch.zeros(x.size(1), self.d_model // 2, 2)
        position = torch.arange(0, x.shape[1], dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe[:, :, 0] = torch.sin(position * div_term)
        # pe[:, :, 1] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        x = x * self.xscale + pe[:, : x.shape[1]]
        return self.dropout(x)


class RelPositionalEncoding(nn.Module):
    """Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, d_model, dropout_rate):
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncoding, self).__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Add positional encoding.
        Args:
            x (Tensor): Input tensor (batch, time, `*`).
        Returns:
            Tensor: Encoded tensor (batch, time, `*`).
        """
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model // 2, 2).to(device=x.device)
        pe_negative = torch.zeros(x.size(1), self.d_model // 2, 2).to(device=x.device)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, :, 0] = torch.sin(position * div_term)
        pe_positive[:, :, 1] = torch.cos(position * div_term)
        pe_negative[:, :, 0] = torch.sin(-1 * position * div_term)
        pe_negative[:, :, 1] = torch.cos(-1 * position * div_term)

        pe_positive = pe_positive.view(x.size(1), self.d_model)
        pe_negative = pe_negative.view(x.size(1), self.d_model)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)

        x = x * self.xscale
        pos_emb = pe[
            :,
            pe.size(1) // 2 - x.size(1) + 1 : pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)
