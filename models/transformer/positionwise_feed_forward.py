#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

from torch import nn, Tensor


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (nn.Module): Activation module.

    """

    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
