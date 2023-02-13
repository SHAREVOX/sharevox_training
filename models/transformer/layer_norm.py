#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Layer normalization module."""

from torch import nn, Tensor


class LayerNorm(nn.Module):
    def __init__(self, nout: int, dim: int = -1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim == -1:
            return self.layer_norm(x)
        return (
            self.layer_norm(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )
