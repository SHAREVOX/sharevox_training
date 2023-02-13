# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Posterior encoder module in VITS.
This code is based on https://github.com/jaywalnut310/vits.
"""

from typing import Optional, Tuple, TypedDict

import torch
from torch import nn, Tensor, LongTensor

from models.wavenet import WaveNet
from models.wavenet.residual_block import Conv1d
from utils.mask import make_non_pad_mask


class PosteriorEncoderConfig(TypedDict):
    hidden: int
    kernel_size: int
    layers: int
    stacks: int
    base_dilation: int
    dropout: int


class PosteriorEncoder(nn.Module):
    """Posterior encoder module in VITS.
    This is a module of posterior encoder described in `Conditional Variational
    Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.
    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558
    """

    def __init__(
        self,
        config: PosteriorEncoderConfig,
        in_channels: int,
        out_channels: int,
        gin_channels: int = -1,
    ):
        """Initilialize PosteriorEncoder module."""
        super().__init__()

        hidden = config["hidden"]
        kernel_size = config["kernel_size"]
        layers = config["layers"]
        stacks = config["stacks"]
        base_dilation = config["base_dilation"]
        dropout = config["dropout"]

        # define modules
        self.input_conv = Conv1d(in_channels, hidden, 1)
        self.encoder = WaveNet(
            in_channels=-1,
            out_channels=-1,
            kernel_size=kernel_size,
            layers=layers,
            stacks=stacks,
            base_dilation=base_dilation,
            residual_channels=hidden,
            aux_channels=-1,
            gate_channels=hidden * 2,
            skip_channels=hidden,
            global_channels=gin_channels,
            dropout_rate=dropout,
            bias=True,
            use_weight_norm=True,
            use_first_conv=False,
            use_last_conv=False,
            scale_residual=False,
            scale_skip_connect=True,
        )
        self.proj = Conv1d(hidden, out_channels * 2, 1)

    def forward(self, x: Tensor, x_lengths: LongTensor, g: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T_feats).
            x_lengths (Tensor): Length tensor (B,).
            g (Optional[Tensor]): Global conditioning tensor (B, global_channels, 1).
        Returns:
            Tensor: Encoded hidden representation tensor (B, out_channels, T_feats).
            Tensor: Projected mean tensor (B, out_channels, T_feats).
            Tensor: Projected scale tensor (B, out_channels, T_feats).
            Tensor: Mask tensor for input tensor (B, 1, T_feats).
        """
        x_mask = (
            make_non_pad_mask(x_lengths)
            .unsqueeze(1)
            .to(
                dtype=x.dtype,
                device=x.device,
            )
        )
        x = self.input_conv(x) * x_mask
        x = self.encoder(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = stats.split(stats.size(1) // 2, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask

        return z, m, logs, x_mask
