# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stochastic predictor modules in VITS.

This code is based on https://github.com/jaywalnut310/vits.

"""

import math
from typing import Optional, TypedDict

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from models.vits.flow import ConvFlow, DilatedDepthSeparableConv, ElementwiseAffineFlow, FlipFlow, LogFlow


class StochasticPreditorConfig(TypedDict):
    kernel_size: int
    flows: int
    dds_conv_layers: int
    dropout: float


class StochasticPredictor(nn.Module):
    """Stochastic predictor module.

    This is a module of stochastic predictor described in `Conditional
    Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech`_.

    .. _`Conditional Variational Autoencoder with Adversarial Learning for End-to-End
        Text-to-Speech`: https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        config: StochasticPreditorConfig,
        channels: int = 192,
        gin_channels: int = -1,
    ):
        """Initialize StochasticDurationPredictor module.

        Args:
            channels (int): Number of channels.
            kernel_size (int): Kernel size.
            dropout_rate (float): Dropout rate.
            flows (int): Number of flows.
            dds_conv_layers (int): Number of conv layers in DDS conv.
            gin_channels (int): Number of global conditioning channels.

        """
        super().__init__()

        kernel_size = config["kernel_size"]
        dds_conv_layers = config["dds_conv_layers"]
        dropout_rate = config["dropout"]
        flows = config["flows"]

        self.pre = nn.Conv1d(channels, channels, 1)
        self.dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.proj = nn.Conv1d(channels, channels, 1)

        self.log_flow = LogFlow()
        self.flows = nn.ModuleList()
        self.flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.flows += [FlipFlow()]

        self.post_pre = nn.Conv1d(1, channels, 1)
        self.post_dds = DilatedDepthSeparableConv(
            channels,
            kernel_size,
            layers=dds_conv_layers,
            dropout_rate=dropout_rate,
        )
        self.post_proj = nn.Conv1d(channels, channels, 1)
        self.post_flows = nn.ModuleList()
        self.post_flows += [ElementwiseAffineFlow(2)]
        for i in range(flows):
            self.post_flows += [
                ConvFlow(
                    2,
                    channels,
                    kernel_size,
                    layers=dds_conv_layers,
                )
            ]
            self.post_flows += [FlipFlow()]

        if gin_channels > 0:
            self.global_conv = nn.Conv1d(gin_channels, channels, 1)

    def forward(
        self,
        x: Tensor,
        x_mask: Tensor,
        w: Optional[Tensor] = None,
        g: Optional[Tensor] = None,
        inverse: bool = False,
        noise_scale: float = 1.0,
    ) -> Tensor:
        """Calculate forward propagation.

        Args:
            x (Tensor): Input tensor (B, channels, T_text).
            x_mask (Tensor): Mask tensor (B, 1, T_text).
            w (Optional[Tensor]): Duration tensor (B, 1, T_text).
            g (Optional[Tensor]): Global conditioning tensor (B, channels, 1)
            inverse (bool): Whether to inverse the flow.
            noise_scale (float): Noise scale value.

        Returns:
            Tensor: If not inverse, negative log-likelihood (NLL) tensor (B,).
                If inverse, log-duration tensor (B, 1, T_text).

        """
        x = x.detach()  # stop gradient
        x = self.pre(x)
        if g is not None:
            x = x + self.global_conv(g.detach())  # stop gradient
        x = self.dds(x, x_mask)
        x = self.proj(x) * x_mask

        if not inverse:
            assert w is not None, "w must be provided."
            h_w = self.post_pre(w)
            h_w = self.post_dds(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = (
                torch.randn(
                    w.size(0),
                    2,
                    w.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * x_mask
            )
            z_q = e_q
            logdet_tot_q = 0.0
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in self.flows:
                z, logdet = flow(z, x_mask, g=x, inverse=inverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # (B,)
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = (
                torch.randn(
                    x.size(0),
                    2,
                    x.size(2),
                ).to(device=x.device, dtype=x.dtype)
                * noise_scale
            )
            for flow in flows:
                z = flow(z, x_mask, g=x, inverse=inverse)
            z0, z1 = z.split(1, 1)
            logw = z0
            return logw
