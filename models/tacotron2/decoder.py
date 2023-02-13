#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Tacotron2 decoder related modules."""

from torch import nn, Tensor


class Postnet(nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail structure of spectrogram.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        dim: int = 80,
        n_layers: int = 5,
        n_chans: int = 512,
        n_filts: int = 5,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize postnet module.
        Args:
            dim (int): Dimension of the input and outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(Postnet, self).__init__()
        self.postnet = nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = dim if layer == 0 else n_chans
            ochans = dim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    nn.Sequential(
                        nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        nn.BatchNorm1d(ochans),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    nn.Sequential(
                        nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        nn.Tanh(),
                        nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else dim
        if use_batch_norm:
            self.postnet += [
                nn.Sequential(
                    nn.Conv1d(
                        ichans,
                        dim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    nn.BatchNorm1d(dim),
                    nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                nn.Sequential(
                    nn.Conv1d(
                        ichans,
                        dim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    nn.Dropout(dropout_rate),
                )
            ]

    def forward(self, xs: Tensor) -> Tensor:
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, dim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, dim, Tmax).
        """
        for postnet in self.postnet:
            xs = postnet(xs)
        return xs
