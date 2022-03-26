# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Optional

from torch import nn, Tensor

from modules.conformer.convolution import ConvolutionModule
from modules.conformer.encoder_layer import EncoderLayer
from modules.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from modules.transformer.embedding import PositionalEncoding, RelPositionalEncoding
from modules.transformer.encoder import EncoderConfig
from modules.transformer.layer_norm import LayerNorm
from modules.transformer.multi_layer_conv import Conv1dLinear
from modules.transformer.multi_layer_conv import MultiLayeredConv1d
from modules.transformer.positionwise_feed_forward import PositionwiseFeedForward
from utils.tools import get_activation


class Encoder(nn.Module):
    """Conformer encoder module."""

    def __init__(self, config: EncoderConfig):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        attention_dim = config["hidden"]
        attention_heads = config["heads"]
        linear_units = config["hidden"] * config["heads"] * 2
        num_blocks = config["layers"]
        dropout_rate = config["dropout_rate"]
        positional_dropout_rate = config["dropout_rate"]
        attention_dropout_rate = config["dropout_rate"]
        normalize_before = config["normalize_before"]
        concat_after = config["concat_after"]
        positionwise_layer_type = config["layer_type"]
        positionwise_conv_kernel_size = config["kernel_size"]
        pos_enc_layer_type = config["pos_enc_layer_type"]
        self_attn_layer_type = config["self_attn_layer_type"]
        activation_type = config["activation_type"]
        cnn_module_kernel = config["cnn_module_kernel"] if "cnn_module_kernel" in config else 31

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert self_attn_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.embed = nn.Sequential(
            pos_enc_class(attention_dim, positional_dropout_rate)
        )
        self.normalize_before = normalize_before

        # self-attention module definition
        if self_attn_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif self_attn_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + self_attn_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    convolution_layer(*convolution_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def forward(self, xs: Tensor, masks: Optional[Tensor]):
        """Encode input sequence.

        Args:
            xs (Tensor): Input tensor (#batch, time, idim).
            masks (Tensor): Mask tensor (#batch, time).

        Returns:
            Tensor: Output tensor (#batch, time, attention_dim).
            Tensor: Mask tensor (#batch, time).

        """
        xs = self.embed(xs)

        for encoder in self.encoders:
            xs, masks = encoder(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
