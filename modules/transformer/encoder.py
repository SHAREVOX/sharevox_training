# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
from typing import Union, Tuple, Type, Optional, List, TypedDict, Literal

from torch import nn, Tensor

from modules.transformer.attention import MultiHeadedAttention
from modules.transformer.embedding import PositionalEncoding
from modules.transformer.encoder_layer import EncoderLayer
from modules.transformer.layer_norm import LayerNorm
from modules.transformer.multi_layer_conv import Conv1dLinear
from modules.transformer.multi_layer_conv import MultiLayeredConv1d
from modules.transformer.positionwise_feed_forward import PositionwiseFeedForward

LayerTypeLiteral = Literal["linear", "conv1d", "conv1d-linear"]


class EncoderConfig(TypedDict):
    hidden: int
    heads: int
    layers: int
    dropout_rate: float
    normalize_before: bool
    concat_after: bool
    layer_type: LayerTypeLiteral
    kernel_size: int


class Encoder(nn.Module):
    """Transformer encoder module."""

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
        pos_enc_class = PositionalEncoding
        normalize_before = config["normalize_before"]
        concat_after = config["concat_after"]
        positionwise_layer_type = config["layer_type"]
        positionwise_conv_kernel_size = config["kernel_size"]

        self.embed = nn.Sequential(
            pos_enc_class(attention_dim, positional_dropout_rate)
        )
        self.normalize_before = normalize_before
        positionwise_layer, positionwise_layer_args = self.get_positionwise_layer(
            positionwise_layer_type,
            attention_dim,
            linear_units,
            dropout_rate,
            positionwise_conv_kernel_size,
        )

        encoder_selfattn_layer = MultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate)

        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    dropout_rate,
                    normalize_before,
                    concat_after,
                )
                for _ in range(num_blocks)
            ]
        )

        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

    def get_positionwise_layer(
        self,
        positionwise_layer_type: LayerTypeLiteral = "linear",
        attention_dim: int = 256,
        linear_units: int = 2048,
        dropout_rate: float = 0.1,
        positionwise_conv_kernel_size: int = 1,
    ) -> Tuple[Type[Union[PositionwiseFeedForward, MultiLayeredConv1d, Conv1dLinear]], Tuple[Union[int, float], ...]]:
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
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
        return positionwise_layer, positionwise_layer_args

    def forward(self, xs: Tensor, masks: Tensor) -> Tuple[Tensor, Tensor]:
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

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks

    def forward_one_step(
        self, xs: Tensor, masks: Tensor, cache: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Tensor, Optional[List[Tensor]]]:
        """Encode input frame.

        Args:
            xs (Tensor): Input tensor.
            masks (Tensor): Mask tensor.
            cache (List[Tensor]): List of cache tensors.

        Returns:
            Tensor: Output tensor.
            Tensor: Mask tensor.
            List[Tensor]: List of new cache tensors.

        """
        xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
