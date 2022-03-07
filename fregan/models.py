import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm

from typing import Tuple, List, Optional

from .config import Config

LRELU_SLOPE = 0.1


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int] = (1, 3, 5, 7)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[3],
                        padding=get_padding(kernel_size, dilation[3]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h: Config, top_k: int = 4):
        super(Generator, self).__init__()

        self.num_kernels: int = len(h.resblock_kernel_sizes)
        self.num_upsamples: int = len(h.upsample_rates)
        self.upsample_rates: List[int] = h.upsample_rates
        self.up_kernels: List[int] = h.upsample_kernel_sizes
        self.cond_level: int = self.num_upsamples - top_k
        self.conv_pre: nn.Module = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock

        self.ups = nn.ModuleList()
        self.cond_up = nn.ModuleList()
        self.res_output = nn.ModuleList()
        upsample_: int = 1
        kr: int = 80

        for i, (u, k) in enumerate(zip(self.upsample_rates, self.up_kernels)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

            if i > (self.num_upsamples - top_k):
                self.res_output.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=u, mode='nearest'),
                        weight_norm(nn.Conv1d(h.upsample_initial_channel // (2 ** i),
                                              h.upsample_initial_channel // (2 ** (i + 1)), 1))
                    )
                )
            if i >= (self.num_upsamples - top_k):
                self.cond_up.append(
                    weight_norm(
                        ConvTranspose1d(kr, h.upsample_initial_channel // (2 ** i),
                                        self.up_kernels[i - 1], self.upsample_rates[i - 1],
                                        padding=(self.up_kernels[i - 1] - self.upsample_rates[i - 1]) // 2))
                )
                kr = h.upsample_initial_channel // (2 ** i)

            upsample_ *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post: nn.Module = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.cond_up.apply(init_weights)
        self.res_output.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        mel = x
        x = self.conv_pre(x)
        output: Optional[Tensor] = None
        for i in range(self.num_upsamples):
            if i >= self.cond_level:
                mel = self.cond_up[i - self.cond_level](mel)
                x += mel
            if i > self.cond_level:
                if output is None:
                    output = self.res_output[i - self.cond_level - 1](x)
                else:
                    output = self.res_output[i - self.cond_level - 1](output)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs: Optional[Tensor] = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            if output is not None:
                output = output + x

        x = F.leaky_relu(output)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self) -> None:
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        for l in self.cond_up:
            remove_weight_norm(l)
        for l in self.res_output:
            remove_weight_norm(l[1])
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)