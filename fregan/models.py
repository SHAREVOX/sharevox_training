import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from typing import Tuple, List, Optional, TypedDict

from stft import STFT

LRELU_SLOPE = 0.1
Pad_Mode = ['constant', 'reflect', 'replicate', 'circular']


class VocoderConfig(TypedDict):
    resblock: str
    upsample_rates: List[int]
    upsample_kernel_sizes:  List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[Tuple[int]]
    segment_size: int


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    stft_fn = STFT(fft_size, hop_size, win_length, window)
    x_stft, _ = stft_fn.transform(x)
    return x_stft


class DWT_1D(nn.Module):
    def __init__(self, pad_type='reflect', wavename='haar',
                 stride=2, in_channels=1, out_channels=None, groups=None,
                 kernel_size=None, trainable=False):

        super(DWT_1D, self).__init__()
        self.trainable = trainable
        self.kernel_size = kernel_size
        if not self.trainable:
            assert self.kernel_size == None
        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels == None else out_channels
        self.groups = self.in_channels if groups == None else groups
        assert isinstance(self.groups, int) and self.in_channels % self.groups == 0
        self.stride = stride
        assert self.stride == 2
        self.wavename = wavename
        self.pad_type = pad_type
        assert self.pad_type in Pad_Mode
        self.get_filters()
        self.initialization()

    def get_filters(self):
        wavelet = pywt.Wavelet(self.wavename)
        band_low = torch.tensor(wavelet.rec_lo)
        band_high = torch.tensor(wavelet.rec_hi)
        length_band = band_low.size()[0]
        self.kernel_size = length_band if self.kernel_size == None else self.kernel_size
        assert self.kernel_size >= length_band
        a = (self.kernel_size - length_band) // 2
        b = - (self.kernel_size - length_band - a)
        b = None if b == 0 else b
        self.filt_low = torch.zeros(self.kernel_size)
        self.filt_high = torch.zeros(self.kernel_size)
        self.filt_low[a:b] = band_low
        self.filt_high[a:b] = band_high

    def initialization(self):
        self.filter_low = self.filt_low[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        self.filter_high = self.filt_high[None, None, :].repeat((self.out_channels, self.in_channels // self.groups, 1))
        if torch.cuda.is_available():
            self.filter_low = self.filter_low.cuda()
            self.filter_high = self.filter_high.cuda()
        if self.trainable:
            self.filter_low = nn.Parameter(self.filter_low)
            self.filter_high = nn.Parameter(self.filter_high)
        if self.kernel_size % 2 == 0:
            self.pad_sizes = [self.kernel_size // 2 - 1, self.kernel_size // 2 - 1]
        else:
            self.pad_sizes = [self.kernel_size // 2, self.kernel_size // 2]

    def forward(self, input):
        assert isinstance(input, torch.Tensor)
        assert len(input.size()) == 3
        assert input.size()[1] == self.in_channels
        input = F.pad(input, pad=self.pad_sizes, mode=self.pad_type)
        return F.conv1d(input, self.filter_low.to(input.device), stride=self.stride, groups=self.groups), \
               F.conv1d(input, self.filter_high.to(input.device), stride=self.stride, groups=self.groups)


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
    def __init__(self, h: VocoderConfig, input_size: int = 80, top_k: int = 4):
        super(Generator, self).__init__()

        self.num_kernels: int = len(h["resblock_kernel_sizes"])
        self.num_upsamples: int = len(h["upsample_rates"])
        self.upsample_rates: List[int] = h["upsample_rates"]
        self.up_kernels: List[int] = h["upsample_kernel_sizes"]
        self.cond_level: int = self.num_upsamples - top_k
        upsample_initial_channel = h["upsample_initial_channel"]
        self.conv_pre: nn.Module = weight_norm(Conv1d(input_size, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock

        self.ups = nn.ModuleList()
        self.cond_up = nn.ModuleList()
        self.res_output = nn.ModuleList()
        upsample_: int = 1
        kr: int = 80

        for i, (u, k) in enumerate(zip(self.upsample_rates, self.up_kernels)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

            if i > (self.num_upsamples - top_k):
                self.res_output.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=u, mode='nearest'),
                        weight_norm(nn.Conv1d(upsample_initial_channel // (2 ** i),
                                              upsample_initial_channel // (2 ** (i + 1)), 1))
                    )
                )
            if i >= (self.num_upsamples - top_k):
                self.cond_up.append(
                    weight_norm(
                        ConvTranspose1d(kr, upsample_initial_channel // (2 ** i),
                                        self.up_kernels[i - 1], self.upsample_rates[i - 1],
                                        padding=(self.up_kernels[i - 1] - self.upsample_rates[i - 1]) // 2))
                )
                kr = upsample_initial_channel // (2 ** i)

            upsample_ *= u

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h["resblock_kernel_sizes"], h["resblock_dilation_sizes"])):
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


class SpecDiscriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann", use_spectral_norm=False):
        super(SpecDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length)
        self.discriminators = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, kernel_size=(3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1,2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1,1), padding=(1, 1))),
        ])

        self.out = norm_f(nn.Conv2d(32, 1, 3, 1, 1))

    def forward(self, y):

        fmap = []
        with torch.no_grad():
            y = y.squeeze(1)
            y = stft(y, self.fft_size, self.shift_size, self.win_length, self.window.to(y.get_device()))
        y = y.unsqueeze(1)
        for i, d in enumerate(self.discriminators):
            y = d(y)
            y = F.leaky_relu(y, LRELU_SLOPE)
            fmap.append(y)

        y = self.out(y)
        fmap.append(y)

        return torch.flatten(y, 1, -1), fmap

class MultiResSpecDiscriminator(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window"):

        super(MultiResSpecDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpecDiscriminator(fft_sizes[0], hop_sizes[0], win_lengths[0], window),
            SpecDiscriminator(fft_sizes[1], hop_sizes[1], win_lengths[1], window),
            SpecDiscriminator(fft_sizes[2], hop_sizes[2], win_lengths[2], window)
            ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_proj1 = norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.dwt_proj2 = norm_f(Conv2d(1, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.dwt_conv3 = norm_f(Conv1d(8, 1, 1))
        self.dwt_proj3 = norm_f(Conv2d(1, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))
        # 1d to 2d
        b, c, t = x_d1.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d1 = F.pad(x_d1, (0, n_pad), "reflect")
            t = t + n_pad
        x_d1 = x_d1.view(b, c, t // self.period, self.period)

        x_d1 = self.dwt_proj1(x_d1)

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))
        # 1d to 2d
        b, c, t = x_d2.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d2 = F.pad(x_d2, (0, n_pad), "reflect")
            t = t + n_pad
        x_d2 = x_d2.view(b, c, t // self.period, self.period)

        x_d2 = self.dwt_proj2(x_d2)

        # DWT 3

        x_d3_high1, x_d3_low1 = self.dwt1d(x_d2_high1)
        x_d3_high2, x_d3_low2 = self.dwt1d(x_d2_low1)
        x_d3_high3, x_d3_low3 = self.dwt1d(x_d2_high2)
        x_d3_high4, x_d3_low4 = self.dwt1d(x_d2_low2)
        x_d3 = self.dwt_conv3(
            torch.cat([x_d3_high1, x_d3_low1, x_d3_high2, x_d3_low2, x_d3_high3, x_d3_low3, x_d3_high4, x_d3_low4],
                      dim=1))
        # 1d to 2d
        b, c, t = x_d3.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x_d3 = F.pad(x_d3, (0, n_pad), "reflect")
            t = t + n_pad
        x_d3 = x_d3.view(b, c, t // self.period, self.period)

        x_d3 = self.dwt_proj3(x_d3)

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        i = 0
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)

            fmap.append(x)
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            elif i == 1:
                x = torch.cat([x, x_d2], dim=2)
            elif i == 2:
                x = torch.cat([x, x_d3], dim=2)
            else:
                x = x
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ResWiseMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(ResWiseMultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 128, 15, 1, padding=7))
        self.dwt_conv2 = norm_f(Conv1d(4, 128, 41, 2, padding=20))
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        # DWT 1
        x_d1_high1, x_d1_low1 = self.dwt1d(x)
        x_d1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        x_d2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        i = 0
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
            if i == 0:
                x = torch.cat([x, x_d1], dim=2)
            if i == 1:
                x = torch.cat([x, x_d2], dim=2)
            i = i + 1
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class ResWiseMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(ResWiseMultiScaleDiscriminator, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.dwt1d = DWT_1D()
        self.dwt_conv1 = norm_f(Conv1d(2, 1, 1))
        self.dwt_conv2 = norm_f(Conv1d(4, 1, 1))
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        # DWT 1
        y_hi, y_lo = self.dwt1d(y)
        y_1 = self.dwt_conv1(torch.cat([y_hi, y_lo], dim=1))
        x_d1_high1, x_d1_low1 = self.dwt1d(y_hat)
        y_hat_1 = self.dwt_conv1(torch.cat([x_d1_high1, x_d1_low1], dim=1))

        # DWT 2
        x_d2_high1, x_d2_low1 = self.dwt1d(y_hi)
        x_d2_high2, x_d2_low2 = self.dwt1d(y_lo)
        y_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        x_d2_high1, x_d2_low1 = self.dwt1d(x_d1_high1)
        x_d2_high2, x_d2_low2 = self.dwt1d(x_d1_low1)
        y_hat_2 = self.dwt_conv2(torch.cat([x_d2_high1, x_d2_low1, x_d2_high2, x_d2_low2], dim=1))

        for i, d in enumerate(self.discriminators):

            if i == 1:
                y = y_1
                y_hat = y_hat_1
            if i == 2:
                y = y_2
                y_hat = y_hat_2

            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
