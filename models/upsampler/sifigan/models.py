import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d, AvgPool1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from models.upsampler.config import UpsamplerConfig
from models.upsampler.univnet.models import DiscriminatorR
from models.upsampler.utils import index_initial, init_weights, get_padding, LRELU_SLOPE, pd_indexing


class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=d,
                    padding=get_padding(kernel_size, d)
                )
            )
            for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            )
            for _ in dilation
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class CustomConv1d(nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(CustomConv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(CustomConv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class AdaptiveResBlock(nn.Module):
    def __init__(self, channels=512, dilations=(1, 2, 4)):
        """Initialize ResidualBlock module.
        Args:
            kernel_size (int): Kernel size of dilation convolution layer.
            channels (int): Number of channels for convolution layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
        """
        super().__init__()
        self.channels = channels
        self.dilations = dilations
        self.nonlinears = nn.ModuleList()
        self.convsC = nn.ModuleList()
        self.convsP = nn.ModuleList()
        self.convsF = nn.ModuleList()
        self.convsA = nn.ModuleList()
        for _ in dilations:
            self.convsC.append(
                Conv1d1x1(channels, channels)
            )
            self.convsP.append(
                Conv1d1x1(channels, channels)
            )
            self.convsF.append(
                Conv1d1x1(channels, channels)
            )
            self.convsA.append(
                weight_norm(
                    Conv1d(channels, channels, kernel_size=3, dilation=1, bias=True, padding=1)
                )
            )

    def forward(self, x, d):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, channels, T).
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        Returns:
            Tensor: Output tensor (B, channels, T).
        """
        batch_index, ch_index = index_initial(x.size(0), self.channels)
        batch_index = batch_index.to(x.device)
        ch_index = ch_index.to(x.device)
        for i, dilation in enumerate(self.dilations):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xP, xF = pd_indexing(xt, d, dilation, batch_index, ch_index)
            xt = self.convsC[i](xt) + self.convsP[i](xP) + self.convsF[i](xF)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = self.convsA[i](xt)
            x = xt + x
        return x


class Generator(nn.Module):
    def __init__(self, h: UpsamplerConfig, input_size: int = 80, gin_channels: int = 0):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h["filter_resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.share_upsamples = h["share_upsamples"]
        self.share_downsamples = h["share_downsamples"]
        upsample_initial_channel = h["upsample_initial_channel"]
        self.conv_pre = weight_norm(Conv1d(input_size, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == '1' else ResBlock2

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # sine embedding layers
        self.sn_embedding = weight_norm(Conv1d(1, upsample_initial_channel // (2 ** len(h["upsample_kernel_sizes"])), 7, padding=3))

        self.sn_ups = nn.ModuleList()
        self.fn_ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.sn_ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

            if not self.share_upsamples:
                self.fn_ups.append(weight_norm(
                    ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))

        self.sn_resblocks = nn.ModuleList()
        self.fn_resblocks = nn.ModuleList()
        for i in range(len(self.sn_ups)):
            ch = upsample_initial_channel//(2**(i+1))
            self.sn_resblocks.append(
                AdaptiveResBlock(
                    channels=ch,
                    dilations=h["source_resblock_dilation_sizes"][i],
                )
            )
            for j, (k, d) in enumerate(zip(h["filter_resblock_kernel_sizes"], h["filter_resblock_dilation_sizes"])):
                self.fn_resblocks.append(resblock(h, ch, k, d))

        self.sn_downs = nn.ModuleList()
        self.fn_downs = nn.ModuleList()

        for i, (u, k) in reversed(list(enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])))):
            if i == 0:
                break
            self.sn_downs.append(weight_norm(
                Conv1d(upsample_initial_channel//(2**(i+1)), upsample_initial_channel//(2**i),
                       k, u, padding=u - (k % 2 == 0))))
            if not self.share_downsamples:
                self.fn_downs.append(weight_norm(
                    Conv1d(upsample_initial_channel//(2**(i+1)), upsample_initial_channel//(2**i),
                           k, u, padding=u - (k % 2 == 0))))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.sn_conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        self.fn_ups.apply(init_weights)
        self.sn_ups.apply(init_weights)
        self.fn_downs.apply(init_weights)
        self.sn_downs.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.sn_conv_post.apply(init_weights)

    def forward(self, x, f0, d, g=None):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, in_channels, T).
            f0 (Tensor): Input sine signal (B, 1, T).
            d (List): F0-dependent dilation factors [(B, 1, T) x num_upsamples].
            g (Tensor): Speaker Embedding
        Returns:
            Tensor: Output tensor (B, out_channels, T).
        """
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        e = x

        # source-network forward
        f0 = self.sn_embedding(f0)
        embs = [f0]
        for i in range(self.num_upsamples - 1):
            f0 = self.sn_downs[i](f0)
            f0 = F.leaky_relu(f0, LRELU_SLOPE)
            embs += [f0]
        for i in range(self.num_upsamples):
            # excitation generation network
            e = F.leaky_relu(e, LRELU_SLOPE)
            e = self.sn_ups[i](e) + embs[-i - 1]
            e = self.sn_resblocks[i](e, d[i])
        e_ = self.sn_conv_post(e)

        # filter-network forward
        embs = [e]
        for i in range(self.num_upsamples - 1):
            if self.share_downsamples:
                e = self.sn_downs[i](e)
            else:
                e = self.fn_downs[i](e)
            e = F.leaky_relu(e, LRELU_SLOPE)
            embs += [e]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            if self.share_upsamples:
                x = self.sn_ups[i](x)
            else:
                x = self.fn_ups[i](x)
            x = x + embs[-i - 1]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.fn_resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.fn_resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x, e_

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.sn_ups:
            remove_weight_norm(l)
        for l in self.fn_ups:
            remove_weight_norm(l)
        for l in self.sn_downs:
            remove_weight_norm(l)
        for l in self.fn_downs:
            remove_weight_norm(l)
        for l in self.sn_resblocks:
            l.remove_weight_norm()
        for l in self.fn_resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
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

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
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
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodAndScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodAndScaleDiscriminator, self).__init__()
        self.discriminators_p = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])
        self.discriminators_s = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators_p):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        for i, d in enumerate(self.discriminators_s):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodAndResolutionDiscriminator(nn.Module):
    def __init__(self):
        super(MultiPeriodAndResolutionDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
            DiscriminatorR(1024, 120, 600),
            DiscriminatorR(2048, 240, 1200),
            DiscriminatorR(512, 50, 240),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
