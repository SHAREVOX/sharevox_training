import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from pytorch_wavelets import DWT1DInverse, DWT1DForward

from models.upsampler.config import UpsamplerConfig
from models.upsampler.sifigan import ResBlock1, ResBlock2, AdaptiveResBlock
from models.upsampler.univnet.models import DiscriminatorR
from models.upsampler.utils import init_weights, get_padding, LRELU_SLOPE
from models.upsampler.sfregan2 import onnx_wavelets


class Generator(nn.Module):
    def __init__(self, h: UpsamplerConfig, input_size: int = 80, gin_channels: int = 0,multi_idwt: bool = False, onnx: bool = False,):
        super(Generator, self).__init__()
        self.num_kernels = len(h["filter_resblock_kernel_sizes"])
        self.num_upsamples = len(h["upsample_rates"])
        self.share_upsamples = h["share_upsamples"]
        self.share_downsamples = h["share_downsamples"]
        upsample_initial_channel = h["upsample_initial_channel"]
        self.multi_idwt = multi_idwt
        self.conv_pre = weight_norm(Conv1d(input_size, upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h["resblock"] == "1" else ResBlock2

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        # sine embedding layers
        self.sn_embedding = weight_norm(
            Conv1d(4 if multi_idwt else 2, upsample_initial_channel // (2 ** len(h["upsample_kernel_sizes"])), 7, padding=3)
        )
        self.vuv_embedding = weight_norm(
            Conv1d(4 if multi_idwt else 2, upsample_initial_channel // (2 ** len(h["upsample_kernel_sizes"])), 7, padding=3)
        )

        self.sn_ups = nn.ModuleList()
        self.fn_ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h["upsample_rates"], h["upsample_kernel_sizes"])):
            self.sn_ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2
                    )
                )
            )
            if not self.share_upsamples:
                self.fn_ups.append(
                    weight_norm(
                        ConvTranspose1d(
                            upsample_initial_channel // (2 ** i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k,
                            u,
                            padding=(k - u) // 2
                        )
                    )
                )

        self.sn_resblocks = nn.ModuleList()
        self.fn_resblocks = nn.ModuleList()
        for i in range(len(self.sn_ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
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
            self.sn_downs.append(
                weight_norm(
                    Conv1d(
                        upsample_initial_channel // (2 ** (i + 1)),
                        upsample_initial_channel // (2 ** i),
                        k,
                        u,
                        padding=u - (k % 2 == 0)
                    )
                )
            )
            if not self.share_downsamples:
                self.fn_downs.append(
                    weight_norm(
                        Conv1d(
                            upsample_initial_channel // (2 ** (i + 1)),
                            upsample_initial_channel // (2 ** i),
                            k,
                            u,
                            padding=u - (k % 2 == 0)
                        )
                    )
                )

        self.conv_post = weight_norm(Conv1d(ch, 4 if multi_idwt else 2, 7, 1, padding=3, bias=False))

        self.sn_conv_post = weight_norm(Conv1d(ch, 4 if multi_idwt else 2, 7, 1, padding=3))

        self.fn_ups.apply(init_weights)
        self.sn_ups.apply(init_weights)
        self.fn_downs.apply(init_weights)
        self.sn_downs.apply(init_weights)
        self.conv_post.apply(init_weights)
        self.sn_conv_post.apply(init_weights)

        if onnx:
            self.dwt = onnx_wavelets.DWT1DForward(J=1, mode='zero', wave='db1')
            self.idwt = onnx_wavelets.DWT1DInverse()
        else:
            self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')
            self.idwt = DWT1DInverse()

    def idwt_forward(self, x):
        if self.multi_idwt:
            x_low_low, x_low_high, x_high_low, x_high_high = x.chunk(4, dim=1)

            x_low = self.idwt([x_low_low, [x_low_high]])
            x_high = self.idwt([x_high_low, [x_high_high]])
        else:
            x_low, x_high = x.chunk(2, dim=1)

        return self.idwt([x_low, [x_high]])

    def forward(self, x, f0, vuv, d, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        e = x

        # source-network forward
        if self.multi_idwt:
            f0A, f0C = self.dwt(f0)
            f0AA, f0AC = self.dwt(f0A)
            f0CA, f0CC = self.dwt(f0C[0])
            f0 = torch.cat([f0AA, f0AC[0], f0CA, f0CC[0]], dim=1)

            vuvA, vuvC = self.dwt(vuv)
            vuvAA, vuvAC = self.dwt(vuvA)
            vuvCA, vuvCC = self.dwt(vuvC[0])
            vuv = torch.cat([vuvAA, vuvAC[0], vuvCA, vuvCC[0]], dim=1)
        else:
            f0A, f0C = self.dwt(f0)
            f0 = torch.cat([f0A, f0C[0]], dim=1)

            vuvA, vuvC = self.dwt(vuv)
            vuv = torch.cat([vuvA, vuvC[0]], dim=1)

        f0 = self.sn_embedding(f0)
        vuv = self.vuv_embedding(vuv)
        f0 = f0 + vuv

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
        e_ = self.idwt_forward(e_)

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
                    xs = self.fn_resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.fn_resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, LRELU_SLOPE)

        x = self.conv_post(x)
        x = torch.tanh(x)

        x = self.idwt_forward(x)

        return x, e_

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.fn_ups:
            remove_weight_norm(l)
        for l in self.sn_resblocks:
            l.remove_weight_norm()


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        # 1x1 conv for residual connection
        self.conv_pre = nn.ModuleList([
            norm_f(Conv2d(2, 32, (1, 1), (1, 1), padding=(0, 0))),
            norm_f(Conv2d(4, 128, (1, 1), (1, 1), padding=(0, 0))),
            norm_f(Conv2d(8, 512, (1, 1), (1, 1), padding=(0, 0)))

        ])

        # Discriminator
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(32, 128, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(128, 512, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def forward(self, x):
        # DWT and channel-wise concat

        yA, yC= self.dwt(x)
        yAA, yAC = self.dwt(yA)
        yCA, yCC = self.dwt(yC[0])
        yAAA, yAAC = self.dwt(yAA)
        yACA, yACC = self.dwt(yAC[0])
        yCAA, yCAC = self.dwt(yCA)
        yCCA, yCCC = self.dwt(yCC[0])

        x12 = torch.cat((yA, yC[0]), dim=1)
        x6 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)
        x3 = torch.cat((yAAA, yAAC[0], yACA, yACC[0], yCAA, yCAC[0], yCCA, yCCC[0]), dim=1)

        # Reshape
        xes = []
        for xs in [x, x12, x6, x3]:
            b, c, t = xs.shape
            if t % self.period != 0:
                n_pad = self.period - (t % self.period)
                xs = F.pad(xs, (0, n_pad), "reflect")
                t = t + n_pad
            xes.append(xs.view(b, c, t // self.period, self.period))

        x, x12, x6, x3 = xes

        fmap = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i < 3:
                fmap.append(x)
                # residual connection
                res = self.conv_pre[i](xes[i + 1])
                res = F.leaky_relu(res, LRELU_SLOPE)
                x = (x + res) / torch.sqrt(torch.tensor(2.))
            else:
                fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        # 1x1 convolutions for residual connections
        self.conv_pre0 = nn.ModuleList([
            norm_f(Conv1d(2, 128, 1, 1, padding=0)),
            norm_f(Conv1d(4, 256, 1, 1, padding=0)),
            norm_f(Conv1d(8, 512, 1, 1, padding=0)),
        ])

        self.conv_pre1 = nn.ModuleList([
            norm_f(Conv1d(4, 128, 1, 1, padding=0)),
            norm_f(Conv1d(8, 256, 1, 1, padding=0)),
        ])

        self.conv_pre2 = nn.ModuleList([
            norm_f(Conv1d(8, 128, 1, 1, padding=0)),
        ])

        # CNNs for discriminator
        self.convs0 = self._make_layers(1, norm_f)
        self.convs1 = self._make_layers(2, norm_f)
        self.convs2 = self._make_layers(4, norm_f)

        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def _make_layers(self, input_channel, norm_f):
        conv_list = nn.ModuleList([
            norm_f(Conv1d(input_channel, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        return conv_list

    def forward(self, x, num_dis):

        if num_dis == 0:
            yA, yC = self.dwt(x)
            yAA, yAC = self.dwt(yA)
            yCA, yCC = self.dwt(yC[0])
            yAAA, yAAC = self.dwt(yAA)
            yACA, yACC = self.dwt(yAC[0])
            yCAA, yCAC = self.dwt(yCA)
            yCCA, yCCC = self.dwt(yCC[0])

            x12 = torch.cat((yA, yC[0]), dim=1)
            x6 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)
            x3 = torch.cat((yAAA, yAAC[0], yACA, yACC[0], yCAA, yCAC[0], yCCA, yCCC[0]), dim=1)

            xes = [x12, x6, x3]
            fmap = []
            for i, l in enumerate(self.convs0):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2, 3]:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre0[i - 1](xes[i - 1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        elif num_dis == 1:
            yA, yC = self.dwt(x)
            yAA, yAC = self.dwt(yA)
            yCA, yCC = self.dwt(yC[0])

            x6 = torch.cat((yA, yC[0]), dim=1)
            x3 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)

            xes = [x6, x3]
            fmap = []
            for i, l in enumerate(self.convs1):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2]:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre1[i-1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        else:
            yA, yC = self.dwt(x)
            x3 = torch.cat((yA, yC[0]), dim=1)

            xes = [x3]
            fmap = []
            for i, l in enumerate(self.convs2):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i == 1:
                    # residual connection
                    fmap.append(x)
                    res = self.conv_pre2[i - 1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
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
        self.dwt = DWT1DForward(J=1, mode='zero', wave='db1')

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        # DWT and channel-wise concat
        yA, yC= self.dwt(y)
        yAA, yAC = self.dwt(yA)
        yCA, yCC = self.dwt(yC[0])

        y_down2 = torch.cat((yA, yC[0]), dim=1)
        y_down4 = torch.cat((yAA, yAC[0], yCA, yCC[0]), dim=1)

        yA_hat, yC_hat = self.dwt(y_hat)
        yAA_hat, yAC_hat = self.dwt(yA_hat)
        yCA_hat, yCC_hat = self.dwt(yC_hat[0])

        yhat_down2 = torch.cat((yA_hat, yC_hat[0]), dim=1)
        yhat_down4 = torch.cat((yAA_hat, yAC_hat[0], yCA_hat, yCC_hat[0]), dim=1)

        input_dic = {0: [y, y_hat], 1: [y_down2, yhat_down2], 2: [y_down4, yhat_down4]}
        for i, d in enumerate(self.discriminators_s):
            y_d_r, fmap_r = d(input_dic[i][0], i)
            y_d_g, fmap_g = d(input_dic[i][1], i)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        for d in self.discriminators_p:
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
