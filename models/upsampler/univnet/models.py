import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm

from models.upsampler.utils import LRELU_SLOPE


class DiscriminatorR(nn.Module):
    def __init__(self, n_fft: int, hop_length: int, win_length: int, use_spectral_norm=False):
        super(DiscriminatorR, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        x = F.pad(x, (int((self.n_fft - self.hop_length) / 2), int((self.n_fft - self.hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=False
        )  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super(MultiResolutionDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorR(1024, 120, 600),
            DiscriminatorR(2048, 240, 1200),
            DiscriminatorR(512, 50, 240),
        ])

    def forward(self, y, y_hat):
        ret_rs = []
        ret_gs = []
        fmap_rs = []
        fmap_gs = []
        for disc in self.discriminators:
            ret_r, fmap_r = disc(y)
            ret_g, fmap_g = disc(y_hat)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
            ret_rs.append(ret_r)
            ret_gs.append(ret_g)

        return ret_rs, ret_gs, fmap_rs, fmap_gs
