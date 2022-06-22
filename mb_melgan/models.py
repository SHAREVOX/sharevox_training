import torch
import torch.nn as nn

from mb_melgan.pqmf import PQMF

MAX_WAV_VALUE = 32768.0


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResStack(nn.Module):
    def __init__(self, channel, dilation=1):
        super(ResStack, self).__init__()

        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1)),
        )

        self.shortcut = nn.utils.weight_norm(nn.Conv1d(channel, channel, kernel_size=1))

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.block[2])
        nn.utils.remove_weight_norm(self.block[4])
        nn.utils.remove_weight_norm(self.shortcut)
        # def _remove_weight_norm(m):
        #     try:
        #         torch.nn.utils.remove_weight_norm(m)
        #     except ValueError:  # this module didn't have weight norm
        #         return
        #
        # self.apply(_remove_weight_norm


class Generator(nn.Module):
    def __init__(self, mel_channel, n_residual_layers, ratios=[8, 5, 5], mult=256, out_band=1):
        super(Generator, self).__init__()
        self.mel_channel = mel_channel

        generator = [
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(
                nn.Conv1d(mel_channel, mult * 2, kernel_size=7, stride=1)
            ),
        ]

        # Upsample to raw audio scale
        for _, r in enumerate(ratios):
            generator += [
                nn.LeakyReLU(0.2),
                nn.utils.weight_norm(
                    nn.ConvTranspose1d(
                        mult * 2, mult, kernel_size=r * 2, stride=r, padding=r // 2 + r % 2, output_padding=r % 2
                    )
                ),
            ]
            for j in range(n_residual_layers):
                generator += [ResStack(mult, dilation=3 ** j)]

            mult //= 2

        generator += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(mult * 2, out_band, kernel_size=7, stride=1)),
            nn.Tanh(),
        ]

        self.generator = nn.Sequential(*generator)
        self.apply(weights_init)

    def forward(self, mel):
        mel = (mel + 5.0) / 5.0  # roughly normalize spectrogram
        return self.generator(mel)

    def eval(self, inference=False):
        super(Generator, self).eval()

        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        for idx, layer in enumerate(self.generator):
            if len(layer.state_dict()) != 0:
                try:
                    nn.utils.remove_weight_norm(layer)
                except:
                    layer.remove_weight_norm()

    def inference(self, mel):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(mel.device)
        mel = torch.cat((mel, zero), dim=2)

        audio = self.forward(mel)
        pqmf = PQMF().to(mel.device)
        audio = pqmf.synthesis(audio).view(-1)
        return audio
