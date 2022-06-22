from typing import List

class Config:
    def __init__(self):
        self.n_mel_channels: int = 80
        self.segment_length: int = 16000
        self.pad_short: int = 2000
        self.filter_length: int = 1024
        self.hop_length: int = 256  # WARNING: this can't be changed.
        self.win_length: int = 1024
        self.sampling_rate: int = 48000
        self.mel_fmin: float = 0.0
        self.mel_fmax: float = 8000.0
        self.feat_match: float = 10.0
        self.lambda_adv: float = 2.5
        self.use_subband_stft_loss: bool = True
        self.feat_loss: bool = False
        self.out_channels: int = 4
        self.generator_ratio: List[int] = [8, 4, 2]  # for 256/4 hop size and 22050 sample rate
        self.mult: int = 192
        self.n_residual_layers: int = 4
        self.num_D: int = 3
        self.ndf: int = 16
        self.n_layers: int = 3
        self.downsampling_factor: int = 4
        self.disc_out: int = 512
