from typing import List

class Config:
    def __init__(self):
        self.resblock: str = "1"
        self.num_gpus: int = 0
        self.batch_size: int = 16
        self.learning_rate: float = 0.0002
        self.adam_b1: float = 0.8
        self.adam_b2: float = 0.99
        self.lr_decay: float = 0.999
        self.seed: int = 1234
        self.disc_start_step: int = 0

        self.upsample_rates: List[int] = [8, 8, 2, 2]
        self.upsample_kernel_sizes: List[int] = [16, 16, 4, 4]
        self.upsample_initial_channel: int = 512
        self.resblock_kernel_sizes: List[int] = [3, 7, 11]
        self.resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.segment_size: int = 8192
        self.num_mels: int = 80
        self.num_freq: int = 1025
        self.n_fft: int = 1024
        self.hop_size: int = 256
        self.win_size: int = 1024

        self.fmin: int = 0
        self.fmax: int = 8000
        self.fmax_for_loss: None = None

        self.num_workers: int = 4
