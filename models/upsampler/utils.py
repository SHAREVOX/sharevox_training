# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import ConstantPad1d
from torch.nn.functional import interpolate


LRELU_SLOPE = 0.1


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def pd_indexing(x: Tensor, d, dilation: int, batch_index: Tensor, ch_index: Tensor) -> Tuple[Tensor, Tensor]:
    """Pitch-dependent indexing of past and future samples.
    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.
        batch_index (Tensor): Batch index
        ch_index (Tensor): Channel index
    Returns:
        Tensor: Past output tensor (B, out_channels, T)
        Tensor: Future output tensor (B, out_channels, T)
    """
    (_, _, batch_length) = d.size()
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)
    
    # get past index (assume reflect padding)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)

    # get future index (assume reflect padding)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T)
    idxF = (batch_index, ch_index, idxF)
    return x[idxP], x[idxF]


def index_initial(n_batch: int, n_ch: int):
    """Tensor batch and channel index initialization.
    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array
    Returns:
        Tensor: Batch index
        Tensor: Channel index
    """
    batch_index = []
    for i in range(n_batch):
        batch_index.append([[i]] * n_ch)
    ch_index = []
    for i in range(n_ch):
        ch_index += [[i]]
    ch_index = [ch_index] * n_batch

    batch_index = torch.tensor(batch_index)
    ch_index = torch.tensor(ch_index)
    return batch_index, ch_index


def dilated_factor(batch_f0: Tensor, fs: int, dense_factor: int) -> Tensor:
    """Pitch-dependent dilated factor
    Args:
        batch_f0 (Tensor): the f0 sequence (T)
        fs (int): sampling rate
        dense_factor (int): the number of taps in one cycle
    Return:
        dilated_factors(Tensor):
            float array of the pitch-dependent dilated factors (T)
    """
    batch_f0[batch_f0 == 0] = fs / dense_factor
    dilated_factors = torch.ones_like(batch_f0) * fs / dense_factor / batch_f0
    # assert np.all(dilated_factors > 0)

    return dilated_factors


class SignalGenerator:
    """Input signal generator module."""

    def __init__(
        self,
        sample_rate=24000,
        hop_size=120,
        sine_amp=0.1,
        noise_amp=0.003,
        signal_types=["sine"],
    ):
        """
        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of input F0.
            sine_amp (float): Sine amplitude for NSF-based sine generation.
            noise_amp (float): Noise amplitude for NSF-based sine generation.
            signal_types (list): List of input signal types for generator.
        """
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.signal_types = signal_types
        self.sine_amp = sine_amp
        self.noise_amp = noise_amp

        # for signal_type in signal_types:
        #     if not signal_type in ["noise", "sine", "sines", "uv"]:
        #         logger.info(f"{signal_type} is not supported type for generator input.")
        #         sys.exit(0)
        # logger.info(f"Use {signal_types} for generator input signals.")

    @torch.no_grad()
    def __call__(self, f0):
        signals = []
        for typ in self.signal_types:
            if "noise" == typ:
                signals.append(self.random_noise(f0))
            if "sine" == typ:
                signals.append(self.sinusoid(f0))
            if "sines" == typ:
                signals.append(self.sinusoids(f0))
            if "uv" == typ:
                signals.append(self.vuv_binary(f0))

        input_batch = signals[0]
        for signal in signals[1:]:
            input_batch = torch.cat([input_batch, signal], axis=1)

        return input_batch

    @torch.no_grad()
    def random_noise(self, f0):
        """Calculate noise signals.
        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).
        Returns:
            Tensor: Gaussian noise signals (B, 1, T).
        """
        B, _, T = f0.size()
        noise = torch.randn((B, 1, T * self.hop_size), device=f0.device)

        return noise

    @torch.no_grad()
    def sinusoid(self, f0):
        """Calculate sine signals.
        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).
        Returns:
            Tensor: Sines generated following NSF (B, 1, T).
        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        radious = (interpolate(f0, T * self.hop_size) / self.sample_rate) % 1
        sine = vuv * torch.sin(torch.cumsum(radious, dim=2) * 2 * math.pi) * self.sine_amp
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sine = sine + noise

        return sine

    @torch.no_grad()
    def sinusoids(self, f0):
        """Calculate sines.
        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).
        Returns:
            Tensor: Sines generated following NSF (B, 1, T).
        """
        B, _, T = f0.size()
        vuv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)
        f0 = interpolate(f0, T * self.hop_size)
        sines = torch.zeros_like(f0, device=f0.device)
        harmonics = 5  # currently only fixed number of harmonics is supported
        for i in range(harmonics):
            radious = (f0 * (i + 1) / self.sample_rate) % 1
            sines += torch.sin(torch.cumsum(radious, dim=2) * 2 * math.pi)
        sines = self.sine_amp * sines * vuv / harmonics
        if self.noise_amp > 0:
            noise_amp = vuv * self.noise_amp + (1.0 - vuv) * self.noise_amp / 3.0
            noise = torch.randn((B, 1, T * self.hop_size), device=f0.device) * noise_amp
            sines = sines + noise

        return sines

    @torch.no_grad()
    def vuv_binary(self, f0):
        """Calculate V/UV binary sequences.
        Args:
            f0 (Tensor): F0 tensor (B, 1, T // hop_size).
        Returns:
            Tensor: V/UV binary sequences (B, 1, T).
        """
        _, _, T = f0.size()
        uv = interpolate((f0 > 0) * torch.ones_like(f0), T * self.hop_size)

        return uv


class AdaptiveWindowing(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(
        self,
        sample_rate,
        hop_size,
        fft_size,
        f0_floor,
        f0_ceil,
    ):
        """Initilize AdaptiveWindowing module.
        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
        """
        super(AdaptiveWindowing, self).__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.register_buffer("window", torch.zeros((f0_ceil + 1, fft_size)))
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        # Pre-calculation of the window functions
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sample_rate / f0)
            base_index = torch.arange(
                -half_win_len, half_win_len + 1, dtype=torch.int64
            )
            position = base_index / 1.5 / self.sample_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window = torch.zeros(fft_size)
            window[left:right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = window / average

    def forward(self, x, f, power=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Waveform (B, fft_size // 2 + 1, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude.
        Returns:
            Tensor: Power spectrogram (B, bin_size, T').
        """
        # Get the matrix of window functions corresponding to F0
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f]

        # Adaptive windowing and calculate power spectrogram.
        # In test, change x[:, : -1, :] to x.
        x = torch.abs(torch.fft.rfft(x[:, :-1, :] * windows))
        x = x.pow(2) if power else x

        return x


class AdaptiveLiftering(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(
        self,
        sample_rate,
        fft_size,
        f0_floor,
        f0_ceil,
        q1=-0.15,
    ):
        """Initilize AdaptiveLiftering module.
        Args:
            sample_rate (int): Sampling rate.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            q1 (float): Parameter to remove effect of adjacent harmonics.
        """
        super(AdaptiveLiftering, self).__init__()

        self.sample_rate = sample_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.register_buffer(
            "smoothing_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )
        self.register_buffer(
            "compensation_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )

        # Pre-calculation of the smoothing lifters and compensation lifters
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = torch.arange(1, self.bin_size) / sample_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (
                math.pi * f0 * quefrency
            )
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(
                2.0 * math.pi * f0 * quefrency
            )
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f, elim_0th=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Power spectrogram (B, bin_size, T').
            f (Tensor): F0 sequence (B, T').
            elim_0th (bool): Whether to eliminate cepstram 0th component.
        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').
        """
        # Setting the smoothing lifter and compensation lifter
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]

        # Calculating cepstrum
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(torch.clamp(tmp, min=1e-7))).real

        # Set the 0th cepstrum to 0
        if elim_0th:
            cepstrum[..., 0] = 0

        # Liftering cepstrum with the lifters
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter

        # Return the result to the spectral domain
        x = torch.fft.irfft(liftered_cepstrum)[:, :, : self.bin_size]

        return x


class CheapTrick(nn.Module):
    """
    Spectral envelopes estimation module based on CheapTrick.
    References:
        - https://www.sciencedirect.com/science/article/pii/S0167639314000697
        - https://github.com/mmorise/World
    """

    def __init__(
        self,
        sample_rate,
        hop_size,
        fft_size,
        f0_floor=70,
        f0_ceil=340,
        uv_threshold=0,
        q1=-0.15,
    ):
        """Initilize AdaptiveLiftering module.
        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            uv_threshold (float): V/UV determining threshold.
            q1 (float): Parameter to remove effect of adjacent harmonics.
        """
        super(CheapTrick, self).__init__()

        # fft_size must be larger than 3.0 * sample_rate / f0_floor
        assert fft_size > 3.0 * sample_rate / f0_floor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold

        self.ada_wind = AdaptiveWindowing(
            sample_rate,
            hop_size,
            fft_size,
            f0_floor,
            f0_ceil,
        )
        self.ada_lift = AdaptiveLiftering(
            sample_rate,
            fft_size,
            f0_floor,
            f0_ceil,
            q1,
        )

    def forward(self, x, f, power=False, elim_0th=False):
        """Calculate forward propagation.
        Args:
            x (Tensor): Power spectrogram (B, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to eliminate cepstram 0th component.
        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').
        """
        # Step0: Round F0 values to integers.
        voiced = (f > self.uv_threshold) * torch.ones_like(f)
        f = voiced * f + (1.0 - voiced) * self.f0_ceil
        f = torch.round(torch.clamp(f, min=self.f0_floor, max=self.f0_ceil)).to(
            torch.int64
        )

        # Step1: Adaptive windowing and calculate power or amplitude spectrogram.
        x = self.ada_wind(x, f, power)

        # Step3: Smoothing (log axis) and spectral recovery on the cepstrum domain.
        x = self.ada_lift(x, f, elim_0th)

        return x
