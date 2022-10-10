import torch
import torch.nn.functional as F
from torch import Tensor
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x: Tensor, C: int = 1, clip_val: float = 1e-5) -> Tensor:
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x: Tensor, C: int = 1) -> Tensor:
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize(magnitudes: Tensor) -> Tensor:
    output = dynamic_range_compression(magnitudes)
    return output


def spectral_de_normalize(magnitudes: Tensor) -> Tensor:
    output = dynamic_range_decompression(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y: Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int,
    center: bool = False
) -> Tensor:
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).T.to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    pad_size = int((n_fft - hop_size) / 2)
    y = F.pad(y.unsqueeze(1), (pad_size, pad_size), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = spec.transpose(1, 2)
    spec = spec[..., 0] ** 2 + spec[..., 1] ** 2
    spec = torch.sqrt(torch.clamp(spec, min=1e-9))

    spec = torch.matmul(spec, mel_basis[fmax_dtype_device])
    spec = spectral_normalize(spec)

    return spec.transpose(1, 2)
