import torch
from torch.nn import functional as F


from scipy.stats import betabinom
from librosa.filters import mel as librosa_mel_fn
import numpy as np

from models.upsampler.utils import CheapTrick


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l.item())
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def stft(
    x, fft_size, hop_size, win_length, window, center=True, onesided=True, power=False
):
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
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        center=center,
        onesided=onesided,
        return_complex=False,
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if power:
        return torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
    else:
        return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class ForwardSumLoss(torch.nn.Module):
    """Forwardsum loss described at https://openreview.net/forum?id=0NQwnnwAORi"""

    def __init__(self, cache_prior: bool = True):
        """Initialize forwardsum loss module.
        Args:
            cache_prior (bool): Whether to cache beta-binomial prior
        """
        super().__init__()
        self.cache_prior = cache_prior
        self._cache = {}

    def forward(
        self,
        log_p_attn: torch.Tensor,
        input_lens: torch.Tensor,
        output_lens: torch.Tensor,
        blank_prob: float = np.e ** -1,
    ) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            log_p_attn (Tensor): Batch of log probability of attention matrix
                (B, T_feats, T_text).
            input_lens (Tensor): Batch of the lengths of each input (B,).
            output_lens (Tensor): Batch of the lengths of each target (B,).
            blank_prob (float): Blank symbol probability.
        Returns:
            Tensor: forwardsum loss value.
        """
        B = log_p_attn.size(0)

        # add beta-binomial prior
        bb_prior = self._generate_prior(input_lens, output_lens)
        bb_prior = bb_prior.to(dtype=log_p_attn.dtype, device=log_p_attn.device)
        log_p_attn = log_p_attn + bb_prior

        # a row must be added to the attention matrix to account for
        #    blank token of CTC loss
        # (B,T_feats,T_text+1)
        log_p_attn_pd = F.pad(log_p_attn, (1, 0, 0, 0, 0, 0), value=np.log(blank_prob))

        loss = 0
        for bidx in range(B):
            # construct target sequnece.
            # Every text token is mapped to a unique sequnece number.
            target_seq = torch.arange(1, input_lens[bidx] + 1).unsqueeze(0)
            cur_log_p_attn_pd = log_p_attn_pd[
                                bidx, : output_lens[bidx], : input_lens[bidx] + 1
                                ].unsqueeze(
                1
            )  # (T_feats,1,T_text+1)
            loss += F.ctc_loss(
                log_probs=cur_log_p_attn_pd,
                targets=target_seq,
                input_lengths=output_lens[bidx: bidx + 1],
                target_lengths=input_lens[bidx: bidx + 1],
                zero_infinity=True,
            )
        loss = loss / B
        return loss

    def _generate_prior(self, phoneme_lens, mel_lens, w=1) -> torch.Tensor:
        """Generate alignment prior formulated as beta-binomial distribution
        Args:
            phoneme_lens (Tensor): Batch of the lengths of each input (B,).
            mel_lens (Tensor): Batch of the lengths of each target (B,).
            w (float): Scaling factor; lower -> wider the width.
        Returns:
            Tensor: Batched 2d static prior matrix (B, T_feats, T_text).
        """
        B = len(phoneme_lens)
        T_text = phoneme_lens.max()
        T_feats = mel_lens.max()

        bb_prior = torch.full((B, T_feats, T_text), fill_value=-np.inf)
        for bidx in range(B):
            T = mel_lens[bidx].item()
            N = phoneme_lens[bidx].item()

            key = str(T) + "," + str(N)
            if self.cache_prior and key in self._cache:
                prob = self._cache[key]
            else:
                alpha = w * np.arange(1, T + 1, dtype=float)  # (T,)
                beta = w * np.array([T - t + 1 for t in alpha])
                k = np.arange(N)
                batched_k = k[..., None]  # (N,1)
                prob = betabinom.logpmf(batched_k, N, alpha, beta)  # (N,T)

            # store cache
            if self.cache_prior and key not in self._cache:
                self._cache[key] = prob

            prob = torch.from_numpy(prob).transpose(0, 1)  # -> (T,N)
            bb_prior[bidx, :T, :N] = prob

        return bb_prior


class ResidualLoss(torch.nn.Module):
    """The regularization loss of hn-uSFGAN."""

    def __init__(
        self,
        sample_rate=24000,
        fft_size=2048,
        hop_size=120,
        f0_floor=100,
        f0_ceil=840,
        n_mels=80,
        fmin=0,
        fmax=None,
        power=False,
        elim_0th=True,
    ):
        """Initialize ResidualLoss module.
        Args:
            sample_rate (int): Sampling rate.
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            f0_floor (int): Minimum F0 value.
            f0_ceil (int): Maximum F0 value.
            n_mels (int): Number of Mel basis.
            fmin (int): Minimum frequency for Mel.
            fmax (int): Maximum frequency for Mel.
            power (bool): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to exclude 0th cepstrum in CheapTrick.
                If set to true, power is estimated by source-network.
        """
        super(ResidualLoss, self).__init__()
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.cheaptrick = CheapTrick(
            sample_rate=sample_rate,
            hop_size=hop_size,
            fft_size=fft_size,
            f0_floor=f0_floor,
            f0_ceil=f0_ceil,
        )
        self.win_length = fft_size
        self.register_buffer("window", torch.hann_window(self.win_length))

        # define mel-filter-bank
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate // 2
        melmat = librosa_mel_fn(sr=sample_rate, n_fft=fft_size, n_mels=n_mels, fmin=fmin, fmax=self.fmax).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

        self.power = power
        self.elim_0th = elim_0th

    def forward(self, s, y, f):
        """Calculate forward propagation.
        Args:
            s (Tensor): Predicted source excitation signal (B, 1, T).
            y (Tensor): Ground truth signal (B, 1, T).
            f (Tensor): F0 sequence (B, 1, T // hop_size).
        Returns:
            Tensor: Loss value.
        """
        s, y, f = s.squeeze(1), y.squeeze(1), f.squeeze(1)

        with torch.no_grad():
            # calculate log power (or magnitude) spectrograms
            e = self.cheaptrick.forward(y, f, self.power, self.elim_0th)
            y = stft(
                y,
                self.fft_size,
                self.hop_size,
                self.win_length,
                self.window,
                power=self.power,
            )
            # adjust length, (B, T', C)
            minlen = min(e.size(1), y.size(1))
            e, y = e[:, :minlen, :], y[:, :minlen, :]

            # calculate mean power (or magnitude) of y
            if self.elim_0th:
                y_mean = y.mean(dim=-1, keepdim=True)

            # calculate target of output source signal
            y = torch.log(torch.clamp(y, min=1e-7))
            t = (y - e).exp()
            if self.elim_0th:
                t_mean = t.mean(dim=-1, keepdim=True)
                t = y_mean / t_mean * t

            # apply mel-filter-bank and log
            t = torch.matmul(t, self.melmat)
            t = torch.log(torch.clamp(t, min=1e-7))

        # calculate power (or magnitude) spectrogram
        s = stft(
            s,
            self.fft_size,
            self.hop_size,
            self.win_length,
            self.window,
            power=self.power,
        )
        # adjust length, (B, T', C)
        minlen = min(minlen, s.size(1))
        s, t = s[:, :minlen, :], t[:, :minlen, :]

        # apply mel-filter-bank and log
        s = torch.matmul(s, self.melmat)
        s = torch.log(torch.clamp(s, min=1e-7))

        loss = F.l1_loss(s, t.detach())

        return loss
