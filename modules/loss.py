# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple, Optional

import numpy as np
from scipy.stats import betabinom
import torch
import torch.nn.functional as F
from torch import nn, Tensor, LongTensor

from utils.mask import make_non_pad_mask


class FastSpeech2Loss(nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self):
        super().__init__()
        # define criterions
        self.l1_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.forward_sum_loss = ForwardSumLoss()

    def forward(
        self,
        outputs: Tensor,
        postnet_outputs: Tensor,
        log_duration_outputs: Tensor,
        pitch_outputs: Tensor,
        mel_targets: Tensor,
        duration_targets: LongTensor,
        pitch_targets: Tensor,
        log_p_attn: Tensor,
        input_lens: LongTensor,
        output_lens: LongTensor,
        variance_learn: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of outputs (B, T_feats, odim).
            postnet_outputs (Tensor): Batch of outputs after postnet (B, T_feats, odim).
            log_duration_outputs (Tensor): Batch of outputs of duration predictor (B, T_text).
            pitch_outputs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            mel_targets (Tensor): Batch of target mel-spectrogram (B, T_feats, odim).
            duration_targets (LongTensor): Batch of durations (B, T_text).
            pitch_targets (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            log_p_attn (Tensor): Batch of log probability of attention matrix (B, T_feats, T_text).
            input_lens (LongTensor): Batch of the lengths of each input (B,).
            output_lens (LongTensor): Batch of the lengths of each target (B,).
            variance_learn (bool): variance predictor learn or not

        Returns:
            Tensor: Total loss value.
            Tensor: Mel-spectrogram loss value
            Tensor: Mel-spectrogram with Postnet loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Forward sum loss value.

        """
        # apply mask to remove padded part
        output_masks = make_non_pad_mask(output_lens).unsqueeze(-1).to(mel_targets.device)
        outputs = outputs.masked_select(output_masks)
        postnet_outputs = postnet_outputs.masked_select(output_masks)
        mel_targets = mel_targets.masked_select(output_masks)
        duration_masks = make_non_pad_mask(input_lens).to(mel_targets.device)
        log_duration_outputs = log_duration_outputs.squeeze(-1).masked_select(duration_masks)
        log_duration_targets = torch.log(duration_targets.float() + 1.0)
        log_duration_targets = log_duration_targets.masked_select(duration_masks)
        pitch_masks = make_non_pad_mask(input_lens).to(mel_targets.device)
        pitch_outputs = pitch_outputs.squeeze(-1).masked_select(pitch_masks)
        pitch_targets = pitch_targets.squeeze(-1).masked_select(pitch_masks)

        # calculate loss
        mel_loss = self.l1_criterion(outputs, mel_targets)
        postnet_mel_loss = self.l1_criterion(postnet_outputs, mel_targets)
        duration_loss = self.mse_criterion(log_duration_outputs, log_duration_targets)
        pitch_loss = self.mse_criterion(pitch_outputs, pitch_targets)
        forward_sum_loss = self.forward_sum_loss(log_p_attn, input_lens, output_lens)
        forward_sum_loss *= 2.0  # loss scaling

        total_loss = mel_loss + postnet_mel_loss + forward_sum_loss
        if variance_learn:
            total_loss += duration_loss + pitch_loss

        return total_loss, mel_loss, postnet_mel_loss, duration_loss, pitch_loss, forward_sum_loss


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
