# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple, Optional

import torch
from torch import nn, Tensor, LongTensor

from utils.mask import make_non_pad_mask


class FastSpeech2Loss(nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self):
        super().__init__()
        # define criterions
        self.l1_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()

    def forward(
        self,
        outputs: Tensor,
        postnet_outputs: Tensor,
        log_duration_outputs: Tensor,
        pitch_outputs: Tensor,
        mel_targets: Tensor,
        duration_targets: LongTensor,
        pitch_targets: Tensor,
        input_lens: LongTensor,
        output_lens: LongTensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of outputs (B, T_feats, odim).
            postnet_outputs (Tensor): Batch of outputs after postnet (B, T_feats, odim).
            log_duration_outputs (Tensor): Batch of outputs of duration predictor (B, T_text).
            pitch_outputs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            mel_targets (Tensor): Batch of target mel-spectrogram (B, T_feats, odim).
            duration_targets (LongTensor): Batch of durations (B, T_text).
            pitch_targets (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            input_lens (LongTensor): Batch of the lengths of each input (B,).
            output_lens (LongTensor): Batch of the lengths of each target (B,).

        Returns:
            Tensor: Total loss value.
            Tensor: Mel-spectrogram loss value
            Tensor: Mel-spectrogram with Postnet loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.

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
        pitch_targets = pitch_targets.masked_select(pitch_masks)

        # calculate loss
        mel_loss = self.l1_criterion(outputs, mel_targets)
        postnet_mel_loss = self.l1_criterion(postnet_outputs, mel_targets)
        duration_loss = self.mse_criterion(log_duration_outputs, log_duration_targets)
        pitch_loss = self.mse_criterion(pitch_outputs, pitch_targets)

        total_loss = mel_loss + postnet_mel_loss + duration_loss + pitch_loss

        return total_loss, mel_loss, postnet_mel_loss, duration_loss, pitch_loss
