# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related loss module for ESPnet2."""

from typing import Tuple, Optional, Union, Any, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor, LongTensor

from modules.jets import VocoderType
from utils.mask import make_non_pad_mask


def feature_loss(fmap_r: List[List[Tensor]], fmap_g: List[List[Tensor]]) -> Tensor:
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(
    disc_real_outputs: List[Tensor], disc_generated_outputs: List[Tensor]
) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


class FastSpeech2Loss(nn.Module):
    """Loss function module for FastSpeech2."""

    def __init__(self):
        super().__init__()
        # define criterions
        self.l1_criterion = nn.L1Loss()
        self.mse_criterion = nn.MSELoss()
        self.forward_sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

    def forward(
        self,
        log_duration_outputs: Tensor,
        pitch_outputs: Tensor,
        mel_targets: Tensor,
        duration_targets: LongTensor,
        pitch_targets: Tensor,
        attn_soft: Tensor,
        attn_hard: Tensor,
        attn_logprob: Tensor,
        input_lens: LongTensor,
        output_lens: LongTensor,
        variance_learn: bool = True,
        outputs: Optional[Tensor] = None,
        postnet_outputs: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Calculate forward propagation.

        Args:
            outputs (Optional[Tensor]): Batch of outputs (B, T_feats, odim).
            postnet_outputs (Optional[Tensor]): Batch of outputs after postnet (B, T_feats, odim).
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
        duration_masks = make_non_pad_mask(input_lens).to(mel_targets.device)
        log_duration_outputs = log_duration_outputs.squeeze(-1).masked_select(duration_masks)
        log_duration_targets = torch.log(duration_targets.float() + 1.0)
        log_duration_targets = log_duration_targets.masked_select(duration_masks)
        pitch_masks = make_non_pad_mask(input_lens).to(pitch_targets.device)
        pitch_outputs = pitch_outputs.squeeze(-1).masked_select(pitch_masks)
        pitch_targets = pitch_targets.squeeze(-1).masked_select(pitch_masks)

        # calculate loss
        duration_loss = self.mse_criterion(log_duration_outputs, log_duration_targets)
        pitch_loss = self.mse_criterion(pitch_outputs, pitch_targets)
        ctc_loss = self.forward_sum_loss(attn_logprob=attn_logprob, in_lens=input_lens, out_lens=output_lens)
        if variance_learn:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = 1.
        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight
        align_loss = ctc_loss + bin_loss

        total_loss = align_loss
        if variance_learn:
            total_loss += duration_loss + pitch_loss

        # for jets
        mel_loss, postnet_mel_loss = None, None
        if outputs is not None and postnet_outputs is not None:
            output_masks = make_non_pad_mask(output_lens).unsqueeze(-1).to(mel_targets.device)
            outputs = outputs.masked_select(output_masks)
            postnet_outputs = postnet_outputs.masked_select(output_masks)
            mel_targets = mel_targets.masked_select(output_masks)
            mel_loss = self.l1_criterion(outputs, mel_targets)
            postnet_mel_loss = self.l1_criterion(postnet_outputs, mel_targets)

            total_loss = total_loss + mel_loss + postnet_mel_loss

        return total_loss, duration_loss, pitch_loss, align_loss, mel_loss, postnet_mel_loss


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob: float = -1.0):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob: Tensor, in_lens: Tensor, out_lens: Tensor):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention: Tensor, soft_attention: Tensor):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()


class GeneratorLoss(nn.Module):
    def __init__(self, vocoder_type: VocoderType = "hifigan"):
        super().__init__()
        # define criterions
        self.l1_criterion = nn.L1Loss()
        self.vocoder_type = vocoder_type

    def forward(
        self,
        mels: Tensor,
        mel_from_outputs: Tensor,
        y_df_hat_g: List[Tensor],
        fmap_f_r: List[List[Tensor]],
        fmap_f_g: List[List[Tensor]],
        y_ds_hat_g: List[Tensor],
        fmap_s_r: List[List[Tensor]],
        fmap_s_g: List[List[Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        loss_mel = self.l1_criterion(mels, mel_from_outputs)

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        if self.vocoder_type == "fregan":
            loss_fm = (2 * (loss_fm_s + loss_fm_f))
        else:
            loss_fm = loss_fm_s + loss_fm_f
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm + (loss_mel * 45)

        return loss_gen_all, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_mel


class DiscriminatorLoss(nn.Module):
    def forward(
        self,
        y_df_hat_r: List[Tensor],
        y_df_hat_g: List[Tensor],
        y_ds_hat_r: List[Tensor],
        y_ds_hat_g: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
        loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f
        return loss_disc_all, loss_disc_s, loss_disc_f
