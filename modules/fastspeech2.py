import torch

from torch import nn, Tensor, LongTensor

from typing import TypedDict, Literal, Optional, Union

import fregan
import hifigan
from modules.alignment import AlignmentModule, viterbi_decode, average_by_duration
# from modules.gaussian_upsampling import GaussianUpsampling
from modules.length_regulator import LengthRegulator
from modules.tacotron2.decoder import Postnet
from modules.conformer.encoder import Encoder as ConformerEncoder
from modules.transformer.encoder import Encoder as TransformerEncoder, EncoderConfig
from modules.variance_predictor import VariancePredictor, VariancePredictorConfig
from text import phoneme_symbols, accent_symbols
from utils.mask import make_non_pad_mask, make_pad_mask


class VarianceEmbedding(TypedDict):
    kernel_size: int
    dropout: float
    pitch_embedding_type: Literal["normal", "fastpitch"]
    n_bins: Optional[int]


VocoderType = Literal["fregan", "hifigan"]
VocoderGenerator = Union[fregan.Generator, hifigan.Generator]


class ModelConfig(TypedDict):
    variance_encoder_type: Literal["transformer", "conformer"]
    variance_encoder: EncoderConfig
    phoneme_encoder_type: Literal["transformer", "conformer"]
    phoneme_encoder: EncoderConfig
    decoder_type: Literal["transformer", "conformer"]
    decoder: EncoderConfig
    variance_predictor: VariancePredictorConfig
    variance_embedding: VarianceEmbedding
    vocoder_type: VocoderType


class BaseModule(nn.Module):
    def source_mask(self, ilens: LongTensor) -> Tensor:
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        # x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)


class FastSpeech2(BaseModule):
    def __init__(self, model_config: ModelConfig, speaker_num: int, pitch_min: float, pitch_max: float):
        super(FastSpeech2, self).__init__()

        padding_idx = 0
        variance_hidden = model_config["variance_encoder"]["hidden"]
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=variance_hidden,
            padding_idx=padding_idx
        )
        self.accent_embedding = nn.Embedding(
            num_embeddings=len(accent_symbols),
            embedding_dim=variance_hidden,
            padding_idx=padding_idx
        )
        self.speaker_embedding = nn.Embedding(
            num_embeddings=speaker_num,
            embedding_dim=variance_hidden,
        )

        variance_encoder_type = model_config["variance_encoder_type"]
        if variance_encoder_type == "conformer":
            self.variance_encoder = ConformerEncoder(model_config["variance_encoder"])
        elif variance_encoder_type == "transformer":
            self.variance_encoder = TransformerEncoder(model_config["variance_encoder"])
        else:
            raise ValueError("unknown variance encoder: " + variance_encoder_type)

        self.duration_predictor = VariancePredictor(
            variance_hidden, model_config["variance_predictor"]
        )
        self.pitch_predictor = VariancePredictor(
            variance_hidden, model_config["variance_predictor"]
        )

        kernel_size = model_config["variance_embedding"]["kernel_size"]
        dropout = model_config["variance_embedding"]["dropout"]

        decoder_hidden = model_config["phoneme_encoder"]["hidden"]
        self.decoder_phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=decoder_hidden,
            padding_idx=padding_idx
        )
        self.decoder_speaker_embedding = nn.Embedding(
            num_embeddings=speaker_num,
            embedding_dim=decoder_hidden,
        )

        self.extractor = PitchAndDurationExtractor(model_config)

        self.pitch_embedding_type = model_config["variance_embedding"]["pitch_embedding_type"]
        if self.pitch_embedding_type == "normal":
            assert model_config["variance_embedding"]["n_bins"], "please specify n_bins"
            n_bins: int = model_config["variance_embedding"]["n_bins"]
            self.pitch_embedding = nn.Embedding(
                n_bins, variance_hidden
            )
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        else:
            # fastpitch style
            self.pitch_embedding = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=1,
                    out_channels=variance_hidden,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Dropout(dropout),
            )

        phoneme_encoder_type = model_config["phoneme_encoder_type"]
        if phoneme_encoder_type == "conformer":
            self.phoneme_encoder = ConformerEncoder(model_config["phoneme_encoder"])
        elif phoneme_encoder_type == "transformer":
            self.phoneme_encoder = TransformerEncoder(model_config["phoneme_encoder"])
        else:
            raise ValueError("unknown phoneme encoder: " + phoneme_encoder_type)

        # self.length_regulator = GaussianUpsampling()
        self.length_regulator = LengthRegulator()

        decoder_type = model_config["decoder_type"]
        if decoder_type == "conformer":
            self.decoder = ConformerEncoder(model_config["decoder"])
        elif decoder_type == "transformer":
            self.decoder = TransformerEncoder(model_config["decoder"])
        else:
            raise ValueError("unknown decoder: " + decoder_type)

        self.mel_channels = 80
        self.mel_linear = nn.Linear(variance_hidden, self.mel_channels)
        self.postnet = Postnet()

    @staticmethod
    def bucketize(tensor: Tensor, bucket_boundaries: Tensor) -> LongTensor:
        """for onnx, https://github.com/ming024/FastSpeech2/issues/98#issuecomment-952490935"""
        result = torch.zeros_like(tensor, dtype=torch.int32)
        for boundary in bucket_boundaries:
            result += (tensor > boundary).int()
        return result.long()

    def source_mask(self, ilens: LongTensor) -> Tensor:
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        # x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)

    def forward(
        self,
        phonemes: Tensor,
        pitches: Tensor,
        accents: Tensor,
        speakers: Tensor,
        phoneme_lens: Optional[LongTensor] = None,
        max_phoneme_len: Optional[LongTensor] = None,
        mels: Optional[Tensor] = None,
        mel_lens: Optional[LongTensor] = None,
    ):
        # forward encoder
        if phoneme_lens is None:
            x_masks = torch.ones_like(phonemes).squeeze(0)
        else:
            # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])
            x_masks = self.source_mask(phoneme_lens)

        phoneme_embedding = self.phoneme_embedding(phonemes)
        accent_embedding = self.accent_embedding(accents)

        # predict pitches with a phoneme and an accent
        pitches_args = self.__forward_preprocessing(
            phonemes, speakers, phoneme_embedding + accent_embedding, x_masks, phoneme_lens, max_phoneme_len)
        pred_avg_pitches: Tensor = self.pitch_predictor(
            pitches_args[0], pitches_args[1].unsqueeze(-1))

        # predict log_durations with a phoneme
        log_durations_args = self.__forward_preprocessing(
            phonemes, speakers, phoneme_embedding, x_masks, phoneme_lens, max_phoneme_len)
        pred_log_durations: Tensor = self.duration_predictor(
            log_durations_args[0], log_durations_args[1].unsqueeze(-1))

        x = self.decoder_phoneme_embedding(phonemes)

        feature_embedded, _ = self.phoneme_encoder(x, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        avg_pitches, durations, log_p_attn, bin_loss = None, None, None, None
        embedding_pitches = pitches
        if mels is not None:
            avg_pitches, durations, log_p_attn, bin_loss = self.extractor(
                hs=feature_embedded,
                pitches=pitches,
                mels=mels,
                phoneme_lens=phoneme_lens,
                mel_lens=mel_lens,
            )
            embedding_pitches = avg_pitches

        if max_phoneme_len is None:
            feature_embedded = feature_embedded + self.decoder_speaker_embedding(speakers).unsqueeze(1).expand(
                -1, phonemes.shape[1], -1
            )
        else:
            feature_embedded = feature_embedded + self.decoder_speaker_embedding(speakers).unsqueeze(1).expand(
                -1, max_phoneme_len, -1
            )

        if self.pitch_embedding_type == "normal":
            pitch_embeds = self.pitch_embedding(self.bucketize(embedding_pitches, self.pitch_bins))
        else:
            # fastpitch style
            pitch_embeds = self.pitch_embedding(embedding_pitches.transpose(1, 2)).transpose(1, 2)

        feature_embedded += pitch_embeds
        length_regulated = self.length_regulator(feature_embedded)

        if mel_lens is not None:
            h_masks = self.source_mask(mel_lens)
        else:
            h_masks = None

        outputs, _ = self.decoder(length_regulated, h_masks)
        outputs = self.mel_linear(outputs).view(
            outputs.size(0), -1, self.mel_channels
        )  # (B, T_feats, odim)

        postnet_outputs = outputs + self.postnet(
            outputs.transpose(1, 2)
        ).transpose(1, 2)

        return pred_avg_pitches, pred_log_durations, avg_pitches, durations, log_p_attn, bin_loss, outputs, postnet_outputs

    def __forward_preprocessing(
        self,
        phonemes: Tensor,
        speakers: Tensor,
        x: Tensor,
        x_masks: Tensor,
        phoneme_lens: Optional[LongTensor] = None,
        max_phoneme_len: Optional[LongTensor] = None,
    ):
        hs, _ = self.variance_encoder(x, x_masks)    # (B, Tmax, adim) -> torch.Size()

        if max_phoneme_len is None:
            hs = hs + self.speaker_embedding(speakers).unsqueeze(1).expand(
                -1, phonemes.shape[1], -1
            )
        else:
            hs = hs + self.speaker_embedding(speakers).unsqueeze(1).expand(
                -1, max_phoneme_len, -1
            )

        # forward duration predictor and variance predictors
        if phoneme_lens is None:
            d_masks = ~torch.ones_like(phonemes).to(
                device=x.device, dtype=torch.bool)
        else:
            d_masks = make_pad_mask(phoneme_lens).to(x.device)

        return hs, d_masks


class PitchAndDurationExtractor(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super(PitchAndDurationExtractor, self).__init__()
        hidden = model_config["phoneme_encoder"]["hidden"]

        self.alignment_module = AlignmentModule(hidden, 80)

    def forward(self, hs: Tensor, pitches: Tensor, mels: Tensor, phoneme_lens: LongTensor, mel_lens: LongTensor):
        h_masks = make_pad_mask(phoneme_lens).to(hs.device)
        log_p_attn = self.alignment_module(hs, mels, h_masks)
        durations, bin_loss = viterbi_decode(log_p_attn, phoneme_lens, mel_lens)
        avg_pitches = average_by_duration(durations, pitches.squeeze(-1), phoneme_lens, mel_lens).unsqueeze(-1)
        return avg_pitches, durations, log_p_attn, bin_loss

