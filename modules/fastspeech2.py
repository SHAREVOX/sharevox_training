import torch
import numpy as np

from torch import nn, Tensor, LongTensor

from typing import TypedDict, Literal, Tuple, Optional

from modules.length_regulator import LengthRegulator
from modules.tacotron2.decoder import Postnet
from modules.transformer.encoder import Encoder, EncoderConfig
from modules.variance_predictor import VariancePredictor, VariancePredictorConfig
from text import phoneme_symbols, accent_symbols
from utils.mask import make_non_pad_mask, make_pad_mask


class VarianceEmbedding(TypedDict):
    kernel_size: int
    dropout: float


class ModelConfig(TypedDict):
    # encoder_type: Literal["transformer", "conformer"]
    encoder: EncoderConfig
    # decoder_type: Literal["transformer", "conformer"]
    decoder: EncoderConfig
    variance_predictor: VariancePredictorConfig
    variance_embedding: VarianceEmbedding


def source_mask(self: nn.Module, ilens: LongTensor) -> Tensor:
    x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
    return x_masks.unsqueeze(-2)


class PitchAndDurationPredictor(nn.Module):
    def __init__(self, model_config: ModelConfig, speaker_num: int):
        super(PitchAndDurationPredictor, self).__init__()

        padding_idx = 0
        hidden = model_config["encoder"]["hidden"]
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=hidden,
            padding_idx=padding_idx
        )
        self.accent_embedding = nn.Embedding(
            num_embeddings=len(accent_symbols),
            embedding_dim=hidden,
            padding_idx=padding_idx
        )
        self.speaker_embedding = nn.Embedding(
            num_embeddings=speaker_num,
            embedding_dim=hidden,
        )

        self.encoder = Encoder(model_config["encoder"])
        self.duration_predictor = VariancePredictor(
            hidden, model_config["variance_predictor"]
        )
        self.pitch_predictor = VariancePredictor(
            hidden, model_config["variance_predictor"]
        )

    def forward(
        self,
        phonemes: Tensor,
        accents: Tensor,
        speakers: Tensor,
        phoneme_lens: LongTensor,
        max_phoneme_len: LongTensor,
    ):
        # forward encoder
        x_masks = source_mask(self, phoneme_lens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        x = self.phoneme_embedding(phonemes) + self.accent_embedding(accents)

        hs, _ = self.encoder(x, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        hs = hs + self.speaker_embedding(speakers).unsqueeze(1).expand(
            -1, max_phoneme_len, -1
        )

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(phoneme_lens).to(x.device)

        pitches: Tensor = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        log_durations: Tensor = self.duration_predictor(hs, d_masks)

        return pitches, log_durations


class MelSpectrogramDecoder(nn.Module):
    def __init__(self, model_config: ModelConfig, speaker_num: int):
        super(MelSpectrogramDecoder, self).__init__()

        kernel_size = model_config["variance_embedding"]["kernel_size"]
        dropout = model_config["variance_embedding"]["dropout"]

        self.length_regulator = LengthRegulator()
        padding_idx = 0
        hidden = model_config["encoder"]["hidden"]
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=hidden,
            padding_idx=padding_idx
        )
        self.speaker_embedding = nn.Embedding(
            num_embeddings=speaker_num,
            embedding_dim=hidden,
        )

        # NOTE(from ESPNet): use continuous pitch + FastPitch style avg
        self.pitch_embedding = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=hidden,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(dropout),
        )

        self.encoder = Encoder(model_config["encoder"])
        self.decoder = Encoder(model_config["decoder"])
        self.mel_channels = 80
        self.mel_linear = nn.Linear(hidden, self.mel_channels)
        self.postnet = Postnet()

    def forward(
        self,
        phonemes: Tensor,
        speakers: Tensor,
        phoneme_lens: LongTensor,
        max_phoneme_len: LongTensor,
        pitches: Tensor,
        durations: LongTensor,
        mel_lens: Optional[LongTensor],
    ):
        """Mel-spectrogram Decoder's forward
        Args:
            phonemes (Tensor): Phoneme Sequences
            speakers (Tensor): Speaker Sequences
            phoneme_lens (LongTensor): Phoneme Sequence Lengths
            max_phoneme_len (LongTensor): Max Phoneme Sequence Length
            pitches (Tensor): Pitch Sequences
            durations (LongTensor): Duration Sequences
            mel_lens (Optional[LongTensor]): Mel-spectrogram Lengths

        Returns:
            outputs (Tensor): Mel-spectrogram
            postnet_outputs (Tensor): Mel-spectrogram added postnet result
        """
        x_masks = source_mask(self, phoneme_lens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        x = self.phoneme_embedding(phonemes)

        hs, _ = self.encoder(x, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        hs = hs + self.speaker_embedding(speakers).unsqueeze(1).expand(
            -1, max_phoneme_len, -1
        )

        pitch_embeds = self.pitch_embedding(pitches.unsqueeze(1)).transpose(1, 2)
        hs = hs + pitch_embeds
        hs = self.length_regulator(hs, durations)  # (B, T_feats, adim)

        if mel_lens is not None:
            h_masks = source_mask(self, mel_lens)
        else:
            h_masks = None

        outputs, _ = self.decoder(hs, h_masks)
        outputs = self.mel_linear(outputs).view(
            outputs.size(0), -1, self.mel_channels
        )  # (B, T_feats, odim)

        postnet_outputs = outputs + self.postnet(
            outputs.transpose(1, 2)
        ).transpose(1, 2)

        return outputs, postnet_outputs
