import torch

from torch import nn, Tensor, LongTensor

from typing import TypedDict, Literal, Optional, Union

import fregan
import hifigan
from fregan import VocoderConfig
from modules.alignment import AlignmentModule, average_by_duration, binarize_attention_parallel
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
VocoderMultiPeriodDiscriminator = Union[fregan.ResWiseMultiPeriodDiscriminator, hifigan.Generator]
VocoderMultiScaleDiscriminator = Union[fregan.ResWiseMultiScaleDiscriminator, hifigan.Generator]


class ModelConfig(TypedDict):
    variance_encoder_type: Literal["transformer", "conformer"]
    variance_encoder: EncoderConfig
    phoneme_encoder_type: Literal["transformer", "conformer"]
    phoneme_encoder: EncoderConfig
    decoder_type: Literal["transformer", "conformer"]
    decoder: EncoderConfig
    variance_predictor: VariancePredictorConfig
    variance_embedding: VarianceEmbedding
    aligner_temperature: float
    vocoder_type: VocoderType
    vocoder: VocoderConfig


class BaseModule(nn.Module):
    def source_mask(self, ilens: LongTensor) -> Tensor:
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        # x_masks = make_non_pad_mask(ilens)
        return x_masks.unsqueeze(-2)


class PitchAndDurationPredictor(BaseModule):
    def __init__(self, model_config: ModelConfig, speaker_num: int):
        super(PitchAndDurationPredictor, self).__init__()

        padding_idx = 0
        hidden = model_config["variance_encoder"]["hidden"]
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

        encoder_type = model_config["variance_encoder_type"]
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(model_config["variance_encoder"])
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(model_config["variance_encoder"])
        else:
            raise ValueError("unknown encoder: " + encoder_type)

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
        phoneme_lens: Optional[LongTensor] = None,
        max_phoneme_len: Optional[LongTensor] = None,
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
        pitches: Tensor = self.pitch_predictor(
            pitches_args[0], pitches_args[1].unsqueeze(-1))

        # predict log_durations with a phoneme
        log_durations_args = self.__forward_preprocessing(
            phonemes, speakers, phoneme_embedding, x_masks, phoneme_lens, max_phoneme_len)
        log_durations: Tensor = self.duration_predictor(
            log_durations_args[0], log_durations_args[1].unsqueeze(-1))

        return pitches, log_durations

    def __forward_preprocessing(
        self,
        phonemes: Tensor,
        speakers: Tensor,
        x: Tensor,
        x_masks: Tensor,
        phoneme_lens: Optional[LongTensor] = None,
        max_phoneme_len: Optional[LongTensor] = None,
    ):
        hs, _ = self.encoder(x, x_masks)    # (B, Tmax, adim) -> torch.Size()

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
        temperature = model_config["aligner_temperature"]

        self.alignment_module = AlignmentModule(hidden, 80, temperature)

    def forward(
        self,
        hs: Tensor,
        pitches: Tensor,
        mels: Tensor,
        phoneme_lens: LongTensor,
        mel_lens: LongTensor,
        attn_priors: Tensor,
        speaker_embedding: Tensor,
    ):
        h_masks = make_pad_mask(phoneme_lens).to(hs.device)
        attn_soft, attn_logprob = self.alignment_module(
            hs,
            mels,
            h_masks.unsqueeze(-1),
            attn_priors.transpose(1, 2),
            speaker_embedding,
        )
        attn_hard = binarize_attention_parallel(attn_soft, phoneme_lens, mel_lens)
        durations = attn_hard.sum(2)[:, 0, :]
        avg_pitches = average_by_duration(durations, pitches.squeeze(-1), phoneme_lens, mel_lens).unsqueeze(-1)

        return avg_pitches, durations, attn_soft, attn_hard, attn_logprob


class FeatureEmbedder(BaseModule):
    def __init__(self, model_config: ModelConfig, speaker_num: int, pitch_min: float, pitch_max: float):
        super(FeatureEmbedder, self).__init__()

        kernel_size = model_config["variance_embedding"]["kernel_size"]
        dropout = model_config["variance_embedding"]["dropout"]

        padding_idx = 0
        hidden = model_config["phoneme_encoder"]["hidden"]
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=hidden,
            padding_idx=padding_idx
        )
        self.speaker_embedding = nn.Embedding(
            num_embeddings=speaker_num,
            embedding_dim=hidden,
        )

        self.extractor = PitchAndDurationExtractor(model_config)

        self.pitch_embedding_type = model_config["variance_embedding"]["pitch_embedding_type"]
        if self.pitch_embedding_type == "normal":
            assert model_config["variance_embedding"]["n_bins"], "please specify n_bins"
            n_bins: int = model_config["variance_embedding"]["n_bins"]
            self.pitch_embedding = nn.Embedding(
                n_bins, hidden
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
                    out_channels=hidden,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                torch.nn.Dropout(dropout),
            )

        encoder_type = model_config["phoneme_encoder_type"]
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(model_config["phoneme_encoder"])
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(model_config["phoneme_encoder"])
        else:
            raise ValueError("unknown encoder: " + encoder_type)

    @staticmethod
    def bucketize(tensor: Tensor, bucket_boundaries: Tensor) -> LongTensor:
        """for onnx, https://github.com/ming024/FastSpeech2/issues/98#issuecomment-952490935"""
        result = torch.zeros_like(tensor, dtype=torch.int32)
        for boundary in bucket_boundaries:
            result += (tensor > boundary).int()
        return result.long()

    def forward(
        self,
        phonemes: Tensor,
        pitches: Tensor,
        speakers: Tensor,
        phoneme_lens: Optional[LongTensor] = None,
        max_phoneme_len: Optional[LongTensor] = None,
        mels: Optional[Tensor] = None,
        mel_lens: Optional[LongTensor] = None,
        attn_priors: Optional[Tensor] = None,
    ):
        """Feature Embedder's forward
        Args:
            phonemes (Tensor): Phoneme Sequences
            pitches (Tensor): Pitch Sequences
            speakers (Tensor): Speaker Sequences
            phoneme_lens (LongTensor): Phoneme Sequence Lengths
            max_phoneme_len (Optional[LongTensor]): Max Phoneme Sequence Length

        Returns:
            feature_embeded (Tensor): Feature embedded tensor
        """
        if phoneme_lens is None:
            x_masks = torch.ones_like(phonemes).squeeze(0)
        else:
            x_masks = self.source_mask(phoneme_lens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        phoneme_embeds = self.phoneme_embedding(phonemes)

        feature_embedded, _ = self.encoder(phoneme_embeds, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        speaker_embedding = self.speaker_embedding(speakers)

        avg_pitches, durations, attn_soft, attn_hard, attn_logprob = None, None, None, None, None
        embedding_pitches = pitches
        if mels is not None:
            avg_pitches, durations, attn_soft, attn_hard, attn_logprob = self.extractor(
                hs=phoneme_embeds,
                pitches=pitches,
                mels=mels,
                phoneme_lens=phoneme_lens,
                mel_lens=mel_lens,
                attn_priors=attn_priors,
                speaker_embedding=speaker_embedding,
            )
            embedding_pitches = avg_pitches

        if max_phoneme_len is None:
            feature_embedded = feature_embedded + speaker_embedding.unsqueeze(1).expand(
                -1, phonemes.shape[1], -1
            )
        else:
            feature_embedded = feature_embedded + speaker_embedding.unsqueeze(1).expand(
                -1, max_phoneme_len, -1
            )

        if self.pitch_embedding_type == "normal":
            pitch_embeds = self.pitch_embedding(self.bucketize(embedding_pitches, self.pitch_bins))
        else:
            # fastpitch style
            pitch_embeds = self.pitch_embedding(embedding_pitches.transpose(1, 2)).transpose(1, 2)

        feature_embedded += pitch_embeds
        return feature_embedded, avg_pitches, durations, attn_soft, attn_hard, attn_logprob


class MelSpectrogramDecoder(BaseModule):
    def __init__(self, model_config: ModelConfig, mel_channels):
        super(MelSpectrogramDecoder, self).__init__()

        hidden = model_config["decoder"]["hidden"]

        decoder_type = model_config["decoder_type"]
        if decoder_type == "conformer":
            self.decoder = ConformerEncoder(model_config["decoder"])
        elif decoder_type == "transformer":
            self.decoder = TransformerEncoder(model_config["decoder"])
        else:
            raise ValueError("unknown decoder: " + decoder_type)

        self.mel_channels = mel_channels
        self.mel_linear = nn.Linear(hidden, self.mel_channels)
        self.postnet = Postnet()

    def forward(
        self,
        length_regulated_tensor: Tensor,
        mel_lens: Optional[LongTensor] = None,
    ):
        """Mel-spectrogram Decoder's forward
        Args:
            length_regulated_tensor (Tensor): Length Regulated Tensor
            mel_lens (Optional[LongTensor]): Mel-spectrogram Lengths

        Returns:
            outputs (Tensor): Mel-spectrogram
            postnet_outputs (Tensor): Mel-spectrogram added postnet result
        """
        if mel_lens is not None:
            h_masks = self.source_mask(mel_lens)
        else:
            h_masks = None

        outputs, _ = self.decoder(length_regulated_tensor, h_masks)
        outputs = self.mel_linear(outputs).view(
            outputs.size(0), -1, self.mel_channels
        )  # (B, T_feats, odim)

        postnet_outputs = outputs + self.postnet(
            outputs.transpose(1, 2)
        ).transpose(1, 2)

        return outputs, postnet_outputs
