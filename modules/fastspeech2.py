import torch

from torch import nn, Tensor, LongTensor

from typing import TypedDict, Literal, Optional

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


VocoderType = Literal["fregan", "hifigan", "melgan"]


class ModelConfig(TypedDict):
    encoder_type: Literal["transformer", "conformer"]
    encoder: EncoderConfig
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


class PitchAndDurationPredictor(BaseModule):
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

        encoder_type = model_config["encoder_type"]
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(model_config["encoder"])
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(model_config["encoder"])
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
            x_masks = torch.ones_like(phonemes).squeeze()
        else:
            x_masks = self.source_mask(phoneme_lens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        x = self.phoneme_embedding(phonemes) + self.accent_embedding(accents)

        hs, _ = self.encoder(x, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

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
            d_masks = ~torch.ones_like(phonemes).to(device=x.device, dtype=torch.bool)
        else:
            d_masks = make_pad_mask(phoneme_lens).to(x.device)

        pitches: Tensor = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        log_durations: Tensor = self.duration_predictor(hs, d_masks.unsqueeze(-1))

        return pitches, log_durations


class FeatureEmbedder(BaseModule):
    def __init__(self, model_config: ModelConfig, speaker_num: int, pitch_min: float, pitch_max: float):
        super(FeatureEmbedder, self).__init__()

        kernel_size = model_config["variance_embedding"]["kernel_size"]
        dropout = model_config["variance_embedding"]["dropout"]

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

        encoder_type = model_config["encoder_type"]
        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(model_config["encoder"])
        elif encoder_type == "transformer":
            self.encoder = TransformerEncoder(model_config["encoder"])
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
            x_masks = torch.ones_like(phonemes).squeeze()
        else:
            x_masks = self.source_mask(phoneme_lens)  # (B, Tmax, Tmax) -> torch.Size([32, 121, 121])

        x = self.phoneme_embedding(phonemes)

        feature_embeded, _ = self.encoder(x, x_masks)  # (B, Tmax, adim) -> torch.Size([32, 121, 256])

        if max_phoneme_len is None:
            feature_embeded = feature_embeded + self.speaker_embedding(speakers).unsqueeze(1).expand(
                -1, phonemes.shape[1], -1
            )
        else:
            feature_embeded = feature_embeded + self.speaker_embedding(speakers).unsqueeze(1).expand(
                -1, max_phoneme_len, -1
            )

        if self.pitch_embedding_type == "normal":
            pitch_embeds = self.pitch_embedding(self.bucketize(pitches, self.pitch_bins))
        else:
            # fastpitch style
            pitch_embeds = self.pitch_embedding(pitches.unsqueeze(1)).transpose(1, 2)

        feature_embeded += pitch_embeds
        return feature_embeded


class MelSpectrogramDecoder(BaseModule):
    def __init__(self, model_config: ModelConfig):
        super(MelSpectrogramDecoder, self).__init__()

        hidden = model_config["encoder"]["hidden"]

        decoder_type = model_config["encoder_type"]
        if decoder_type == "conformer":
            self.decoder = ConformerEncoder(model_config["encoder"])
        elif decoder_type == "transformer":
            self.decoder = TransformerEncoder(model_config["encoder"])
        else:
            raise ValueError("unknown decoder: " + decoder_type)

        self.mel_channels = 80
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
