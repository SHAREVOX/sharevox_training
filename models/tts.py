import math
from typing import Optional, Tuple, TypedDict, Literal

import numpy as np
import torch
from torch import nn, Tensor, LongTensor
from models.fastspeech.length_regulator import LengthRegulator
from models.fastspeech.variance_predictor import VariancePredictorConfig, VariancePredictor
from models.jets.alignment import PitchAndDurationExtractor
from models.upsampler import UpsamplerConfig, UpsamplerTypeConfig, SiFiGANGenerator, SFreGAN2Generator, SignalGenerator
from models.upsampler.utils import dilated_factor
from models.vits.posterior_encoder import PosteriorEncoder, PosteriorEncoderConfig
from models.vits.residual_coupling import FlowConfig, ResidualAffineCouplingBlock

from models.transformer.encoder import Encoder as TransformerEncoder, EncoderConfig, EncoderConfigType
from models.conformer.encoder import Encoder as ConformerEncoder
from models.vits.stochastic_predictor import StochasticPredictor, StochasticPreditorConfig

from text import phoneme_symbols, accent_symbols, _symbol_to_id
from utils.mask import make_non_pad_mask
from utils.slice import slice_segments, rand_slice_segments


ModelType = Literal["jets", "vits"]

class VarianceEmbeddingConfig(TypedDict):
    hidden: int
    kernel_size: int
    dropout: float


class ModelConfig(TypedDict):
    model_type: ModelType
    variance_encoder_type: EncoderConfigType
    variance_encoder: EncoderConfig
    prior_encoder_type: EncoderConfigType
    prior_encoder: EncoderConfig
    posterior_encoder: PosteriorEncoderConfig
    flow: FlowConfig
    frame_prior_network_type: EncoderConfigType
    frame_prior_network: EncoderConfig
    duration_predictor: VariancePredictorConfig
    pitch_predictor: VariancePredictorConfig
    pitch_embedding: VarianceEmbeddingConfig
    pitch_upsampler: VariancePredictorConfig
    upsampler_type: UpsamplerTypeConfig
    upsampler: UpsamplerConfig
    global_hidden: int


class TextEncoder(nn.Module):
    def __init__(
        self,
        encoder_type: EncoderConfigType,
        config: EncoderConfig,
    ):
        super().__init__()
        self.hidden_channels = config["hidden"]

        self.emb = nn.Embedding(len(phoneme_symbols), self.hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, self.hidden_channels ** -0.5)

        if encoder_type == "conformer":
            self.encoder = ConformerEncoder(config)
        else:
            self.encoder = TransformerEncoder(config)

    def forward(self, x: Tensor, x_lengths: Optional[LongTensor] = None) -> Tuple[Tensor, Tensor]:
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        if x_lengths is None:
            x_mask = None
        else:
            x_mask = make_non_pad_mask(x_lengths).to(x.device).unsqueeze(1)

        x, _ = self.encoder(x, x_mask)
        x = x.transpose(1, 2)
        return x, x_mask


class FramePriorNetwork(nn.Module):
    def __init__(
        self,
        encoder_type: EncoderConfigType,
        config: EncoderConfig,
        out_channels: int,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.hidden_channels = config["hidden"]

        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(config)
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(config)
        else:
            raise Exception(f"Unknown frame prior network type: {encoder_type}")

        self.proj = nn.Conv1d(self.hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask=None):
        x, _ = self.encoder(x, x_mask)
        x = x.transpose(1, 2)
        stats = self.proj(x)
        if x_mask is not None:
            stats.masked_fill_(~x_mask, 0)

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class JETS(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        spec_channels: int,
        pitch_mean: int,
        pitch_std: int,
        sampling_rate=24000,
        hop_length=256,
        n_speakers=0,
        onnx=False,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.segment_size = config["upsampler"]["segment_size"] // hop_length
        self.hop_length = hop_length
        self.n_speakers = n_speakers
        self.global_hidden = config["global_hidden"]
        self.sampling_rate = sampling_rate
        self.hidden_channels = config["prior_encoder"]["hidden"]
        self.dense_factors = config["upsampler"]["dense_factors"]
        self.prod_upsample_scales = np.cumprod(config["upsampler"]["upsample_rates"])

        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std
        self.unvoice_pitch = (1 - self.pitch_mean) / self.pitch_std

        self.onnx = onnx

        variance_encoder_type = config["variance_encoder_type"]
        if variance_encoder_type == "transformer":
            self.variance_encoder = TransformerEncoder(config["variance_encoder"])
        elif variance_encoder_type == "conformer":
            self.variance_encoder = ConformerEncoder(config["variance_encoder"])
        else:
            raise Exception(f"Unknown upsampler type: {variance_encoder_type}")

        padding_idx = 0
        self.phoneme_embedding = nn.Embedding(
            num_embeddings=len(phoneme_symbols),
            embedding_dim=config["variance_encoder"]["hidden"],
            padding_idx=padding_idx
        )
        self.accent_embedding = nn.Embedding(
            num_embeddings=len(accent_symbols),
            embedding_dim=config["variance_encoder"]["hidden"],
            padding_idx=padding_idx
        )
        self.duration_predictor = VariancePredictor(
            config["duration_predictor"],
            config["variance_encoder"]["hidden"],
            self.global_hidden
        )
        self.pitch_predictor = VariancePredictor(
            config["pitch_predictor"],
            config["variance_encoder"]["hidden"],
            self.global_hidden
        )

        self.enc_p = TextEncoder(config["prior_encoder_type"], config["prior_encoder"])
        self.length_regulator = LengthRegulator()
        self.signal_generator = SignalGenerator(
            sample_rate=sampling_rate, hop_size=hop_length, noise_amp=0.003 if not onnx else 0,
        )
        upsampler_type = config["upsampler_type"]
        if upsampler_type == "sfregan2" or upsampler_type == "sfregan2m":
            self.dec = SFreGAN2Generator(
                config["upsampler"],
                input_size=self.hidden_channels,
                multi_idwt=config["upsampler_type"] == "sfregan2m",
                gin_channels=self.global_hidden,
                onnx=onnx,
            )
        elif upsampler_type == "sifigan":
            self.dec = SiFiGANGenerator(
                config["upsampler"],
                input_size=self.hidden_channels,
                gin_channels=self.global_hidden,
            )
        else:
            raise Exception(f"Unknown upsampler type: {upsampler_type}")

        frame_prior_network_type = config["frame_prior_network_type"]
        if frame_prior_network_type == "transformer":
            self.frame_prior_network = TransformerEncoder(config["frame_prior_network"])
        elif frame_prior_network_type == "conformer":
            self.frame_prior_network = ConformerEncoder(config["frame_prior_network"])
        else:
            raise Exception(f"Unknown frame prior network type: {frame_prior_network_type}")

        self.extractor = PitchAndDurationExtractor(self.hidden_channels, spec_channels)

        self.pitch_embedding = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config["pitch_embedding"]["hidden"],
                kernel_size=config["pitch_embedding"]["kernel_size"],
                padding=(config["pitch_embedding"]["kernel_size"] - 1) // 2,
            ),
            torch.nn.Dropout(config["pitch_embedding"]["dropout"]),
        )

        self.pitch_upsampler = VariancePredictor(
            config["pitch_upsampler"],
            config["pitch_embedding"]["hidden"],
            self.global_hidden
        )
        # self.f0_upsampler = nn.Upsample(scale_factor=2, mode="linear")

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, self.global_hidden)

    def forward_variance(self, phonemes: Tensor, accents: Tensor, mask: Optional[Tensor] = None, g: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        phoneme_embedding = self.phoneme_embedding(phonemes)
        accent_embedding = self.accent_embedding(accents)

        duration_hs, _ = self.variance_encoder(phoneme_embedding, None)
        pitch_hs, _ = self.variance_encoder(phoneme_embedding + accent_embedding, None)

        if mask is not None:
            pred_pitches = self.pitch_predictor(pitch_hs, ~mask, g=g)
            pred_durations = self.duration_predictor(duration_hs, ~mask, g=g)
        else:
            pred_pitches = self.pitch_predictor(pitch_hs, g=g)
            pred_durations = self.duration_predictor(duration_hs, g=g)

        return pred_pitches, pred_durations

    def forward_pitch_upsampler(self, pitches: Tensor, y_mask: Optional[Tensor] = None, g: Optional[Tensor] = None) -> Tensor:
        pitch_embed = self.pitch_embedding(pitches).transpose(1, 2)
        if y_mask is not None:
            y_mask = y_mask.bool()
            pred_frame_pitches = self.pitch_upsampler(pitch_embed, ~y_mask, g=g)
        else:
            pred_frame_pitches = self.pitch_upsampler(pitch_embed, g=g)

        return pred_frame_pitches

    def make_unvoice_mask(self, phonemes: Tensor, durations: LongTensor) -> Tensor:
        phonemes = phonemes.unsqueeze(-1)
        B = phonemes.size(0)
        unvoice_mask = (
            (phonemes == _symbol_to_id["pau"]) |
            (phonemes == _symbol_to_id["cl"]) |
            (phonemes == _symbol_to_id["I"]) |
            (phonemes == _symbol_to_id["U"]) |
            torch.cat(
                ((phonemes == _symbol_to_id["I"])[:, 1:], torch.zeros(B, 1, 1).to(phonemes.device, dtype=torch.bool)), dim=1
            ) |
            torch.cat(
                ((phonemes == _symbol_to_id["U"])[:, 1:], torch.zeros(B, 1, 1).to(phonemes.device, dtype=torch.bool)), dim=1
            )
        )
        regulated_unvoice_mask = self.length_regulator(unvoice_mask, durations).transpose(1, 2)
        return regulated_unvoice_mask

    def pitch_smoothly(self, pitches: Tensor, unvoice_mask: Optional[Tensor] = None) -> Tensor:
        pitches = pitches * self.pitch_std + self.pitch_mean
        pitches[pitches < 10] = 1
        pitches[pitches > 750] = 1
        if unvoice_mask is not None:
            pitches[unvoice_mask] = 1

        # downsampled_pitches = torch.cat(
        #     (pitches[:, :, ::2], torch.ones(pitches.shape[0], 1, 2).to(pitches.device)), dim=2
        # )
        # upsampled_pitches = self.f0_upsampler(downsampled_pitches)[:, :, :pitches.shape[2]]
        # upsampled_pitches[pitches == 1] = 1
        # return upsampled_pitches
        return pitches

    def forward_upsampler(self, z: Tensor, pitches: Tensor, g: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        dfs = []
        for df, us in zip(self.dense_factors, self.prod_upsample_scales):
            result = []
            for pitch in pitches:
                dilated_tensor = dilated_factor(pitch, self.sampling_rate, df)
                if self.onnx:
                    result += [
                        torch.stack([dilated_tensor for _ in range(us)], -1).reshape(dilated_tensor.shape[0], -1)
                    ]
                else:
                    result += [
                        torch.repeat_interleave(dilated_tensor, us, dim=1)
                    ]
            dfs.append(torch.cat(result, dim=0).unsqueeze(1))
        sin_waves = self.signal_generator(pitches)

        # forward upsampler with(out) random segments
        # exc = excitation
        o, excs = self.dec(z, f0=sin_waves, d=dfs, g=g)

        return o, excs

    def forward(
        self,
        phonemes: Tensor,
        phoneme_lens: LongTensor,
        moras: LongTensor,
        accents: Tensor,
        pitches: Tensor,
        specs: Tensor,
        spec_lens: LongTensor,
        sid: Optional[Tensor] = None,
        slice: bool = True,
    ):
        x, x_mask = self.enc_p(phonemes, phoneme_lens)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        pred_pitches, pred_durations = self.forward_variance(phonemes, accents, x_mask, g)

        pitches = pitches.unsqueeze(1)
        mora_avg_pitches, durations, mora_durations, attn, bin_loss = self.extractor(
            (x + g).transpose(1, 2), moras, pitches, specs, phoneme_lens, spec_lens
        )
        with torch.no_grad():
            avg_pitches = (moras.transpose(1, 2).to(mora_avg_pitches.dtype) * mora_avg_pitches).sum(dim=-1).unsqueeze(1)

        x = self.length_regulator(x.transpose(1, 2), durations)

        regulated_pitches = self.length_regulator(mora_avg_pitches.transpose(1, 2), mora_durations).transpose(1, 2)
        unvoice_mask = self.make_unvoice_mask(phonemes, durations)
        regulated_pitches[unvoice_mask] = self.unvoice_pitch
        y_mask = make_non_pad_mask(spec_lens).unsqueeze(1).to(x.device)

        pred_frame_pitches = self.forward_pitch_upsampler(regulated_pitches, y_mask, g)
        z, _ = self.frame_prior_network(x, y_mask)
        smoothly_pitches = self.pitch_smoothly(pitches, unvoice_mask)

        if slice:
            z_slice, ids_slice = rand_slice_segments(
                z.transpose(1, 2), spec_lens, self.segment_size
            )

            pitch_slices = slice_segments(
                smoothly_pitches, ids_slice, self.segment_size
            )
        else:
            ids_slice = torch.zeros_like(spec_lens)
            z_slice = z.transpose(1, 2)
            pitch_slices = smoothly_pitches

        o, excs = self.forward_upsampler(z_slice, pitch_slices, g)

        return (
            o,
            excs,
            pitch_slices,
            durations,
            pred_durations,
            avg_pitches,
            pred_pitches,
            pred_frame_pitches,
            unvoice_mask,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            bin_loss,
            None,
        )

    def infer(
        self,
        phonemes: Tensor,
        phoneme_lens: LongTensor,
        moras: LongTensor,
        accents: Tensor,
        pitches: TextEncoder,
        specs: Tensor,
        spec_lens: LongTensor,
        sid: Optional[Tensor] = None,
    ):
        x, x_mask = self.enc_p(phonemes, phoneme_lens)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        pred_pitches, pred_durations = self.forward_variance(phonemes, accents, x_mask, g)

        pitches = pitches.unsqueeze(1)
        mora_avg_pitches, durations, mora_durations, attn, _ = self.extractor(
            (x + g).transpose(1, 2), moras, pitches, specs, phoneme_lens, spec_lens
        )

        w = torch.exp(pred_durations) * x_mask
        pred_durations = torch.clip(torch.round(w).squeeze(1) - 1, min=1).long()
        pred_mora_pitches = (moras.to(pred_pitches.dtype) * pred_pitches).sum(dim=-1) / moras.sum(-1)
        pred_mora_pitches = pred_mora_pitches.unsqueeze(1)
        pred_mora_durations = (moras * pred_durations).sum(dim=-1)

        x = self.length_regulator(x.transpose(1, 2), pred_durations)

        regulated_pitches = self.length_regulator(mora_avg_pitches.transpose(1, 2), mora_durations).transpose(1, 2)
        unvoice_mask = self.make_unvoice_mask(phonemes, durations)
        regulated_pitches[unvoice_mask] = self.unvoice_pitch

        pred_regulated_pitches = self.length_regulator(pred_mora_pitches.transpose(1, 2), pred_mora_durations).transpose(1, 2)
        pred_unvoice_mask = self.make_unvoice_mask(phonemes, pred_durations)
        pred_regulated_pitches[pred_unvoice_mask] = self.unvoice_pitch

        y_mask = make_non_pad_mask(torch.tensor([pred_regulated_pitches.shape[2]])).unsqueeze(1).to(specs.device)

        pred_frame_pitches = self.forward_pitch_upsampler(pred_regulated_pitches, y_mask, g)
        z, _ = self.frame_prior_network(x, y_mask)
        smoothly_pitches = self.pitch_smoothly(pred_frame_pitches, pred_unvoice_mask)

        o, excs = self.forward_upsampler(z.transpose(1, 2), smoothly_pitches, g=g)

        return o, excs, attn, regulated_pitches, pred_regulated_pitches, smoothly_pitches, y_mask


class VITS(JETS):
    def __init__(
        self,
        config: ModelConfig,
        spec_channels: int,
        pitch_mean: int,
        pitch_std: int,
        sampling_rate=24000,
        hop_length=256,
        n_speakers=0,
        onnx=False,
        **kwargs
    ):
        super().__init__(
            config,
            spec_channels,
            pitch_mean,
            pitch_std,
            sampling_rate,
            hop_length,
            n_speakers,
            onnx,
            **kwargs
        )

        self.frame_prior_network = FramePriorNetwork(
            config["frame_prior_network_type"], config["frame_prior_network"], self.hidden_channels
        )

        self.enc_q = PosteriorEncoder(
            config["posterior_encoder"],
            in_channels=spec_channels,
            out_channels=self.hidden_channels,
            gin_channels=self.global_hidden,
        )
        self.flow = ResidualAffineCouplingBlock(config["flow"], self.hidden_channels)

    def forward(
        self,
        phonemes: Tensor,
        phoneme_lens: LongTensor,
        moras: LongTensor,
        accents: Tensor,
        pitches: Tensor,
        specs: Tensor,
        spec_lens: LongTensor,
        sid: Optional[Tensor] = None,
        slice: bool = True,
    ):
        x, x_mask = self.enc_p(phonemes, phoneme_lens)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        pred_pitches, pred_durations = self.forward_variance(phonemes, accents, x_mask, g)

        pitches = pitches.unsqueeze(1)
        mora_avg_pitches, durations, mora_durations, attn, bin_loss = self.extractor(
            (x + g).transpose(1, 2), moras, pitches, specs, phoneme_lens, spec_lens
        )
        with torch.no_grad():
            avg_pitches = (moras.transpose(1, 2).to(mora_avg_pitches.dtype) * mora_avg_pitches).sum(dim=-1).unsqueeze(1)

        x = self.length_regulator(x.transpose(1, 2), durations)

        regulated_pitches = self.length_regulator(mora_avg_pitches.transpose(1, 2), mora_durations).transpose(1, 2)
        unvoice_mask = self.make_unvoice_mask(phonemes, durations)
        regulated_pitches[unvoice_mask] = self.unvoice_pitch

        z, m_q, logs_q, y_mask = self.enc_q(specs.transpose(1, 2), spec_lens, g=g)
        z_p = self.flow(z, y_mask, g=g)

        pred_frame_pitches = self.forward_pitch_upsampler(regulated_pitches, y_mask, g)
        x, m_p, logs_p, _ = self.frame_prior_network(x, y_mask.bool())
        smoothly_pitches = self.pitch_smoothly(pitches, unvoice_mask)

        if slice:
            z_slice, ids_slice = rand_slice_segments(
                z, spec_lens, self.segment_size
            )

            pitch_slices = slice_segments(
                smoothly_pitches, ids_slice, self.segment_size
            )
        else:
            ids_slice = torch.zeros_like(spec_lens)
            z_slice = z
            pitch_slices = smoothly_pitches

        o, excs = self.forward_upsampler(z_slice, pitch_slices)

        return (
            o,
            excs,
            pitch_slices,
            durations,
            pred_durations,
            avg_pitches,
            pred_pitches,
            pred_frame_pitches,
            unvoice_mask,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            bin_loss,
            (z, z_p, m_p, logs_p, m_q, logs_q),
        )


    def infer(
        self,
        phonemes: Tensor,
        phoneme_lens: LongTensor,
        moras: LongTensor,
        accents: Tensor,
        pitches: TextEncoder,
        specs: Tensor,
        spec_lens: LongTensor,
        sid: Optional[Tensor] = None,
    ):
        x, x_mask = self.enc_p(phonemes, phoneme_lens)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        pred_pitches, pred_durations = self.forward_variance(phonemes, accents, x_mask, g)

        pitches = pitches.unsqueeze(1)
        mora_avg_pitches, durations, mora_durations, attn, _ = self.extractor(
            (x + g).transpose(1, 2), moras, pitches, specs, phoneme_lens, spec_lens
        )

        w = torch.exp(pred_durations) * x_mask
        pred_durations = torch.clip(torch.round(w).squeeze(1) - 1, min=1).long()
        pred_mora_pitches = (moras.to(pred_pitches.dtype) * pred_pitches).sum(dim=-1) / moras.sum(-1)
        pred_mora_pitches = pred_mora_pitches.unsqueeze(1)
        pred_mora_durations = (moras * pred_durations).sum(dim=-1)

        x = self.length_regulator(x.transpose(1, 2), pred_durations)

        regulated_pitches = self.length_regulator(mora_avg_pitches.transpose(1, 2), mora_durations).transpose(1, 2)
        unvoice_mask = self.make_unvoice_mask(phonemes, durations)
        regulated_pitches[unvoice_mask] = self.unvoice_pitch

        pred_regulated_pitches = self.length_regulator(pred_mora_pitches.transpose(1, 2), pred_mora_durations).transpose(1, 2)
        pred_unvoice_mask = self.make_unvoice_mask(phonemes, pred_durations)
        pred_regulated_pitches[pred_unvoice_mask] = self.unvoice_pitch

        y_mask = make_non_pad_mask(torch.tensor([pred_regulated_pitches.shape[2]])).unsqueeze(1).to(specs.device)

        pred_frame_pitches = self.forward_pitch_upsampler(pred_regulated_pitches, y_mask, g)
        _, m_p, logs_p, _ = self.frame_prior_network(x, y_mask.bool())
        smoothly_pitches = self.pitch_smoothly(pred_frame_pitches, pred_unvoice_mask)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
        z = self.flow(z_p, y_mask, g=g, inverse=True)

        o, excs = self.forward_upsampler(z, smoothly_pitches)

        return o, excs, attn, regulated_pitches, pred_regulated_pitches, smoothly_pitches, y_mask

