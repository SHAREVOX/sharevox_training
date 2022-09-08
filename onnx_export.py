import argparse
import json
import os
from typing import List

import numpy as np
import torch
import yaml

from torch import nn, Tensor, LongTensor

from fregan import Generator
from modules.fastspeech2 import MelSpectrogramDecoder, PitchAndDurationPredictor, FeatureEmbedder, VocoderGenerator
from text import phoneme_to_id, accent_to_id
from utils.model import Config, get_model, get_vocoder

from torch.onnx.symbolic_registry import _onnx_stable_opsets

OPSET = _onnx_stable_opsets[-1]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Variance(nn.Module):
    def __init__(self, config: Config, variance_model: PitchAndDurationPredictor, pitch_mean: float, pitch_std: float):
        super(Variance, self).__init__()

        self.sampling_rate = config["preprocess"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocess"]["stft"]["hop_length"]
        self.variance_model = variance_model
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std

    def forward(
        self,
        phonemes: Tensor,
        accents: Tensor,
        speakers: Tensor,
    ):
        pitches, log_durations, _ = self.variance_model(phonemes, accents, speakers)
        pitches = torch.log(pitches * self.pitch_std + self.pitch_mean)
        durations = torch.clamp((torch.exp(log_durations) - 1) / (self.sampling_rate / self.hop_length), min=0.01)
        return pitches, durations


class Embedder(nn.Module):
    def __init__(self, config: Config, embedder_model: FeatureEmbedder, pitch_mean: float, pitch_std: float):
        super(Embedder, self).__init__()

        self.sampling_rate = config["preprocess"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocess"]["stft"]["hop_length"]
        self.embedder_model = embedder_model
        self.pitch_mean = pitch_mean
        self.pitch_std = pitch_std

    def forward(
        self,
        phonemes: Tensor,
        pitches: Tensor,
        speakers: Tensor,
    ):
        pitches = (torch.exp(pitches) - self.pitch_mean) / self.pitch_std
        feature_embedded = self.embedder_model(phonemes, pitches.unsqueeze(-1), speakers)
        return feature_embedded


class Decoder(nn.Module):
    def __init__(self, config: Config, decoder: MelSpectrogramDecoder, vocoder: VocoderGenerator):
        super(Decoder, self).__init__()

        self.max_wav_value = config["preprocess"]["audio"]["max_wav_value"]
        self.decoder = decoder
        self.vocoder_type = config["model"]["vocoder_type"]
        self.vocoder = vocoder

    def forward(self, length_regulated_tensor: Tensor) -> Tensor:
        _, postnet_outputs = self.decoder(length_regulated_tensor)
        if self.vocoder_type == "melgan":
            wavs = self.vocoder.inference(postnet_outputs[0].transpose(0, 1).unsqueeze(0)).unsqueeze(0)
        else:
            wavs = self.vocoder(postnet_outputs[0].transpose(0, 1).unsqueeze(0)).squeeze(1)
        return wavs


class GaussianUpsampling(torch.nn.Module):
    """Gaussian upsampling with fixed temperature as in:

    https://arxiv.org/abs/2010.04301

    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds):
        """Upsample hidden states according to durations.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        """
        device = ds.device

        T_feats = ds.sum().int()
        t = torch.arange(0, T_feats).unsqueeze(0).to(device).float()

        c = ds.cumsum(dim=-1) - ds / 2
        c = c.float()
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)
        return hs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_num", type=int, default=10)
    args = parser.parse_args()
    config: Config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    variance_model, embedder_model, decoder_model, _, _ = get_model(args.restore_step, config, device, args.speaker_num, False)
    gaussian_model = GaussianUpsampling()
    gaussian_model = gaussian_model.eval()
    fregan_model = get_vocoder(device, config["model"]["vocoder_type"])
    with open(
        os.path.join(config["preprocess"]["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        pitch_data: List[float] = stats["pitch"]
        pitch_mean, pitch_std = pitch_data[2], pitch_data[3]
    variance_model = Variance(config, variance_model, pitch_mean, pitch_std)
    embedder_model = Embedder(config, embedder_model, pitch_mean, pitch_std)
    decoder_model = Decoder(config, decoder_model, fregan_model)
    decoder_model.eval()
    decoder_model.requires_grad_ = False

    phonemes = torch.from_numpy(np.array([[phoneme_to_id[p] for p in "k o N n i ch i w a".split(" ")]])).to(dtype=torch.int64, device=device)
    accents = torch.from_numpy(np.array([[accent_to_id[a] for a in "_ [ _ _ _ _ _ _ #".split(" ")]])).to(dtype=torch.int64, device=device)
    speakers = torch.from_numpy(np.array([0])).to(dtype=torch.int64, device=device)

    variance_input = (phonemes, accents, speakers)
    torch.onnx.export(
        variance_model,
        variance_input,
        "variance_model.onnx",
        input_names=["phonemes", "accents", "speakers"],
        output_names=["pitches", "durations"],
        dynamic_axes={
            "phonemes": {1: "inLength"},
            "accents": {1: "inLength"},
            "pitches": {1: "outLength"},
            "durations": {1: "outLength"}
        },
        opset_version=OPSET,
    )

    pitches, durations = variance_model(*variance_input)
    embedber_input = (phonemes, pitches.squeeze(0).transpose(0, 1), speakers)
    torch.onnx.export(
        embedder_model,
        embedber_input,
        "embedder_model.onnx",
        input_names=["phonemes", "pitches", "speakers"],
        output_names=["feature_embedded"],
        dynamic_axes={
            "phonemes": {1: "inLength"},
            "pitches": {1: "inLength"},
            "feature_embedded": {1: "outLength"},
        },
        opset_version=OPSET,
    )
    embedded_tensor = embedder_model(*embedber_input)
    durations = (durations * (48000 / 256)).to(torch.int64).transpose(1, 2).squeeze(0)
    torch.onnx.export(
        gaussian_model,
        (
            embedded_tensor,
            durations,
        ),
        "gaussian_model.onnx",
        input_names=["embedded_tensor", "durations"],
        output_names=["length_regulated_tensor"],
        dynamic_axes={
            "embedded_tensor": {1: "length"},
            "durations": {1: "length"},
            "length_regulated_tensor": {1: "outLength"},
        },
        opset_version=OPSET,
    )

    torch.onnx.export(
        decoder_model,
        (
            embedded_tensor,
        ),
        "decoder_model.onnx",
        input_names=["length_regulated_tensor"],
        output_names=["wav"],
        dynamic_axes={
            "length_regulated_tensor": {1: "length"},
            "wav": {1: "outLength"},
        },
        opset_version=OPSET,
    )
