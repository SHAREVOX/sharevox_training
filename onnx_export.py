import argparse
import json
import os
from typing import List

import numpy as np
import torch
import yaml

from torch import nn, Tensor, LongTensor

from fregan import Generator
from modules.fastspeech2 import MelSpectrogramDecoder, PitchAndDurationPredictor, FeatureEmbedder
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
        pitches, log_durations = self.variance_model(phonemes, accents, speakers)
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
        feature_embedded = self.embedder_model(phonemes, pitches, speakers)
        return feature_embedded


class Decoder(nn.Module):
    def __init__(self, config: Config, decoder: MelSpectrogramDecoder, fregan: Generator):
        super(Decoder, self).__init__()

        self.max_wav_value = config["preprocess"]["audio"]["max_wav_value"]
        self.decoder = decoder
        self.fregan = fregan

    def forward(self, length_regulated_tensor: Tensor) -> Tensor:
        _, postnet_outputs = self.decoder(length_regulated_tensor)
        wavs = self.fregan(postnet_outputs[0].transpose(0, 1).unsqueeze(0)).squeeze(1)
        return wavs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_num", type=int, default=10)
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to config yaml"
    )
    args = parser.parse_args()
    config: Config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    variance_model, embedder_model, decoder_model, _ = get_model(args.restore_step, config, device, args.speaker_num, False)
    fregan_model = get_vocoder(device)
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
    torch.onnx.export(
        variance_model,
        (
            phonemes, accents, speakers
        ),
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

    pitches = torch.from_numpy(np.array([[5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]])).to(dtype=torch.float, device=device)
    embedber_input = (phonemes, pitches, speakers)
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
