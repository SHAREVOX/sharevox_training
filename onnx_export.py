import argparse
import os
import json
from typing import Union

import torch
import yaml

from config import Config
from models.fastspeech.length_regulator import LengthRegulator
from models.tts import VITS, JETS
from models.upsampler.utils import dilated_factor
from text import _symbol_to_id

try:
    from torch.onnx._constants import ONNX_DEFAULT_OPSET
    onnx_stable_opsets = [ONNX_DEFAULT_OPSET]
except:
    try:
        from torch.onnx._constants import onnx_stable_opsets
    except:
        from torch.onnx.symbolic_helper import _onnx_stable_opsets as onnx_stable_opsets

from utils.checkpoint import load_checkpoint, latest_checkpoint_path

OPSET = onnx_stable_opsets[-1]


class VariancePredictor(torch.nn.Module):
    def __init__(self, generator: Union[VITS, JETS]) -> None:
        super().__init__()
        self.generator = generator
        self.sampling_rate = generator.sampling_rate
        self.hop_length = generator.hop_length
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean

    def forward(self, phonemes, accents, speakers):
        g = self.generator.emb_g(speakers).unsqueeze(-1)  # [b, h, 1]

        pred_pitches, pred_durations = self.generator.forward_variance(phonemes, accents, g=g)

        durations = torch.clamp((torch.exp(pred_durations) - 1) / (self.sampling_rate / self.hop_length), min=0.01)
        pitches = pred_pitches * self.pitch_std + self.pitch_mean
        pitches[pitches < 1] = 1
        pitches[pitches > 750] = 1
        pitches = torch.log(pitches)

        return pitches.transpose(1, 2), durations.transpose(1, 2)


class FeatureEmbedder(torch.nn.Module):
    def __init__(self, generator: Union[VITS, JETS]) -> None:
        super().__init__()
        self.enc_p = generator.enc_p
        self.pitch_embedding = generator.pitch_embedding
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean

    @torch.no_grad()
    def forward(self, phonemes):
        x, _ = self.enc_p(phonemes)
        return x.transpose(1, 2)


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


class Decoder(torch.nn.Module):
    def __init__(self, generator: Union[VITS, JETS]) -> None:
        super().__init__()
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean
        self.generator = generator

    @torch.no_grad()
    def forward(self, x, pitch, speaker):
        g = self.generator.emb_g(speaker).unsqueeze(-1)

        pitch = pitch.unsqueeze(1)
        unvoice_mask = pitch == 0
        pitch = (torch.exp(pitch) - self.pitch_mean) / self.pitch_std
        pred_frame_pitches = self.generator.forward_pitch_upsampler(pitch, g=g)
        smoothly_pitches = self.generator.pitch_smoothly(pred_frame_pitches, unvoice_mask)

        if hasattr(self.generator, "flow"):
            _, m_p, logs_p, _ = self.generator.frame_prior_network(x)
            z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p)
            y_mask = torch.ones_like(pitch)
            z = self.generator.flow(z_p, y_mask, g=g, inverse=True)
            wav, _ = self.generator.forward_upsampler(z, smoothly_pitches)
        else:
            z, _ = self.generator.frame_prior_network(x, None)
            z = z.transpose(1, 2)
            wav, _ = self.generator.forward_upsampler(z, smoothly_pitches, g=g)

        return wav.squeeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="YAML file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-s", "--speakers", type=int, default=10, help="speaker count")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    with open(args.config, "r") as f:
        config: Config = yaml.load(f, Loader=yaml.FullLoader)

    preprocessed_path = config["preprocess"]["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "stats.json")) as f:
        stats_text = f.read()
    stats_json = json.loads(stats_text)
    pitch_mean, pitch_std = stats_json["pitch"][2:]

    device = torch.device("cpu")
    length_regulator = LengthRegulator().to(device)
    gaussian_model = GaussianUpsampling().to(device)
    gaussian_model.eval()

    model_config = config["model"]
    model_type = model_config["model_type"]
    
    if model_type == "vits":
        Model = VITS
    elif model_type == "jets":
        Model = JETS
    else:
        raise Exception(f"Unknown model type: {model_type}")

    net_g = Model(
        model_config,
        spec_channels=config["preprocess"]["stft"]["filter_length"] // 2 + 1,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        sampling_rate=config["preprocess"]["audio"]["sampling_rate"],
        hop_length=config["preprocess"]["stft"]["hop_length"],
        n_speakers=args.speakers,
        onnx=True,
    ).to(device)

    try:
        _, _, _, epoch_str, step = load_checkpoint(
            latest_checkpoint_path(model_dir, "G_*.pth"), net_g, None
        )
    except:
        epoch_str = 1
        step = 0

    variance = VariancePredictor(net_g).to(device).eval()
    feature_embedder = FeatureEmbedder(net_g).to(device).eval()
    decoder = Decoder(net_g).to(device).eval()

    x = torch.tensor([_symbol_to_id[p] for p in "pau d o r e m i f a s o pau".split(" ")]).unsqueeze(0).to(device)
    accent = torch.tensor([2, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 2]).unsqueeze(0).to(device)
    # x_len = torch.tensor([x.shape[1]])
    duration = torch.tensor([50, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 50]).unsqueeze(0).to(device)
    pitch = torch.tensor([0.0, 5.56, 5.56, 5.68, 5.68, 5.8, 5.8, 5.86, 5.86, 5.97, 5.97, 0.0]).unsqueeze(0).to(device)
    speaker = torch.tensor([0]).to(device)

    torch.onnx.export(
        variance,
        (x, accent, speaker),
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

    feature_embedded = feature_embedder(x)
    feature_embedded_lr = length_regulator(feature_embedded, duration)
    pitch_lr = length_regulator(pitch, duration)

    torch.onnx.export(
        feature_embedder,
        (x),
        "embedder_model.onnx",
        input_names=["phonemes"],
        output_names=["feature_embedded"],
        dynamic_axes={
            "phonemes": {1: "inLength"},
            "feature_embedded": {1: "outLength"},
        },
        opset_version=OPSET,
    )

    # torch.onnx.export(
    #     gaussian_model,
    #     (
    #         feature_embedded,
    #         duration,
    #     ),
    #     "gaussian_model.onnx",
    #     input_names=["embedded_tensor", "durations"],
    #     output_names=["length_regulated_tensor"],
    #     dynamic_axes={
    #         "embedded_tensor": {1: "length"},
    #         "durations": {1: "length"},
    #         "length_regulated_tensor": {1: "outLength"},
    #     },
    #     opset_version=OPSET,
    # )

    torch.onnx.export(
        decoder,
        (
            feature_embedded_lr, pitch_lr, speaker
        ),
        "decoder_model.onnx",
        input_names=["length_regulated_tensor", "pitches", "speakers"],
        output_names=["wav"],
        dynamic_axes={
            "length_regulated_tensor": {1: "length"},
            "pitches": {1: "length"},
            "wav": {1: "outLength"},
        },
        opset_version=OPSET,
    )
