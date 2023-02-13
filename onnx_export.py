import argparse
import os
import json

import torch
import yaml

from config import Config
from models.fastspeech.length_regulator import LengthRegulator
from models.tts import JETS
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
    def __init__(self, generator: JETS) -> None:
        super().__init__()
        self.phoneme_embedding = generator.phoneme_embedding
        self.accent_embedding = generator.accent_embedding
        self.variance_encoder = generator.variance_encoder
        self.pitch_predictor = generator.pitch_predictor
        self.duration_predictor = generator.duration_predictor
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean
        self.sampling_rate = generator.sampling_rate
        self.hop_length = generator.hop_length

        self.emb_g = generator.emb_g

    def forward(self, phonemes, accents, speakers):
        g = self.emb_g(speakers).unsqueeze(-1)  # [b, h, 1]

        phoneme_embedding = self.phoneme_embedding(phonemes)
        accent_embedding = self.accent_embedding(accents)

        duration_hs, _ = self.variance_encoder(phoneme_embedding, None)
        pitch_hs, _ = self.variance_encoder(phoneme_embedding + accent_embedding, None)

        pred_durations = self.duration_predictor(duration_hs, g=g).transpose(1, 2)
        pred_pitches = self.pitch_predictor(pitch_hs, g=g).transpose(1, 2)

        durations = torch.clamp((torch.exp(pred_durations) - 1) / (self.sampling_rate / self.hop_length), min=0.01)
        pitches = pred_pitches * self.pitch_std + self.pitch_mean
        pitches[pitches < 1] = 1
        pitches[pitches > 750] = 1
        pitches = torch.log(pitches)

        return pitches, durations


class FeatureEmbedder(torch.nn.Module):
    def __init__(self, generator: JETS) -> None:
        super().__init__()
        self.enc_p = generator.enc_p
        self.pitch_embedding = generator.pitch_embedding
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean

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
    def __init__(self, generator: JETS) -> None:
        super().__init__()
        self.pitch_std = generator.pitch_std
        self.pitch_mean = generator.pitch_mean
        self.emb_g = generator.emb_g
        self.frame_prior_network = generator.frame_prior_network
        self.signal_generator = generator.signal_generator
        self.dec = generator.dec
        self.dense_factors = generator.dense_factors
        self.prod_upsample_scales = generator.prod_upsample_scales.tolist()
        self.sampling_rate = generator.sampling_rate
        self.pitch_embedding = generator.pitch_embedding
        self.pitch_upsampler = generator.pitch_upsampler
        self.f0_upsampler = generator.f0_upsampler

    @torch.no_grad()
    def forward(self, x, pitch, speaker):
        g = self.emb_g(speaker).unsqueeze(-1)

        pitch = (torch.exp(pitch) - self.pitch_mean) / self.pitch_std

        pitch_embed = self.pitch_embedding(pitch)
        pred_frame_pitches = self.pitch_upsampler(pitch_embed, g=g)
        frame_pitches = pred_frame_pitches * self.pitch_std + self.pitch_mean
        frame_pitches[frame_pitches < 1] = 1
        downsampled_pitches = torch.cat((frame_pitches[:, :, ::4], torch.ones(1, 1, 4)), dim=2)
        upsampled_pitches = self.f0_upsampler(downsampled_pitches)
        linear_pitches = upsampled_pitches[:, :, :pitch.shape[1]]
        linear_pitches[pitch.unsqueeze(1) == 1] = 1

        z, _ = self.frame_prior_network(x, None)

        dfs = []
        for df, us in zip(self.dense_factors, self.prod_upsample_scales):
            result = []
            # print(frame_pitches.shape)
            for frame_pitch in linear_pitches:
                dilated_tensor = dilated_factor(frame_pitch, self.sampling_rate, df)
                result += [
                    torch.stack([dilated_tensor for _ in range(us)], -1).reshape(dilated_tensor.shape[0], -1)
                ]
                dfs.append(torch.cat(result, dim=0).unsqueeze(1))
        sin_waves = self.signal_generator(linear_pitches)
        wav, _ = self.dec(z.transpose(1, 2), f0=sin_waves, d=dfs, g=g)
        return wav.squeeze(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.yaml",
        help="YAML file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

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

    net_g = JETS(
        config["model"],
        spec_channels=config["preprocess"]["stft"]["filter_length"] // 2 + 1,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        sampling_rate=config["preprocess"]["audio"]["sampling_rate"],
        hop_length=config["preprocess"]["stft"]["hop_length"],
        n_speakers=10,
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
