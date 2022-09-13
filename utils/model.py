import json
import os
from typing import Optional, Union, Tuple, TypedDict, Literal, overload

import torch
from torch import nn

from dataset import TrainConfig
from modules.jets import PitchAndDurationPredictor, PitchAndDurationExtractor, MelSpectrogramDecoder, \
    ModelConfig, FeatureEmbedder, VocoderGenerator, VocoderMultiPeriodDiscriminator, VocoderMultiScaleDiscriminator
from modules.optimizer import ScheduledOptim
from preprocessor import PreProcessConfig


class Config(TypedDict):
    preprocess: PreProcessConfig
    model: ModelConfig
    train: TrainConfig


@overload
def get_model(
    restore_step: int,
    config: Config,
    device: torch.device,
    speaker_num: int,
    train: True,
) -> Tuple[
    PitchAndDurationPredictor,
    FeatureEmbedder,
    MelSpectrogramDecoder,
    PitchAndDurationExtractor,
    VocoderGenerator,
    VocoderMultiPeriodDiscriminator,
    VocoderMultiScaleDiscriminator,
    ScheduledOptim
]:
    pass


@overload
def get_model(
    restore_step: int,
    config: Config,
    device: torch.device,
    speaker_num: int,
    train: False,
) -> Tuple[
    PitchAndDurationPredictor,
    FeatureEmbedder,
    MelSpectrogramDecoder,
    PitchAndDurationExtractor,
    VocoderGenerator,
    None,
    None,
    None
]:
    pass


def get_model(
    restore_step: int,
    config: Config,
    device: torch.device,
    speaker_num: int,
    train: bool = False,
):
    preprocessed_path = config["preprocess"]["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "stats.json")) as f:
        stats_text = f.read()
    stats_json = json.loads(stats_text)
    pitch_min, pitch_max = stats_json["pitch"][:2]
    variance_model = PitchAndDurationPredictor(config["model"], speaker_num).to(device)
    embedder_model = FeatureEmbedder(config["model"], speaker_num, pitch_min, pitch_max).to(device)
    decoder_model = MelSpectrogramDecoder(config["model"]).to(device)
    extractor_model = PitchAndDurationExtractor(config["model"]).to(device)
    vocoder_type = config["model"]["vocoder_type"]
    if vocoder_type == "fregan":
        import fregan
        generator_model = fregan.Generator(config["model"]["vocoder"])
        mpd_model = fregan.ResWiseMultiPeriodDiscriminator()
        msd_model = fregan.ResWiseMultiScaleDiscriminator()
    elif vocoder_type == "hifigan":
        import hifigan
        generator_model = hifigan.Generator(config["model"]["vocoder"])
        mpd_model = hifigan.MultiPeriodDiscriminator()
        msd_model = hifigan.MultiScaleDiscriminator()
    else:
        raise Exception(f"Unsupported vocoder: {vocoder_type}")
    generator_model.to(device)
    mpd_model.to(device)
    msd_model.to(device)

    if restore_step:
        ckpt_path = os.path.join(
            config["train"]["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        variance_model.load_state_dict(ckpt["variance_model"])
        embedder_model.load_state_dict(ckpt["embedder_model"])
        decoder_model.load_state_dict(ckpt["decoder_model"])
        extractor_model.load_state_dict(ckpt["extractor_model"])
        generator_model.load_state_dict(ckpt["generator_model"])
        mpd_model.load_state_dict(ckpt["mpd_model"])
        msd_model.load_state_dict(ckpt["msd_model"])

    if train:
        scheduled_optim = ScheduledOptim(
            variance_model,
            embedder_model,
            decoder_model,
            extractor_model,
            generator_model,
            mpd_model,
            msd_model,
            config["train"],
            config["model"],
            restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        variance_model.train()
        embedder_model.train()
        decoder_model.train()
        extractor_model.train()
        generator_model.train()
        mpd_model.train()
        msd_model.train()
        return variance_model, embedder_model, decoder_model, extractor_model, generator_model, mpd_model, msd_model, scheduled_optim

    variance_model.eval()
    embedder_model.eval()
    decoder_model.eval()
    extractor_model.eval()
    generator_model.eval()
    variance_model.requires_grad_ = False
    embedder_model.requires_grad_ = False
    decoder_model.requires_grad_ = False
    generator_model.requires_grad_ = False
    return variance_model, embedder_model, decoder_model, extractor_model, generator_model, None, None, None


def get_param_num(model: nn.Module) -> int:
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
