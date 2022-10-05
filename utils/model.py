import json
import os
from typing import Optional, Union, Tuple, TypedDict, Literal, overload

import torch
from torch import nn, device as TorchDevice

from dataset import TrainConfig
from modules.jets import PitchAndDurationPredictor, MelSpectrogramDecoder, \
    ModelConfig, FeatureEmbedder, VocoderType, VocoderGenerator, VocoderMultiPeriodDiscriminator, \
    VocoderMultiScaleDiscriminator
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
    VocoderGenerator,
    VocoderMultiPeriodDiscriminator,
    VocoderMultiScaleDiscriminator,
    ScheduledOptim,
    int,
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
    VocoderGenerator,
    None,
    None,
    None,
    int,
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
    decoder_model = MelSpectrogramDecoder(config["model"], config["preprocess"]["mel"]["n_mel_channels"]).to(device)
    vocoder_type = config["model"]["vocoder_type"]
    hidden_size = config["preprocess"]["mel"]["n_mel_channels"]

    if vocoder_type == "fregan":
        import fregan
        generator_model = fregan.Generator(config["model"]["vocoder"], hidden_size)
        mpd_model = fregan.ResWiseMultiPeriodDiscriminator()
        msd_model = fregan.ResWiseMultiScaleDiscriminator()
    elif vocoder_type == "hifigan":
        import hifigan
        generator_model = hifigan.Generator(config["model"]["vocoder"], hidden_size)
        mpd_model = hifigan.MultiPeriodDiscriminator()
        msd_model = hifigan.MultiScaleDiscriminator()
    else:
        raise Exception(f"Unsupported vocoder: {vocoder_type}")
    generator_model.to(device)
    mpd_model.to(device)
    msd_model.to(device)

    epoch = -1
    if restore_step:
        ckpt_path = os.path.join(
            config["train"]["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        variance_model.load_state_dict(ckpt["variance_model"])
        embedder_model.load_state_dict(ckpt["embedder_model"])
        decoder_model.load_state_dict(ckpt["decoder_model"])
        generator_model.load_state_dict(ckpt["generator_model"])
        mpd_model.load_state_dict(ckpt["mpd_model"])
        msd_model.load_state_dict(ckpt["msd_model"])
        epoch = ckpt["epoch"]

    if train:
        scheduled_optim = ScheduledOptim(
            variance_model, embedder_model, decoder_model, generator_model, mpd_model, msd_model, config["train"], epoch
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        variance_model.train()
        embedder_model.train()
        decoder_model.train()
        generator_model.train()
        mpd_model.train()
        msd_model.train()
        return variance_model, embedder_model, decoder_model, generator_model, mpd_model, msd_model, scheduled_optim, epoch

    variance_model.eval()
    embedder_model.eval()
    decoder_model.eval()
    generator_model.eval()
    generator_model.remove_weight_norm()
    variance_model.requires_grad_ = False
    embedder_model.requires_grad_ = False
    decoder_model.requires_grad_ = False
    generator_model.requires_grad_ = False
    return variance_model, embedder_model, decoder_model, None, None, None, epoch


def get_param_num(model: nn.Module) -> int:
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(device: TorchDevice, type: VocoderType = "fregan") -> VocoderGenerator:
    if type == "fregan":
        import fregan
        config = fregan.Config()
        vocoder = fregan.Generator(config)
        ckpt = torch.load(f"fregan/g_0003000.pth.tar", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
    elif type == "hifigan":
        import hifigan
        config = hifigan.Config()
        vocoder = hifigan.Generator(config)
        ckpt = torch.load(f"hifigan/g_00445000", map_location=device)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
    else:
        raise Exception(f"Unsupported vocoder: {type}")
    vocoder.to(device)
    return vocoder
