import os
from typing import Optional, Union, Tuple, TypedDict, overload

import torch

from dataset import TrainConfig
from modules.fastspeech2 import PitchAndDurationPredictor, MelSpectrogramDecoder, ModelConfig
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
) -> Tuple[PitchAndDurationPredictor, MelSpectrogramDecoder, ScheduledOptim]:
    pass


@overload
def get_model(
    restore_step: int,
    config: Config,
    device: torch.device,
    speaker_num: int,
    train: False,
) -> Tuple[PitchAndDurationPredictor, MelSpectrogramDecoder, None]:
    pass


def get_model(
    restore_step: int,
    config: Config,
    device: torch.device,
    speaker_num: int,
    train: bool = False,
):
    variance_model = PitchAndDurationPredictor(config["model"], speaker_num).to(device)
    decoder_model = MelSpectrogramDecoder(config["model"], speaker_num).to(device)
    if restore_step:
        ckpt_path = os.path.join(
            config["train"]["path"]["ckpt_path"],
            "{}.pth.tar".format(restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        variance_model.load_state_dict(ckpt["variance_model"])
        decoder_model.load_state_dict(ckpt["decoder_model"])

    if train:
        scheduled_optim = ScheduledOptim(
            variance_model, decoder_model, config["train"], config["model"], restore_step
        )
        if restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        variance_model.train()
        decoder_model.train()
        return variance_model, decoder_model, scheduled_optim

    variance_model.eval()
    decoder_model.eval()
    variance_model.requires_grad_ = False
    decoder_model.requires_grad_ = False
    return variance_model, decoder_model, None


def get_param_num(model: nn.DataParallel) -> int:
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

