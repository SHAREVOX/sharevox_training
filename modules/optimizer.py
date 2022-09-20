import itertools

import torch
import numpy as np

from dataset import TrainConfig
from modules.jets import PitchAndDurationPredictor, PitchAndDurationExtractor, MelSpectrogramDecoder, \
    ModelConfig, FeatureEmbedder, VocoderGenerator, VocoderMultiPeriodDiscriminator, VocoderMultiScaleDiscriminator


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self,
        variance_model: PitchAndDurationPredictor,
        embedder_model: FeatureEmbedder,
        decoder_model: MelSpectrogramDecoder,
        extractor_model: PitchAndDurationExtractor,
        generator_model: VocoderGenerator,
        mpd_model: VocoderMultiPeriodDiscriminator,
        msd_model: VocoderMultiScaleDiscriminator,
        train_config: TrainConfig,
        model_config: ModelConfig,
        last_epoch: int
    ):

        betas = train_config["optimizer"]["betas"]
        lr = train_config["optimizer"]["learning_rate"]
        lr_decay = train_config["optimizer"]["lr_decay"]

        self._variance_optimizer = torch.optim.AdamW(variance_model.parameters(), lr=lr, betas=betas)
        self._embedder_optimizer = torch.optim.AdamW(embedder_model.parameters(), lr=lr, betas=betas)
        self._decoder_optimizer = torch.optim.AdamW(decoder_model.parameters(), lr=lr, betas=betas)
        self._extractor_optimizer = torch.optim.AdamW(extractor_model.parameters(), lr=lr, betas=betas)
        self._generator_optimizer = torch.optim.AdamW(generator_model.parameters(), lr=lr, betas=betas)
        self._discriminator_optimizer = torch.optim.AdamW(
            itertools.chain(msd_model.parameters(), mpd_model.parameters()), lr=lr, betas=betas
        )

        self._variance_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._variance_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )
        self._embedder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._embedder_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )
        self._decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._decoder_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )
        self._extractor_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._extractor_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )
        self._generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._generator_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )
        self._discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._discriminator_optimizer, gamma=lr_decay, last_epoch=last_epoch
        )

    def step_and_update_lr(self) -> None:
        self._variance_optimizer.step()
        self._embedder_optimizer.step()
        self._decoder_optimizer.step()
        self._extractor_optimizer.step()
        self._generator_optimizer.step()
        self._discriminator_optimizer.step()
        self._update_learning_rate()

    def zero_grad(self) -> None:
        self._variance_optimizer.zero_grad()
        self._embedder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()
        self._extractor_optimizer.zero_grad()
        self._generator_optimizer.zero_grad()
        self._discriminator_optimizer.zero_grad()

    def load_state_dict(self, path: dict) -> None:
        self._variance_optimizer.load_state_dict(path["variance"])
        self._embedder_optimizer.load_state_dict(path["embedder"])
        self._decoder_optimizer.load_state_dict(path["decoder"])
        self._extractor_optimizer.load_state_dict(path["extractor"])
        self._generator_optimizer.load_state_dict(path["generator"])
        self._discriminator_optimizer.load_state_dict(path["discriminator"])

    def _update_learning_rate(self) -> None:
        """ Learning rate scheduling per step """
        self._variance_scheduler.step()
        self._embedder_scheduler.step()
        self._decoder_scheduler.step()
        self._extractor_scheduler.step()
        self._generator_scheduler.step()
        self._discriminator_scheduler.step()
