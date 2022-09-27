import itertools

import torch
import numpy as np

from dataset import TrainConfig
from modules.fastspeech2 import PitchAndDurationPredictor, PitchAndDurationExtractor, MelSpectrogramDecoder, \
    ModelConfig, FeatureEmbedder, VocoderGenerator, VocoderMultiPeriodDiscriminator, VocoderMultiScaleDiscriminator


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self,
        variance_model: PitchAndDurationPredictor,
        embedder_model: FeatureEmbedder,
        decoder_model: MelSpectrogramDecoder,
        generator_model: VocoderGenerator,
        mpd_model: VocoderMultiPeriodDiscriminator,
        msd_model: VocoderMultiScaleDiscriminator,
        train_config: TrainConfig,
        last_epoch: int
    ):
        betas = train_config["optimizer"]["betas"]
        lr = train_config["optimizer"]["learning_rate"]
        self.lr_decay = train_config["optimizer"]["lr_decay"]
        self.last_epoch = last_epoch

        self._variance_optimizer = torch.optim.AdamW(variance_model.parameters(), lr=lr, betas=betas)
        self._embedder_optimizer = torch.optim.AdamW(embedder_model.parameters(), lr=lr, betas=betas)
        self._decoder_optimizer = torch.optim.AdamW(decoder_model.parameters(), lr=lr, betas=betas)
        self._generator_optimizer = torch.optim.AdamW(generator_model.parameters(), lr=lr, betas=betas)
        self._discriminator_optimizer = torch.optim.AdamW(
            itertools.chain(msd_model.parameters(), mpd_model.parameters()), lr=lr, betas=betas
        )

        try:
            self._set_scheduler()
        except KeyError:
            self._variance_scheduler = None
            self._embedder_scheduler = None
            self._decoder_scheduler = None
            self._generator_scheduler = None
            self._discriminator_scheduler = None

    def _set_scheduler(self) -> None:
        self._variance_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._variance_optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )
        self._embedder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._embedder_optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )
        self._decoder_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._decoder_optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )
        self._generator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._generator_optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )
        self._discriminator_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._discriminator_optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )

    def step_and_update_lr_gen(self) -> None:
        self._variance_optimizer.step()
        self._embedder_optimizer.step()
        self._decoder_optimizer.step()
        self._generator_optimizer.step()
        self._update_learning_rate_gen()

    def step_and_update_lr_disc(self) -> None:
        self._discriminator_optimizer.step()
        self._update_learning_rate_gen()

    def zero_grad_gen(self) -> None:
        self._variance_optimizer.zero_grad()
        self._embedder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()
        self._generator_optimizer.zero_grad()

    def zero_grad_disc(self) -> None:
        self._discriminator_optimizer.zero_grad()

    def load_state_dict(self, path: dict) -> None:
        self._variance_optimizer.load_state_dict(path["variance"])
        self._embedder_optimizer.load_state_dict(path["embedder"])
        self._decoder_optimizer.load_state_dict(path["decoder"])
        self._generator_optimizer.load_state_dict(path["generator"])
        self._discriminator_optimizer.load_state_dict(path["discriminator"])
        self._set_scheduler()

    def _update_learning_rate_gen(self) -> None:
        self._variance_scheduler.step()
        self._embedder_scheduler.step()
        self._decoder_scheduler.step()
        self._generator_scheduler.step()

    def _update_learning_rate_disc(self) -> None:
        self._discriminator_scheduler.step()
