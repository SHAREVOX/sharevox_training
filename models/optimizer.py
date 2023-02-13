import torch
import numpy as np

from dataset import TrainConfig
from models.fastspeech2 import PitchAndDurationPredictor, PitchAndDurationExtractor, MelSpectrogramDecoder, ModelConfig, FeatureEmbedder


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self,
        variance_model: PitchAndDurationPredictor,
        embedder_model: FeatureEmbedder,
        decoder_model: MelSpectrogramDecoder,
        train_config: TrainConfig,
        model_config: ModelConfig,
        last_epoch: int
    ):
        betas = train_config["optimizer"]["betas"]
        lr = train_config["optimizer"]["learning_rate"]
        self.lr_decay = train_config["optimizer"]["lr_decay"]
        self.last_epoch = last_epoch

        self._variance_optimizer = torch.optim.AdamW(variance_model.parameters(), lr=lr, betas=betas)
        self._embedder_optimizer = torch.optim.AdamW(embedder_model.parameters(), lr=lr, betas=betas)
        self._decoder_optimizer = torch.optim.AdamW(decoder_model.parameters(), lr=lr, betas=betas)

        try:
            self._set_scheduler()
        except KeyError:
            self._variance_scheduler = None
            self._embedder_scheduler = None
            self._decoder_scheduler = None

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

    def step_and_update_lr(self) -> None:
        self._variance_optimizer.step()
        self._embedder_optimizer.step()
        self._decoder_optimizer.step()
        self._update_learning_rate()

    def zero_grad(self) -> None:
        # print(self.init_lr)
        self._variance_optimizer.zero_grad()
        self._embedder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()

    def load_state_dict(self, path: dict) -> None:
        self._variance_optimizer.load_state_dict(path["variance"])
        self._embedder_optimizer.load_state_dict(path["embedder"])
        self._decoder_optimizer.load_state_dict(path["decoder"])
        self._set_scheduler()

    def _update_learning_rate(self) -> None:
        """ Learning rate scheduling per step """
        self._variance_scheduler.step()
        self._embedder_scheduler.step()
        self._decoder_scheduler.step()
