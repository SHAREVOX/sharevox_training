import torch
import numpy as np

from dataset import TrainConfig
from modules.fastspeech2 import FastSpeech2, ModelConfig


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self,
        fs2_model: FastSpeech2,
        train_config: TrainConfig,
        model_config: ModelConfig,
        last_epoch: int
    ):
        betas = train_config["optimizer"]["betas"]
        lr = train_config["optimizer"]["learning_rate"]
        self.lr_decay = train_config["optimizer"]["lr_decay"]
        self.last_epoch = last_epoch

        self._optimizer = torch.optim.AdamW(fs2_model.parameters(), lr=lr, betas=betas)

        try:
            self._set_scheduler()
        except KeyError:
            self._scheduler = None

    def _set_scheduler(self) -> None:
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizer, gamma=self.lr_decay, last_epoch=self.last_epoch
        )

    def step_and_update_lr(self) -> None:
        self._optimizer.step()
        self._update_learning_rate()

    def zero_grad(self) -> None:
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path: dict) -> None:
        self._optimizer.load_state_dict(path)

    def _update_learning_rate(self) -> None:
        """ Learning rate scheduling per step """
        self._scheduler.step()
