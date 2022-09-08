import torch
import numpy as np

from dataset import TrainConfig
from modules.fastspeech2 import PitchAndDurationPredictor, PitchAndDurationExtractor, MelSpectrogramDecoder, ModelConfig, FeatureEmbedder


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(
        self,
        variance_model: PitchAndDurationPredictor,
        embedder_model: FeatureEmbedder,
        decoder_model: MelSpectrogramDecoder,
        extractor_model: PitchAndDurationExtractor,
        train_config: TrainConfig,
        model_config: ModelConfig,
        current_step: int
    ):

        self._variance_optimizer = torch.optim.Adam(
            variance_model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self._embedder_optimizer = torch.optim.Adam(
            embedder_model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self._decoder_optimizer = torch.optim.Adam(
            decoder_model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self._extractor_optimizer = torch.optim.Adam(
            extractor_model.parameters(),
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )

        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step
        self.init_lr = np.power(model_config["variance_encoder"]["hidden"], -0.5)

    def step_and_update_lr(self) -> None:
        self._update_learning_rate()
        self._variance_optimizer.step()
        self._embedder_optimizer.step()
        self._decoder_optimizer.step()
        self._extractor_optimizer.step()

    def zero_grad(self) -> None:
        # print(self.init_lr)
        self._variance_optimizer.zero_grad()
        self._embedder_optimizer.zero_grad()
        self._decoder_optimizer.zero_grad()
        self._extractor_optimizer.zero_grad()

    def load_state_dict(self, path: dict) -> None:
        self._variance_optimizer.load_state_dict(path["variance"])
        self._embedder_optimizer.load_state_dict(path["embedder"])
        self._decoder_optimizer.load_state_dict(path["decoder"])
        self._extractor_optimizer.load_state_dict(path["extractor"])

    def _get_lr_scale(self) -> None:
        lr = np.min(
            [
                np.power(self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ]
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr = lr * self.anneal_rate
        return lr

    def _update_learning_rate(self) -> None:
        """ Learning rate scheduling per step """
        self.current_step += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._variance_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self._embedder_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self._decoder_optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in self._extractor_optimizer.param_groups:
            param_group["lr"] = lr
