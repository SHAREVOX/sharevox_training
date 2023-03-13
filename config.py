from typing import TypedDict

from models.tts import ModelConfig
from preprocessor import PreProcessConfig
from dataset import TrainConfig


class Config(TypedDict):
    preprocess: PreProcessConfig
    model: ModelConfig
    train: TrainConfig
