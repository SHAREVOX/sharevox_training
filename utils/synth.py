import time
from typing import List, Tuple, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, LongTensor

import fregan
from models.fastspeech2 import VocoderGenerator
from utils.model import Config
from utils.plot import plot_mel


def expand(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def vocoder_infer(
    mels: torch.Tensor,
    vocoder: VocoderGenerator,
    config: Config,
    lengths: Optional[int] = None
) -> List[np.ndarray]:
    with torch.no_grad():
        start = time.time()
        if config["model"]["vocoder_type"] == "melgan":
            wavs = vocoder.inference(mels[0]).unsqueeze(0)
        else:
            wavs = vocoder(mels).squeeze(1)
        print("RTF:", f"{time.time() - start}s")

    wavs = (
        wavs.cpu().numpy()
        * config["preprocess"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs: List[np.ndarray] = [wav for wav in wavs]

    if lengths is not None:
        for i in range(len(mels)):
            wavs[i] = wavs[i][: lengths[i]]

    return wavs


def synth_one_sample(
    ids: List[str],
    duration_targets: Tensor,
    pitch_targets: Tensor,
    mel_targets: Tensor,
    mel_predictions: Tensor,
    phoneme_lens: LongTensor,
    mel_lens: LongTensor,
    vocoder: fregan.Generator,
    config: Config,
    synthesis_target: bool = False
) -> Tuple[plt.Figure, Optional[np.ndarray], np.ndarray, str]:
    basename = ids[0]
    phoneme_len = phoneme_lens[0].item()
    mel_len = mel_lens[0].item()
    mel_target = mel_targets[0, :mel_len].detach().transpose(0, 1)
    mel_prediction = mel_predictions[0, :mel_len].detach().transpose(0, 1)
    duration_target = duration_targets[0, :phoneme_len].detach().cpu().numpy()
    pitch = pitch_targets[0, :phoneme_len].detach().cpu().numpy()
    pitch = expand(pitch, duration_target)

    fig = plot_mel(
        [
            (mel_prediction.cpu().numpy(), pitch),
            (mel_target.cpu().numpy(), pitch),
        ],
        ["Synthesized Spectrogram", "Ground-Truth Spectrogram"],
    )

    wav_reconstruction = None
    if synthesis_target:
        wav_reconstruction = vocoder_infer(
            mel_target.unsqueeze(0),
            vocoder,
            config,
        )[0]
    wav_prediction = vocoder_infer(
        mel_prediction.unsqueeze(0),
        vocoder,
        config,
    )[0]

    return fig, wav_reconstruction, wav_prediction, basename
