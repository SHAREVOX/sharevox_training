from typing import List, Tuple

from matplotlib import pyplot as plt, use as matplotlib_use
import numpy as np
from torch import Tensor, LongTensor

# https://python-climbing.com/runtimeerror_main_thread_is_not_in_main_loop/
matplotlib_use('Agg')

def plot_mel(
    data: List[Tuple[np.ndarray, np.ndarray]], titles: List[str]
) -> plt.Figure:
    plot_data: Tuple[plt.Figure, plt.Axes] = plt.subplots(len(data), 1, squeeze=False)
    fig, axes = plot_data
    if titles is None:
        titles = [None for i in range(len(data))]

    def add_axis(fig: plt.Figure, old_ax: plt.Axes) -> plt.Axes:
        ax = fig.add_axes(old_ax.get_position(), anchor="W")
        ax.set_facecolor("None")
        return ax

    for i in range(len(data)):
        mel, pitch = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

        ax1 = add_axis(fig, axes[i][0])
        ax1.plot(pitch, color="tomato")
        ax1.set_xlim(0, mel.shape[1])
        ax1.set_ylabel("F0", color="tomato")
        ax1.tick_params(
            labelsize="x-small", colors="tomato", bottom=False, labelbottom=False
        )

    return fig


def expand(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def plot_one_sample(
    ids: List[str],
    duration_targets: Tensor,
    pitch_targets: Tensor,
    mel_targets: Tensor,
    mel_predictions: Tensor,
    phoneme_lens: LongTensor,
    mel_lens: LongTensor,
) -> Tuple[plt.Figure, str]:
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

    return fig, basename
