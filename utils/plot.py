from typing import List, Tuple

from matplotlib import pyplot as plt, use as matplotlib_use
import numpy as np

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
