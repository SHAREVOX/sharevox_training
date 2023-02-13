from typing import Optional

import numpy as np
import matplotlib.pylab as plt

MATPLOTLIB_FLAG = False


def plot_spectrogram_to_numpy(spectrogram: np.ndarray) -> np.ndarray:
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment: np.ndarray, info: Optional[str] = None) -> np.ndarray:
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_f0_to_numpy(
    f0_gt: np.ndarray,
    f0_avg_regulated: Optional[np.ndarray] = None,
    f0_pred_regulated: Optional[np.ndarray] = None,
    f0_pred: Optional[np.ndarray] = None
):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True

    fig = plt.figure()
    plt.plot(f0_gt, color="r", label="gt")
    if f0_avg_regulated is not None:
        plt.plot(f0_avg_regulated, color="b", label="gt_avg")
    if f0_pred_regulated is not None:
        plt.plot(f0_pred_regulated, color="orange", label="pred_avg")
    if f0_pred is not None:
        plt.plot(f0_pred, color="green", label="pred")
    plt.ylim(0, 800)
    plt.legend()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


