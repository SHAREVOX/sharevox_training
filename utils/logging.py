from typing import Optional, Sequence, Union

import numpy as np
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt


def log(
    logger: SummaryWriter,
    step: Optional[int] = None,
    losses: Optional[Sequence[Union[Tensor, np.ndarray, float]]] = None,
    fig: Optional[plt.Figure] = None,
    audio: Optional[np.ndarray] = None,
    sampling_rate: int = 48000,
    tag: str = ""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/mel_loss", losses[1], step)
        logger.add_scalar("Loss/mel_postnet_loss", losses[2], step)
        logger.add_scalar("Loss/duration_loss", losses[3], step)
        logger.add_scalar("Loss/pitch_loss", losses[4], step)
        logger.add_scalar("Loss/alignment_loss", losses[5], step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )
