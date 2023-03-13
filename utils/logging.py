import logging
from logging import handlers
import os
import sys

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class LoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(model_dir: str, filename: str = "train.log"):
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    fh = logging.FileHandler(os.path.join(model_dir, filename))
    fh.setFormatter(formatter)
    sh = LoggingHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def summarize(
    writer: SummaryWriter,
    global_step: int,
    scalars: dict = {},
    histograms: dict = {},
    images: dict = {},
    audios: dict = {},
    audio_sampling_rate: int = 22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)
