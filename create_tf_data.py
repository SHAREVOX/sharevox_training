import argparse
import os

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from modules.length_regulator import LengthRegulator
from utils.model import Config, get_model
from utils.tools import to_device, ReProcessedItemTorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(restore_step: int, speaker_num, config: Config):
    # Prepare model
    _, embedder_model, decoder_model, _ = get_model(restore_step, config, device, speaker_num)
    embedder_model = nn.DataParallel(embedder_model)
    decoder_model = nn.DataParallel(decoder_model)
    length_regulator = nn.DataParallel(LengthRegulator())
    # Get dataset
    dataset = Dataset(
        "train.txt", config["preprocess"], config["train"], sort=False, drop_last=False
    )
    batch_size = config["train"]["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [to_device(d, device) for d in dataset.collate_fn(x)],
    )

    tf_data_path = config["train"]["path"]["tf_data_path"]
    inner_bar = tqdm(total=len(dataset), desc="Creating TF Data...")
    for batchs in loader:
        for batch in batchs:
            batch: ReProcessedItemTorch
            (
                ids,
                speakers,
                phonemes,
                phoneme_lens,
                max_phoneme_len,
                accents,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
                durations,
            ) = batch
            with torch.no_grad():
                feature_embedded = embedder_model(
                    phonemes=phonemes,
                    pitches=pitches,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                )
                length_regulated_tensor = length_regulator(
                    xs=feature_embedded,
                    ds=durations,
                )
                _, postnet_outputs = decoder_model(
                    length_regulated_tensor=length_regulated_tensor,
                    mel_lens=mel_lens,
                )
            for i, id in enumerate(ids):
                filename = f"{id}.npy"
                np.save(os.path.join(tf_data_path, filename), postnet_outputs[i].cpu().numpy())
                inner_bar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_num", type=int, default=10)
    args = parser.parse_args()

    config: Config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    main(args.restore_step, args.speaker_num, config)
