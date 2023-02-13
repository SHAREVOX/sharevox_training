import argparse
import json
import os

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Dataset
from models.length_regulator import LengthRegulator
from preprocessor import get_tgt_and_wav
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
    batch_size = 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [to_device(d, device) for d in dataset.collate_fn(x)],
    )

    tf_data_path = config["train"]["path"]["tf_data_path"]
    out_dir = config["preprocess"]["path"]["preprocessed_path"]
    with open(os.path.join(out_dir, "speakers.json")) as f:
        speakers_data = json.load(f)
    os.makedirs(os.path.join(tf_data_path, "wav"), exist_ok=True)
    os.makedirs(os.path.join(tf_data_path, "mel"), exist_ok=True)
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
                speaker = None
                for k, v in speakers_data.items():
                    if v == int(speakers[i]):
                        speaker = k
                        break
                filename = speaker + "_" + id
                wav_filepath = os.path.join(tf_data_path, "wav", f"{filename}.wav")
                mel_filepath = os.path.join(tf_data_path, "mel", f"{filename}.npy")
                _, _, cut_wav = get_tgt_and_wav(config["preprocess"], speaker, id)
                wavfile.write(wav_filepath, config["preprocess"]["audio"]["sampling_rate"], cut_wav)
                np.save(mel_filepath, postnet_outputs[i].cpu().numpy().T)
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
