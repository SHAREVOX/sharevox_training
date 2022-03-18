import argparse
import os

import torch
import yaml
from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import fregan
from dataset import Dataset
from modules.loss import FastSpeech2Loss
from utils.logging import log
from utils.model import Config, get_model, get_param_num, get_vocoder
from utils.synth import synth_one_sample
from utils.tools import to_device, ReProcessedItemTorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    variance_model: DataParallel,  # PitchAndDurationPredictor
    decoder_model: DataParallel,  # MelSpectrogramDecoder
    step: int,
    config: Config,
    logger: SummaryWriter,
    vocoder: fregan.Generator = None,
):
    # Get dataset
    dataset = Dataset(
        "val.txt", config["preprocess"], config["train"], sort=False, drop_last=False
    )
    batch_size = config["train"]["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: [to_device(d, device) for d in dataset.collate_fn(x)],
    )

    # Get loss function
    Loss = FastSpeech2Loss().to(device)

    # Evaluation
    loss_sums = [0 for _ in range(5)]
    count = 0
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
                log_pitch_outputs, log_duration_outputs = variance_model(
                    phonemes=phonemes,
                    accents=accents,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                )

                outputs, postnet_outputs = decoder_model(
                    phonemes=phonemes,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                    pitches=pitches,
                    durations=durations,
                    mel_lens=mel_lens,
                )

                # Cal Loss
                losses = Loss(
                    outputs=outputs,
                    postnet_outputs=postnet_outputs,
                    log_duration_outputs=log_duration_outputs,
                    log_pitch_outputs=log_pitch_outputs,
                    mel_targets=mels,
                    duration_targets=durations,
                    pitch_targets=pitches,
                    input_lens=phoneme_lens,
                    output_lens=mel_lens,
                )

                for i in range(len(losses)):
                    loss_sums[i] += losses[i].item() * len(batch[0])

            if count < 5:
                fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                    ids=ids,
                    duration_targets=durations,
                    pitch_targets=pitches,
                    mel_targets=mels,
                    mel_predictions=postnet_outputs,
                    phoneme_lens=phoneme_lens,
                    mel_lens=mel_lens,
                    vocoder=vocoder,
                    config=config,
                    synthesis_target=(step == int(config["train"]["step"]["val_step"]))
                )

                log(
                    logger,
                    fig=fig,
                    tag="Validation/{}".format(tag),
                    step=step,
                )
                sampling_rate = config["preprocess"]["audio"]["sampling_rate"]
                log(
                    logger,
                    audio=wav_reconstruction,
                    sampling_rate=sampling_rate,
                    tag="Validation/{}_gt".format(tag),
                )
                log(
                    logger,
                    audio=wav_prediction,
                    sampling_rate=sampling_rate,
                    tag="Validation/{}_synthesized".format(tag),
                    step=step,
                )
            count += 1

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    log(logger, step, losses=loss_means)

    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Duration Loss: {:.4f}".format(
        *([step] + loss_means)
    )

    return message


def main(restore_step: int, speaker_num, config: Config):
    print("Prepare training ...")

    # Get dataset
    dataset = Dataset(
        "train.txt", config["preprocess"], config["train"], sort=True, drop_last=True
    )
    batch_size = config["train"]["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=lambda x: [to_device(d, device) for d in dataset.collate_fn(x)],
    )

    # Prepare model
    variance_model, decoder_model, optimizer = get_model(restore_step, config, device, speaker_num, train=True)
    variance_model = nn.DataParallel(variance_model)
    decoder_model = nn.DataParallel(decoder_model)
    num_param = get_param_num(variance_model) + get_param_num(decoder_model)
    Loss = FastSpeech2Loss().to(device)
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(device)

    # Init logger
    for p in config["train"]["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(config["train"]["path"]["log_path"], "train")
    val_log_path = os.path.join(config["train"]["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = config["train"]["optimizer"]["grad_acc_step"]
    grad_clip_thresh = config["train"]["optimizer"]["grad_clip_thresh"]
    total_step = config["train"]["step"]["total_step"]
    log_step = config["train"]["step"]["log_step"]
    save_step = config["train"]["step"]["save_step"]
    synth_step = config["train"]["step"]["synth_step"]
    val_step = config["train"]["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
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

                # Forward
                log_pitch_outputs, log_duration_outputs = variance_model(
                    phonemes=phonemes,
                    accents=accents,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                )
                outputs, postnet_outputs = decoder_model(
                    phonemes=phonemes,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                    pitches=pitches,
                    durations=durations,
                    mel_lens=mel_lens,
                )

                # Cal Loss
                losses = Loss(
                    outputs=outputs,
                    postnet_outputs=postnet_outputs,
                    log_duration_outputs=log_duration_outputs,
                    log_pitch_outputs=log_pitch_outputs,
                    mel_targets=mels,
                    duration_targets=durations,
                    pitch_targets=pitches,
                    input_lens=phoneme_lens,
                    output_lens=mel_lens,
                )
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(variance_model.parameters(), grad_clip_thresh)
                    nn.utils.clip_grad_norm_(decoder_model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Duration Loss: {:.4f}, Pitch Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        ids=ids,
                        duration_targets=durations,
                        pitch_targets=pitches,
                        mel_targets=mels,
                        mel_predictions=postnet_outputs,
                        phoneme_lens=phoneme_lens,
                        mel_lens=mel_lens,
                        vocoder=vocoder,
                        config=config
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = config["preprocess"]["audio"]["sampling_rate"]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )

                if step % val_step == 0:
                    variance_model.eval()
                    decoder_model.eval()
                    message = evaluate(variance_model, decoder_model, step, config, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    variance_model.train()
                    decoder_model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "variance_model": variance_model.module.state_dict(),
                            "decoder_model": decoder_model.module.state_dict(),
                            "optimizer": {
                                "variance": optimizer._variance_optimizer.state_dict(),
                                "decoder": optimizer._decoder_optimizer.state_dict(),
                            },
                        },
                        os.path.join(
                            config["train"]["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_num", type=int, default=1)
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to config yaml"
    )
    args = parser.parse_args()

    config: Config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    main(args.restore_step, args.speaker_num, config)
