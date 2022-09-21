import argparse
import os

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
from torch import nn
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import Dataset
from modules.gaussian_upsampling import GaussianUpsampling
from modules.loss import VarianceLoss, DiscriminatorLoss, GeneratorLoss
from stft import mel_spectrogram
from utils.logging import log, LossDict
from utils.mask import make_non_pad_mask
from utils.model import Config, get_model, get_param_num
from utils.plot import plot_one_sample
from utils.random_segments import get_random_segments, get_segments
from utils.tools import to_device, ReProcessedItemTorch

torch.backends.cudnn.benchmark = True


def evaluate(
    variance_model: nn.Module,  # PitchAndDurationPredictor
    embedder_model: nn.Module,  # FeatureEmbedder
    decoder_model: nn.Module,  # MelSpectrogramDecoder
    generator_model: nn.Module,  # {FreGAN|HifiGAN}Generator
    length_regulator: nn.Module,
    step: int,
    config: Config,
    logger: SummaryWriter,
):
    # Get dataset
    dataset = Dataset(
        "val.txt", config["preprocess"], config["train"], sort=False, drop_last=False
    )
    batch_size = config["train"]["optimizer"]["batch_size"]
    device = torch.device('cuda:0')
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    preprocess_config = config["preprocess"]

    # Get loss function
    variance_loss = VarianceLoss().to(device)

    # Evaluation
    loss_dict: LossDict = {
        "total_loss": 0.0,
        "mel_loss": 0.0,
        "duration_loss": 0.0,
        "pitch_loss": 0.0,
        "alignment_loss": 0.0,
        "generator_loss": None,
        "discriminator_loss": None
    }

    count = 0
    for batchs in loader:
        for _batch in batchs:
            batch: ReProcessedItemTorch = to_device(_batch, device)
            (
                ids,
                speakers,
                phonemes,
                phoneme_lens,
                max_phoneme_len,
                accents,
                wavs,
                mels,
                mel_lens,
                max_mel_len,
                pitches,
            ) = batch
            with torch.no_grad():
                pitch_outputs, log_duration_outputs = variance_model(
                    phonemes=phonemes,
                    accents=accents,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                )
                feature_embedded, avg_pitches, durations, log_p_attn, bin_loss = embedder_model(
                    phonemes=phonemes,
                    pitches=pitches,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                    mels=mels,
                    mel_lens=mel_lens,
                )
                h_masks = make_non_pad_mask(mel_lens).to(feature_embedded.device)
                d_masks = make_non_pad_mask(phoneme_lens).to(durations.device)
                length_regulated_tensor = length_regulator(
                    hs=feature_embedded,
                    ds=durations,
                    h_masks=h_masks,
                    d_masks=d_masks,
                )
                outputs, postnet_outputs = decoder_model(
                    length_regulated_tensor=length_regulated_tensor,
                    mel_lens=mel_lens,
                )
                wav_outputs = generator_model(outputs.transpose(1, 2))
                mel_from_outputs = mel_spectrogram(
                    y=wav_outputs.squeeze(1),
                    n_fft=preprocess_config["stft"]["filter_length"],
                    num_mels=preprocess_config["mel"]["n_mel_channels"],
                    sampling_rate=preprocess_config["audio"]["sampling_rate"],
                    hop_size=preprocess_config["stft"]["hop_length"],
                    win_size=preprocess_config["stft"]["win_length"],
                    fmin=preprocess_config["mel"]["mel_fmin"],
                    fmax=preprocess_config["mel"]["mel_fmax"],
                    val=True
                ).transpose(1, 2)

                # Cal Loss
                variance_loss_all, duration_loss, pitch_loss, align_loss = variance_loss(
                    log_duration_outputs=log_duration_outputs,
                    pitch_outputs=pitch_outputs,
                    duration_targets=durations,
                    pitch_targets=avg_pitches,
                    log_p_attn=log_p_attn,
                    input_lens=phoneme_lens,
                    output_lens=mel_lens,
                )
                mel_loss = F.l1_loss(mels, mel_from_outputs)

                align_loss += bin_loss

                total_loss = variance_loss_all + bin_loss + (mel_loss * 45)

                if config["model"]["mode"] == "mel":
                    decoder_loss = F.l1_loss(mels, outputs)
                    postnet_loss = F.l1_loss(mels, postnet_outputs)
                    total_loss += decoder_loss + postnet_loss

                loss_dict["total_loss"] = total_loss
                loss_dict["mel_loss"] = mel_loss
                loss_dict["duration_loss"] = duration_loss
                loss_dict["pitch_loss"] = pitch_loss
                loss_dict["alignment_loss"] = align_loss

                for key in loss_dict.keys():
                    loss_value = loss_dict[key]
                    if loss_value is not None:
                        loss_dict[key] += loss_dict[key].item() * len(batch[0])

            if count < 5:
                fig, tag = plot_one_sample(
                    ids=ids,
                    duration_targets=durations,
                    pitch_targets=avg_pitches,
                    mel_targets=mels,
                    mel_predictions=mel_from_outputs,
                    phoneme_lens=phoneme_lens,
                    mel_lens=mel_lens
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
                    audio=wavs[0],
                    sampling_rate=sampling_rate,
                    tag="Validation/{}_gt".format(tag),
                )
                log(
                    logger,
                    audio=wav_outputs[0],
                    sampling_rate=sampling_rate,
                    tag="Validation/{}_synthesized".format(tag),
                    step=step,
                )
            count += 1

    for key in loss_dict.keys():
        loss_value = loss_dict[key]
        if loss_value is not None:
            loss_dict[key] = loss_dict[key] /len(dataset)

    log(logger, step, loss_dict=loss_dict)

    message1 = "Validation Step {}, ".format(step)
    message2 = (
            "Total Loss: {total_loss:.4f}, " +
            "Mel Loss: {mel_loss:.4f}, " +
            "Duration Loss: {duration_loss:.4f}, " +
            "Pitch Loss: {pitch_loss:.4f}, " +
            "Alignment Loss: {alignment_loss:.4f}"
    ).format(
        **loss_dict
    )

    return message1 + message2


def main(rank: int, restore_step: int, speaker_num, config: Config, num_gpus: int):
    if rank == 0:
        print("Prepare training ...")
    if num_gpus > 1:
        init_process_group(backend="nccl", init_method="env://",
                           world_size=num_gpus, rank=rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Get dataset
    dataset = Dataset(
        "train.txt", config["preprocess"], config["train"], sort=True, drop_last=True
    )
    batch_size = config["train"]["optimizer"]["batch_size"]
    if num_gpus > 1:
        group_size = 1
    else:
        group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)

    sampler = DistributedSampler(dataset) if num_gpus > 1 else None

    cpu_count = os.cpu_count()
    if cpu_count > 8:
        cpu_count = 8
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=num_gpus == 1,
        num_workers=cpu_count,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    variance_model, embedder_model, decoder_model, generator_model, mpd_model, msd_model, optimizer, epoch = \
        get_model(restore_step, config, device, speaker_num, train=True)
    length_regulator = GaussianUpsampling().to(device)
    preprocess_config = config["preprocess"]
    if num_gpus > 1:
        variance_model = DistributedDataParallel(variance_model, device_ids=[rank]).to(device)
        embedder_model = DistributedDataParallel(embedder_model, device_ids=[rank]).to(device)
        decoder_model = DistributedDataParallel(decoder_model, device_ids=[rank]).to(device)
        generator_model = DistributedDataParallel(generator_model, device_ids=[rank]).to(device)
        mpd_model = DistributedDataParallel(mpd_model, device_ids=[rank]).to(device)
        msd_model = DistributedDataParallel(msd_model, device_ids=[rank]).to(device)

    num_generate_param = get_param_num(variance_model) + get_param_num(embedder_model) + get_param_num(decoder_model) + get_param_num(generator_model)
    num_discriminate_param = get_param_num(mpd_model) + get_param_num(msd_model)
    variance_loss = VarianceLoss().to(device)
    generator_loss = GeneratorLoss().to(device)
    discriminator_loss = DiscriminatorLoss().to(device)

    if rank == 0:
        print("Number of JETS Generator Parameters:", num_generate_param)
        print("Number of JETS Discriminator Parameters:", num_discriminate_param)

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
    step = restore_step + 1
    epoch = max(1, epoch)
    total_step = config["train"]["step"]["total_step"]
    log_step = config["train"]["step"]["log_step"]
    save_step = config["train"]["step"]["save_step"]
    synth_step = config["train"]["step"]["synth_step"]
    val_step = config["train"]["step"]["val_step"]

    segment_size = config["model"]["vocoder"]["segment_size"]
    hop_length = config["preprocess"]["stft"]["hop_length"]

    if rank == 0:
        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = restore_step
        outer_bar.update()

    while True:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)

        if num_gpus > 1:
            sampler.set_epoch(epoch)

        for batchs in loader:
            for _batch in batchs:
                batch: ReProcessedItemTorch = to_device(_batch, device)
                (
                    ids,
                    speakers,
                    phonemes,
                    phoneme_lens,
                    max_phoneme_len,
                    accents,
                    wavs,
                    mels,
                    mel_lens,
                    max_mel_len,
                    pitches,
                ) = batch

                # Forward
                pitch_outputs, log_duration_outputs = variance_model(
                    phonemes=phonemes,
                    accents=accents,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                )
                feature_embedded, avg_pitches, durations, log_p_attn, bin_loss = embedder_model(
                    phonemes=phonemes,
                    pitches=pitches,
                    speakers=speakers,
                    phoneme_lens=phoneme_lens,
                    max_phoneme_len=max_phoneme_len,
                    mels=mels,
                    mel_lens=mel_lens,
                )
                h_masks = make_non_pad_mask(mel_lens).to(feature_embedded.device)
                d_masks = make_non_pad_mask(phoneme_lens).to(durations.device)
                length_regulated_tensor = length_regulator(
                    hs=feature_embedded,
                    ds=durations,
                    h_masks=h_masks,
                    d_masks=d_masks,
                )
                outputs, postnet_outputs = decoder_model(
                    length_regulated_tensor=length_regulated_tensor,
                    mel_lens=mel_lens,
                )

                if config["model"]["mode"] == "mel":
                    segmented_outputs, start_idxs = get_random_segments(postnet_outputs.transpose(1, 2), mel_lens, segment_size // hop_length)
                else:
                    segmented_outputs, start_idxs = get_random_segments(outputs.transpose(1, 2), mel_lens, segment_size // hop_length)
                segumented_wavs = get_segments(wavs.unsqueeze(1), start_idxs * hop_length, segment_size)

                wav_outputs = generator_model(segmented_outputs)
                mel_from_outputs = mel_spectrogram(
                    y=wav_outputs.squeeze(1),
                    n_fft=preprocess_config["stft"]["filter_length"],
                    num_mels=preprocess_config["mel"]["n_mel_channels"],
                    sampling_rate=preprocess_config["audio"]["sampling_rate"],
                    hop_size=preprocess_config["stft"]["hop_length"],
                    win_size=preprocess_config["stft"]["win_length"],
                    fmin=preprocess_config["mel"]["mel_fmin"],
                    fmax=preprocess_config["mel"]["mel_fmax"],
                    val=True
                )
                segumented_mels = get_segments(mels.transpose(1, 2), start_idxs, segment_size // hop_length)

                y_df_hat_r, y_df_hat_g, _, _ = mpd_model(segumented_wavs, wav_outputs.detach())
                y_ds_hat_r, y_ds_hat_g, _, _ = msd_model(segumented_wavs, wav_outputs.detach())

                # Discriminator Loss
                loss_disc_all, loss_disc_s, loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g, y_ds_hat_r, y_ds_hat_g)
                loss_disc_all = loss_disc_all * 2.0
                loss_disc_all.backward()

                # Variance & Generator Loss
                variance_loss_all, duration_loss, pitch_loss, align_loss = variance_loss(
                    log_duration_outputs=log_duration_outputs,
                    pitch_outputs=pitch_outputs,
                    duration_targets=durations,
                    pitch_targets=avg_pitches,
                    log_p_attn=log_p_attn,
                    input_lens=phoneme_lens,
                    output_lens=mel_lens,
                )
                _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd_model(segumented_wavs, wav_outputs)
                _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd_model(segumented_wavs, wav_outputs)
                loss_gen_all, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_mel = generator_loss(
                    mels=segumented_mels,
                    mel_from_outputs=mel_from_outputs,
                    y_df_hat_g=y_df_hat_g,
                    fmap_f_r=fmap_f_r,
                    fmap_f_g=fmap_f_g,
                    y_ds_hat_g=y_ds_hat_g,
                    fmap_s_r=fmap_s_r,
                    fmap_s_g=fmap_s_g,
                )

                align_loss += bin_loss

                total_loss = variance_loss_all + (bin_loss * 2.0) + loss_gen_all

                if config["model"]["mode"] == "mel":
                    mel_loss = F.l1_loss(mels, outputs)
                    postnet_loss = F.l1_loss(mels, postnet_outputs)
                    total_loss += mel_loss + postnet_loss

                # Backward
                total_loss.backward()

                # Update weights
                optimizer.zero_grad()
                optimizer.step_and_update_lr()

                if rank == 0:
                    if step % log_step == 0:
                        loss_dict: LossDict = {
                            "total_loss": total_loss + loss_disc_all,
                            "mel_loss": loss_mel,
                            "duration_loss": duration_loss,
                            "pitch_loss": pitch_loss,
                            "alignment_loss": align_loss,
                            "generator_loss": loss_gen_all,
                            "discriminator_loss": loss_disc_all
                        }
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = (
                           "Total Loss: {total_loss:.4f}, " +
                           "Mel Loss: {mel_loss:.4f}, " +
                           "Duration Loss: {duration_loss:.4f}, " +
                           "Pitch Loss: {pitch_loss:.4f}, " +
                           "Alignment Loss: {alignment_loss:.4f}, " +
                           "Generator Loss: {generator_loss:.4f}, " +
                           "Discriminator Loss: {discriminator_loss:.4f}"
                        ).format(
                            **loss_dict
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, loss_dict=loss_dict)

                    if step % synth_step == 0:
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            wav_outputs = generator_model(outputs[0].unsqueeze(0).transpose(1, 2))
                            mel_from_wavs = mel_spectrogram(
                                y=wav_outputs.squeeze(1),
                                n_fft=preprocess_config["stft"]["filter_length"],
                                num_mels=preprocess_config["mel"]["n_mel_channels"],
                                sampling_rate=preprocess_config["audio"]["sampling_rate"],
                                hop_size=preprocess_config["stft"]["hop_length"],
                                win_size=preprocess_config["stft"]["win_length"],
                                fmin=preprocess_config["mel"]["mel_fmin"],
                                fmax=preprocess_config["mel"]["mel_fmax"]
                            ).transpose(1, 2)
                        fig, tag = plot_one_sample(
                            ids=ids,
                            duration_targets=durations,
                            pitch_targets=avg_pitches,
                            mel_targets=mels,
                            mel_predictions=mel_from_wavs,
                            phoneme_lens=phoneme_lens,
                            mel_lens=mel_lens
                        )
                        log(
                            train_logger,
                            fig=fig,
                            tag="Training/step_{}_{}".format(step, tag),
                        )
                        sampling_rate = config["preprocess"]["audio"]["sampling_rate"]
                        log(
                            train_logger,
                            audio=wavs[0],
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_reconstructed".format(step, tag),
                        )
                        log(
                            train_logger,
                            audio=wav_outputs[0],
                            sampling_rate=sampling_rate,
                            tag="Training/step_{}_{}_synthesized".format(step, tag),
                        )

                    if step % val_step == 0:
                        variance_model.eval()
                        embedder_model.eval()
                        decoder_model.eval()
                        generator_model.eval()
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            message = evaluate(
                                variance_model, embedder_model, decoder_model, generator_model, length_regulator, step, config, val_logger
                            )
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        variance_model.train()
                        embedder_model.train()
                        decoder_model.train()
                        generator_model.train()

                    if step % save_step == 0:
                        torch.save(
                            {
                                "variance_model": (variance_model.module if num_gpus > 1 else variance_model).state_dict(),
                                "embedder_model": (embedder_model.module if num_gpus > 1 else embedder_model).state_dict(),
                                "decoder_model": (decoder_model.module if num_gpus > 1 else decoder_model).state_dict(),
                                "generator_model": (generator_model.module if num_gpus > 1 else generator_model).state_dict(),
                                "mpd_model": (mpd_model.module if num_gpus > 1 else mpd_model).state_dict(),
                                "msd_model": (msd_model.module if num_gpus > 1 else msd_model).state_dict(),
                                "optimizer": {
                                    "variance": optimizer._variance_optimizer.state_dict(),
                                    "embedder": optimizer._embedder_optimizer.state_dict(),
                                    "decoder": optimizer._decoder_optimizer.state_dict(),
                                    "generator": optimizer._generator_optimizer.state_dict(),
                                    "discriminator": optimizer._discriminator_optimizer.state_dict(),
                                },
                                "epoch": epoch - 1,
                            },
                            os.path.join(
                                config["train"]["path"]["ckpt_path"],
                                "{:08d}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    quit()
                step += 1

                if rank == 0:
                    outer_bar.update()

            if rank == 0:
                inner_bar.update()
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config yaml")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--speaker_num", type=int, default=10)
    args = parser.parse_args()

    config: Config = yaml.load(
        open(args.config, "r"), Loader=yaml.FullLoader
    )

    num_gpus = 0
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        batch_size = config["train"]["optimizer"]["batch_size"]
        config["train"]["optimizer"]["batch_size"] = int(batch_size / num_gpus)
        print('Batch size per GPU :', config["train"]["optimizer"]["batch_size"])
    else:
        raise Exception("cuda is not available")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54321'

    if num_gpus > 1:
        mp.spawn(main, nprocs=num_gpus, args=(args.restore_step, args.speaker_num, config, num_gpus,))
    else:
        main(0, args.restore_step, args.speaker_num, config, num_gpus)
