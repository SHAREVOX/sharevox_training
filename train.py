import logging
import os
import json
import argparse
import platform
from typing import List, Optional, Tuple

import yaml

from config import Config
from dataset import Dataset, DistributedBucketSampler
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from models.tts import VITS, JETS
from models.upsampler import (
    # SiFiGANMultiPeriodAndScaleDiscriminator,
    SiFiGANMultiPeriodAndResolutionDiscriminator,
    # SFreGAN2MultiPeriodAndScaleDiscriminator,
    SFreGAN2MultiPeriodAndResolutionDiscriminator,
)
from models.loss import ForwardSumLoss, ResidualLoss, generator_loss, discriminator_loss, feature_loss, kl_loss
from utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from utils.checkpoint import load_checkpoint, latest_checkpoint_path, save_checkpoint
from utils.logging import get_logger, summarize
from utils.plot import plot_spectrogram_to_numpy, plot_alignment_to_numpy, plot_f0_to_numpy
from utils.slice import slice_segments
from utils.tools import clip_grad_value_, to_device

# torch.backends.cudnn.benchmark = True
global_step = 0
outer_bar: Optional[tqdm] = None


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65432"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="YAML file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument("-s", "--speakers", type=int, required=False, default=10, help="speaker count")
    parser.add.argument("-g", "--g_checkpoint_path", type=str, required=False, default=None, help="generator checkpoint path")
    parser.add.argument("-d", "--d_checkpoint_path", type=str, required=False, default=None, help="discriminator checkpoint path")
    parser.add_argument("-l", "--log_dir", type=str, required=False, default="./logs", help="log dir path")

    args = parser.parse_args()
    model_dir = os.path.join(args.log_dir, args.model)

    with open(args.config, "r") as f:
        config: Config = yaml.load(f, Loader=yaml.FullLoader)

    if n_gpus > 1:
        mp.spawn(
            run,
            nprocs=n_gpus,
            args=(
                n_gpus,
                config,
                model_dir,
                args.speakers,
                args.g_checkpoint_path,
                args.d_checkpoint_path
            ),
        )
    else:
        run(
            0,
            n_gpus,
            config,
            model_dir,
            args.speakers,
            args.g_checkpoint_path,
            args.d_checkpoint_path
        )


def run(
    rank: int,
    n_gpus: int,
    config: Config,
    model_dir: str,
    speakers: int,
    g_checkpoint_path: Optional[str] = None,
    d_checkpoint_path: Optional[str] = None
):
    global global_step
    global outer_bar
    if rank == 0:
        logger = get_logger(model_dir)
        logger.info(config)
        writer = SummaryWriter(log_dir=model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))

    if n_gpus > 1:
        backend = "gloo" if platform.system() == "Windows" else "nccl"
        dist.init_process_group(
            backend, init_method="env://", world_size=n_gpus, rank=rank
        )
    torch.manual_seed(config["train"]["seed"])
    torch.cuda.set_device(rank)

    train_dataset = Dataset("train.txt", config["preprocess"], config["train"])
    train_sampler = DistributedBucketSampler(
        train_dataset,
        config["train"]["optimizer"]["batch_size"],
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # train_sampler = DistributedSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = Dataset("val.txt", config["preprocess"], config["train"])
        eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=config["train"]["optimizer"]["batch_size"],
            pin_memory=True,
            drop_last=False,
            collate_fn=eval_dataset.collate_fn,
        )

    device = torch.device('cuda:{:d}'.format(rank))

    preprocessed_path = config["preprocess"]["path"]["preprocessed_path"]
    with open(os.path.join(preprocessed_path, "stats.json")) as f:
        stats_text = f.read()
    stats_json = json.loads(stats_text)
    pitch_mean, pitch_std = stats_json["pitch"][2:]

    model_config = config["model"]
    model_type = model_config["model_type"]
    
    if model_type == "vits":
        Model = VITS
    elif model_type == "jets":
        Model = JETS
    else:
        raise Exception(f"Unknown model type: {model_type}")

    net_g = Model(
        model_config,
        spec_channels=config["preprocess"]["stft"]["filter_length"] // 2 + 1,
        pitch_mean=pitch_mean,
        pitch_std=pitch_std,
        sampling_rate=config["preprocess"]["audio"]["sampling_rate"],
        hop_length=config["preprocess"]["stft"]["hop_length"],
        n_speakers=speakers,
    ).to(device)

    if model_config["upsampler_type"] == "sifigan":
        net_d = SiFiGANMultiPeriodAndResolutionDiscriminator().to(device)
    else:
        net_d = SFreGAN2MultiPeriodAndResolutionDiscriminator().to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        config["train"]["optimizer"]["learning_rate"],
        betas=config["train"]["optimizer"]["betas"],
        eps=config["train"]["optimizer"]["eps"],
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config["train"]["optimizer"]["learning_rate"],
        betas=config["train"]["optimizer"]["betas"],
        eps=config["train"]["optimizer"]["eps"],
    )

    if n_gpus > 1:
        net_g = DDP(net_g, device_ids=[rank]).to(device)
        net_d = DDP(net_d, device_ids=[rank]).to(device)
    else:
        net_g = DP(net_g, device_ids=[rank]).to(device)
        net_d = DP(net_d, device_ids=[rank]).to(device)

    try:
        _, _, _, epoch_str, _ = load_checkpoint(
            latest_checkpoint_path(model_dir, "G_*.pth"), net_g, optim_g
        )
        _, _, _, epoch_str, step = load_checkpoint(
            latest_checkpoint_path(model_dir, "D_*.pth"), net_d, optim_d
        )
        global_step = step
    except:
        epoch_str = 1
        global_step = 0
        if g_checkpoint_path is not None:
            _, _, _, epoch_str, _ = load_checkpoint(g_checkpoint_path, net_g, optim_g)
        if d_checkpoint_path is not None:
            _, _, _, epoch_str, _ = load_checkpoint(d_checkpoint_path, net_d, optim_d)

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config["train"]["optimizer"]["lr_decay"], last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config["train"]["optimizer"]["lr_decay"], last_epoch=epoch_str - 2
    )

    forward_sum_loss = ForwardSumLoss().to(device)
    residual_loss = ResidualLoss(
        sample_rate=config["preprocess"]["audio"]["sampling_rate"],
        fft_size=config["preprocess"]["stft"]["filter_length"],
        hop_size=config["preprocess"]["stft"]["hop_length"],
    ).to(device)

    scaler = GradScaler(enabled=config["train"]["fp16_run"])

    total_epoch = config["train"]["step"]["total_epoch"]
    train_loader_len = len(train_loader)
    outer_bar = tqdm(total=total_epoch * train_loader_len, desc="Training", position=0, dynamic_ncols=True)
    outer_bar.n = global_step

    for epoch in range(epoch_str, total_epoch + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                config,
                model_dir,
                (net_g, net_d),
                (optim_g, optim_d),
                (scheduler_g, scheduler_d),
                scaler,
                (train_loader, eval_loader),
                logger,
                (writer, writer_eval),
                (forward_sum_loss, residual_loss),
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                config,
                model_dir,
                (net_g, net_d),
                (optim_g, optim_d),
                (scheduler_g, scheduler_d),
                scaler,
                (train_loader, None),
                None,
                None,
                (forward_sum_loss, residual_loss),
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank: int,
    epoch: int,
    config: Config,
    model_dir: str,
    nets: Tuple[nn.Module, ...],
    optims: Tuple[optim.Optimizer, ...],
    schedulers: Tuple[optim.lr_scheduler.ExponentialLR, ...],
    scaler: GradScaler,
    loaders: Tuple[DataLoader, Optional[DataLoader]],
    logger: Optional[logging.Logger],
    writers: Optional[Tuple[SummaryWriter]],
    loss_funcs: List[nn.Module]
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    forward_sum_loss, residual_loss = loss_funcs
    if writers is not None:
        writer: SummaryWriter
        writer_eval: SummaryWriter
        writer, writer_eval = writers
    device = torch.device('cuda:{:d}'.format(rank))

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    global outer_bar
    if rank == 0:
        inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)

    net_g.train()
    net_d.train()
    pitch_std = net_g.module.pitch_std
    pitch_mean = net_g.module.pitch_mean

    for batch_idx, batch in enumerate(train_loader):
        batch = to_device(batch, device)
        (
            _,
            speakers,
            phonemes,
            phoneme_lens,
            _,
            moras,
            accents,
            wavs,
            specs,
            spec_lens,
            _,
            pitches,
        ) = batch
        wavs = wavs.unsqueeze(1)

        with autocast(enabled=config["train"]["fp16_run"]):
            (
                y_hat,
                excs,
                pitch_slices,
                durations,
                pred_durations,
                avg_pitches,
                pred_pitches,
                pred_frame_pitches,
                unvoice_mask,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                loss_bin,
                loss_vae,
            ) = net_g(phonemes, phoneme_lens, moras, accents, pitches, specs, spec_lens, speakers)

            mel = spec_to_mel_torch(
                specs.transpose(1, 2),
                config["preprocess"]["stft"]["filter_length"],
                config["preprocess"]["mel"]["n_mel_channels_loss"],
                config["preprocess"]["audio"]["sampling_rate"],
                config["preprocess"]["mel"]["mel_fmin"],
                config["preprocess"]["mel"]["mel_fmax_loss"],
            )
            y_mel = slice_segments(
                mel, ids_slice,
                config["model"]["upsampler"]["segment_size"] // config["preprocess"]["stft"]["hop_length"]
            )

        if config["train"]["fp16_run"]:
            y_hat = y_hat.float()
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1),
            config["preprocess"]["stft"]["filter_length"],
            config["preprocess"]["mel"]["n_mel_channels_loss"],
            config["preprocess"]["audio"]["sampling_rate"],
            config["preprocess"]["stft"]["hop_length"],
            config["preprocess"]["stft"]["win_length"],
            config["preprocess"]["mel"]["mel_fmin"],
            config["preprocess"]["mel"]["mel_fmax_loss"],
        )

        y = slice_segments(
            wavs, ids_slice * config["preprocess"]["stft"]["hop_length"],
            config["model"]["upsampler"]["segment_size"]
        )  # slice

        with autocast(enabled=config["train"]["fp16_run"]):
            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc
        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()

        scaler.unscale_(optim_d)
        grad_norm_d = clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=config["train"]["fp16_run"]):
            # Generator
            # y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            _, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                loss_duration = F.mse_loss(
                    torch.log(durations.to(dtype=pred_durations.dtype).unsqueeze(1).masked_select(x_mask) + 1.0),
                    pred_durations.masked_select(x_mask)
                )
                loss_pitch = F.mse_loss(
                    avg_pitches.to(dtype=pred_pitches.dtype).masked_select(x_mask),
                    pred_pitches.masked_select(x_mask)
                )
                loss_frame_pitch = F.mse_loss(
                    pitches.unsqueeze(1).to(dtype=pred_pitches.dtype).masked_select(z_mask.bool()),
                    pred_frame_pitches.masked_select(z_mask.bool())
                )
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * config["train"]["loss_balance"]["mel"]
                fs_loss = forward_sum_loss(attn, phoneme_lens, spec_lens) * config["train"]["loss_balance"]["align"]
                reg_loss = residual_loss(excs, y, pitch_slices)

                loss_fm = feature_loss(fmap_r, fmap_g) * config["train"]["loss_balance"]["fm"]
                loss_bin = loss_bin * config["train"]["loss_balance"]["align"]
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen = loss_gen * config["train"]["loss_balance"]["gen"]
                loss_gen_all = loss_gen + loss_mel + loss_bin + fs_loss + reg_loss + loss_fm
                if config["train"]["step"]["feat_learn_start"] <= global_step:
                    loss_gen_all = loss_gen_all + loss_duration + loss_pitch + loss_frame_pitch
                if loss_vae is not None:
                    _, z_p, m_p, logs_p, _, logs_q = loss_vae
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config["train"]["loss_balance"]["kl"]
                    loss_gen_all = loss_gen_all + loss_kl

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()

        scaler.unscale_(optim_g)
        grad_norm_g = clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % config["train"]["step"]["log_step"] == 0:
                lr = optim_g.param_groups[0]["lr"]
                losses = {
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/duration": loss_duration,
                    "loss/g/pitch": loss_pitch,
                    "loss/g/frame_pitch": loss_frame_pitch,
                    "loss/g/bin": loss_bin,
                    "loss/g/fs": fs_loss,
                    "loss/g/reg": reg_loss,
                }
                if loss_vae is not None:
                    losses["loss/g/kl"] = loss_kl
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info(f"Step: {global_step} LR: {lr}")
                logger.info(", ".join([f"{k.split('/')[-1]}: {v.item()}" for k, v in losses.items()]))

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    losses
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": plot_spectrogram_to_numpy(
                        y_mel[0].detach().cpu().numpy()
                    ),
                    "slice/mel_gen": plot_spectrogram_to_numpy(
                        y_hat_mel[0].detach().cpu().numpy()
                    ),
                    "all/mel": plot_spectrogram_to_numpy(
                        mel[0, :, :spec_lens[0]].detach().cpu().numpy()
                    ),
                    "all/attn": plot_alignment_to_numpy(
                        attn[0, :spec_lens[0], :phoneme_lens[0]].detach().cpu().numpy()
                    ),
                }
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )

            if global_step % config["train"]["step"]["val_step"] == 0:
                net_g.eval()
                with torch.no_grad():
                    hop_length = config["preprocess"]["stft"]["hop_length"]
                    x_lengths = phoneme_lens[:1]
                    x = phonemes[:1, :x_lengths[0]]
                    spec_lengths = spec_lens[:1]
                    spec = specs[:1, :spec_lengths[0]]
                    accent = accents[:1, :x_lengths[0]]
                    mora = moras[:1,:,:x_lengths]
                    mora = mora[:,:(mora == 1).nonzero(as_tuple=True)[1][-1]+1,:]
                    pitch = pitches[:1, :spec_lens[0]]
                    y = wavs[:1]
                    speaker = speakers[:1]
                    (
                        y_reconst, *_
                    ) = net_g(x, x_lengths, mora, accent, pitch, spec, spec_lengths, speaker, slice=False)
                    y_hat, excs, attn, regulated_pitches, pred_regulated_pitches, frame_pitches, mask = \
                        net_g.module.infer(x, x_lengths, mora, accent, pitch, spec, spec_lengths, sid=speaker)
                    y_length = spec_lengths * hop_length
                    y_hat_lengths = mask.sum([1, 2]).long() * hop_length

                    mel = spec_to_mel_torch(
                        spec.transpose(1, 2),
                        config["preprocess"]["stft"]["filter_length"],
                        config["preprocess"]["mel"]["n_mel_channels_loss"],
                        config["preprocess"]["audio"]["sampling_rate"],
                        config["preprocess"]["mel"]["mel_fmin"],
                        config["preprocess"]["mel"]["mel_fmax_loss"],
                    )
                    y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1).float(),
                        config["preprocess"]["stft"]["filter_length"],
                        config["preprocess"]["mel"]["n_mel_channels_loss"],
                        config["preprocess"]["audio"]["sampling_rate"],
                        config["preprocess"]["stft"]["hop_length"],
                        config["preprocess"]["stft"]["win_length"],
                        config["preprocess"]["mel"]["mel_fmin"],
                        config["preprocess"]["mel"]["mel_fmax_loss"],
                    )
                    image_dict = {
                        "train/mel": plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().numpy())
                    }
                    audio_dict = {
                        "train/audio": y_hat[0, :, : y_hat_lengths[0]],
                        "train/exc": excs[0, :, : y_hat_lengths[0]]
                    }
                    image_dict.update(
                        {
                            "train/gt_mel": plot_spectrogram_to_numpy(mel[0].detach().cpu().numpy()),
                            "train/compare_f0": plot_f0_to_numpy(
                                pitches[0].detach().cpu().numpy() * pitch_std + pitch_mean,
                                regulated_pitches[0, 0].detach().cpu().numpy() * pitch_std + pitch_mean,
                                pred_regulated_pitches[0, 0].detach().cpu().numpy() * pitch_std + pitch_mean,
                                frame_pitches[0, 0].detach().cpu().numpy()
                            )
                        }
                    )
                    audio_dict.update({
                        "train/gt_audio": y[0, :, : y_length[0]],
                        "train/reconst": y_reconst[0, :, : y_length[0]],
                    })

                    summarize(
                        writer=writer,
                        global_step=global_step,
                        images=image_dict,
                        audios=audio_dict,
                        audio_sampling_rate=config["preprocess"]["audio"]["sampling_rate"],
                    )
                # net_g.train()
                evaluate(config, device, net_g, eval_loader, writer_eval)
            if global_step % config["train"]["step"]["save_step"] == 0:
                save_checkpoint(
                    net_g,
                    optim_g,
                    config["train"]["optimizer"]["learning_rate"],
                    epoch,
                    os.path.join(model_dir, "G_{}.pth".format(global_step)),
                )
                save_checkpoint(
                    net_d,
                    optim_d,
                    config["train"]["optimizer"]["learning_rate"],
                    epoch,
                    os.path.join(model_dir, "D_{}.pth".format(global_step)),
                )
            global_step += 1
            outer_bar.update()
            inner_bar.update()

    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(
    config: Config, device: torch.device, generator: nn.Module, eval_loader: DataLoader, writer_eval: SummaryWriter
):
    generator.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            batch = to_device(batch, device)
            (
                ids,
                speakers,
                phonemes,
                phoneme_lens,
                max_phoneme_len,
                moras,
                accents,
                wavs,
                specs,
                spec_lens,
                max_spec_len,
                pitches,
            ) = batch
            wavs = wavs.unsqueeze(1)
            # x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            # spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
            # y, y_lengths = y.cuda(0), y_lengths.cuda(0)

            # remove else
            hop_length = config["preprocess"]["stft"]["hop_length"]
            x_lengths = phoneme_lens[:1]
            x = phonemes[:1,:x_lengths]
            spec_lengths = spec_lens[:1]
            spec = specs[:1,:spec_lengths]
            pitch = pitches[:1,:spec_lengths]
            accent = accents[:1,:x_lengths]
            mora = moras[:1,:,:x_lengths]
            mora = mora[:,:(mora == 1).nonzero(as_tuple=True)[1][-1]+1,:]
            y = wavs[:1]
            y_lengths = spec_lens[:1] * hop_length
            speaker = speakers[:1]
            break
        y_hat, excs, attn, regulated_pitches, pred_regulated_pitches, frame_pitches, mask = \
            generator.module.infer(x, x_lengths, mora, accent, pitch, spec, spec_lengths, sid=speaker)

        y_hat_lengths = mask.sum([1, 2]).long() * hop_length

        mel = spec_to_mel_torch(
            spec.transpose(1, 2),
            config["preprocess"]["stft"]["filter_length"],
            config["preprocess"]["mel"]["n_mel_channels_loss"],
            config["preprocess"]["audio"]["sampling_rate"],
            config["preprocess"]["mel"]["mel_fmin"],
            config["preprocess"]["mel"]["mel_fmax_loss"],
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            config["preprocess"]["stft"]["filter_length"],
            config["preprocess"]["mel"]["n_mel_channels_loss"],
            config["preprocess"]["audio"]["sampling_rate"],
            config["preprocess"]["stft"]["hop_length"],
            config["preprocess"]["stft"]["win_length"],
            config["preprocess"]["mel"]["mel_fmin"],
            config["preprocess"]["mel"]["mel_fmax_loss"],
        )
    pitch_std = generator.module.pitch_std
    pitch_mean = generator.module.pitch_mean
    image_dict = {
        "gen/mel": plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
        "gen/compare_f0": plot_f0_to_numpy(
            pitches[0].detach().cpu().numpy() * pitch_std + pitch_mean,
            regulated_pitches[0, 0].detach().cpu().numpy() * pitch_std + pitch_mean,
            pred_regulated_pitches[0, 0].detach().cpu().numpy() * pitch_std + pitch_mean,
            frame_pitches[0, 0].detach().cpu().numpy()
        )
    }
    audio_dict = {
        "gen/audio": y_hat[0, :, : y_hat_lengths[0]],
        "gen/exc": excs[0, :, : y_hat_lengths[0]]
    }
    if global_step == 0:
        image_dict.update(
            {"gt/mel": plot_spectrogram_to_numpy(mel[0].cpu().numpy())}
        )
        audio_dict.update({"gt/audio": y[0, :, : y_lengths[0]]})

    summarize(
        writer=writer_eval,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=config["preprocess"]["audio"]["sampling_rate"],
    )
    generator.train()


if __name__ == "__main__":
    main()
