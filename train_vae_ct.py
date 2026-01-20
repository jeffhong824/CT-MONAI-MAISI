#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE Training Script for CT Dataset
Supports multi-GPU training with DDP
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, DistributedSampler
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from scripts.transforms import VAE_Transform
from scripts.utils import KL_loss, define_instance, dynamic_infer
from scripts.utils_plot import find_label_center_loc, get_xyz_plot

warnings.filterwarnings("ignore")


def setup_ddp(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    return torch.device(f"cuda:{rank}")


def main():
    parser = argparse.ArgumentParser(description="Train VAE for CT dataset")
    parser.add_argument(
        "-e",
        "--environment-file",
        default="./configs/environment_maisi_vae_train_ct.json",
        help="Environment configuration file",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        default="./configs/config_network_rflow.json",
        help="Network configuration file",
    )
    parser.add_argument(
        "-t",
        "--training-config",
        default="./configs/config_maisi_vae_train_ct.json",
        help="Training configuration file",
    )
    parser.add_argument("-g", "--gpus", default=1, type=int, help="Number of GPUs")
    
    args = parser.parse_args()

    # Initialize distributed training
    use_ddp = args.gpus > 1
    if use_ddp:
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = setup_ddp(rank, world_size)
            is_distributed = True
        else:
            raise RuntimeError("DDP requires RANK and WORLD_SIZE environment variables")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False

    if rank == 0:
        print("MONAI Configuration:")
        print_config()
        print(f"\nTraining with {world_size} GPU(s)")

    # Load configurations
    with open(args.environment_file, "r") as f:
        env_dict = json.load(f)
    with open(args.config_file, "r") as f:
        config_dict = json.load(f)
    with open(args.training_config, "r") as f:
        train_config_dict = json.load(f)

    # Merge configurations into args
    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in config_dict.items():
        setattr(args, k, v)
    for k, v in train_config_dict["data_option"].items():
        setattr(args, k, v)
    for k, v in train_config_dict["autoencoder_train"].items():
        setattr(args, k, v)

    if rank == 0:
        print("\n--- Configuration ---")
        print(f"Model dir: {args.model_dir}")
        print(f"TensorBoard path: {args.tfevent_path}")
        print(f"JSON data list: {args.json_data_list}")
        print(f"Batch size: {args.batch_size}")
        print(f"Patch size: {args.patch_size}")
        print(f"Epochs: {args.n_epochs}")

    # Load data from JSON
    with open(args.json_data_list, "r") as f:
        datalist = json.load(f)
    
    train_files = datalist["training"]
    val_files = datalist["validation"]
    
    if rank == 0:
        print(f"\nFound {len(train_files)} training files")
        print(f"Found {len(val_files)} validation files")

    # Set deterministic training
    set_determinism(seed=0)

    # Setup directories
    if rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
        trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
        print(f"Trained model will be saved as {trained_g_path}")

        Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
        tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
        Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(tensorboard_path)
        print(f"Tensorboard events will be saved in {tensorboard_path}")
    else:
        trained_g_path = os.path.join(args.model_dir, "autoencoder.pt")
        trained_d_path = os.path.join(args.model_dir, "discriminator.pt")
        tensorboard_writer = None

    # Define transforms
    train_transform = VAE_Transform(
        is_train=True,
        random_aug=args.random_aug,
        k=4,
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    val_transform = VAE_Transform(
        is_train=False,
        random_aug=False,
        k=4,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float16,
        image_keys=["image"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )

    # Build datasets
    dataset_train = CacheDataset(data=train_files, transform=train_transform, cache_rate=args.cache, num_workers=8)
    dataset_val = CacheDataset(data=val_files, transform=val_transform, cache_rate=args.cache, num_workers=8)

    # Create distributed samplers
    train_sampler = DistributedSampler(dataset=dataset_train, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(dataset=dataset_val, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=(train_sampler is None),
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.val_batch_size,
        num_workers=4,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True
    )

    # Initialize models
    args.autoencoder_def["num_splits"] = 1
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    ).to(device)

    # Load pretrained weights if finetuning
    if args.finetune and args.trained_autoencoder_path and os.path.exists(args.trained_autoencoder_path):
        if rank == 0:
            print(f"\n--- Loading pretrained autoencoder from: {args.trained_autoencoder_path} ---")
        checkpoint = torch.load(args.trained_autoencoder_path, map_location=device)
        if "unet_state_dict" in checkpoint.keys():
            checkpoint = checkpoint["unet_state_dict"]
        autoencoder.load_state_dict(checkpoint)
        if rank == 0:
            print("Pretrained weights loaded successfully")
    else:
        if rank == 0:
            print("\n--- Training from scratch ---")

    # Wrap with DDP
    if is_distributed:
        autoencoder = DDP(autoencoder, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        model_for_training = autoencoder.module
        discriminator_for_training = discriminator.module
    else:
        model_for_training = autoencoder
        discriminator_for_training = discriminator

    # Setup losses
    if args.recon_loss == "l2":
        intensity_loss = MSELoss()
    else:
        intensity_loss = L1Loss(reduction="mean")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(device)

    # Setup optimizers
    optimizer_g = torch.optim.Adam(params=model_for_training.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)
    optimizer_d = torch.optim.Adam(params=discriminator_for_training.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08)

    def warmup_rule(epoch):
        if epoch < 10:
            return 0.01
        elif epoch < 20:
            return 0.1
        return 1.0

    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5) if args.amp else None
    scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5) if args.amp else None

    # Training loop
    val_interval = args.val_interval
    best_val_recon_epoch_loss = float('inf')
    total_step = 0
    start_epoch = 0
    max_epochs = args.n_epochs

    val_inferer = (
        SlidingWindowInferer(
            roi_size=args.val_sliding_window_patch_size,
            sw_batch_size=1,
            progress=False,
            overlap=0.0,
            device=torch.device("cpu"),
            sw_device=device,
        )
        if args.val_sliding_window_patch_size
        else SimpleInferer()
    )

    def loss_weighted_sum(losses):
        return losses["recons_loss"] + args.kl_weight * losses["kl_loss"] + args.perceptual_weight * losses["p_loss"]

    for epoch in range(start_epoch, max_epochs):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            print("-" * 20)
            print(f"Epoch {epoch + 1}/{max_epochs}, LR: {scheduler_g.get_last_lr()[0]:.6f}")

        model_for_training.train()
        discriminator_for_training.train()
        train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

        for batch in dataloader_train:
            images = batch["image"].to(device).contiguous()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=args.amp):
                # Train Generator
                reconstruction, z_mu, z_sigma = autoencoder(images)
                losses = {
                    "recons_loss": intensity_loss(reconstruction, images),
                    "kl_loss": KL_loss(z_mu, z_sigma),
                    "p_loss": loss_perceptual(reconstruction.float(), images.float()),
                }
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

                if args.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                # Train Discriminator
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                loss_d = (loss_d_fake + loss_d_real) * 0.5

                if args.amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

            total_step += 1
            for loss_name, loss_value in losses.items():
                train_epoch_losses[loss_name] += loss_value.item()
                if rank == 0 and tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(f"train_iter/{loss_name}", loss_value.item(), total_step)

        scheduler_g.step()
        scheduler_d.step()

        if len(dataloader_train) > 0:
            for key in train_epoch_losses:
                train_epoch_losses[key] /= len(dataloader_train)
        else:
            if rank == 0:
                print("WARNING: dataloader_train is empty")
            continue

        # Aggregate losses across GPUs
        if is_distributed:
            for key in train_epoch_losses:
                loss_tensor = torch.tensor(train_epoch_losses[key], device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                train_epoch_losses[key] = loss_tensor.item() / world_size

        if rank == 0:
            print(f"Epoch {epoch + 1} Train VAE Loss: {loss_weighted_sum(train_epoch_losses):.4f}, Details: {train_epoch_losses}")
            for loss_name, loss_value in train_epoch_losses.items():
                tensorboard_writer.add_scalar(f"train_epoch/{loss_name}", loss_value, epoch + 1)

        # Save model only on rank 0
        if rank == 0:
            torch.save(model_for_training.state_dict(), trained_g_path)
            torch.save(discriminator_for_training.state_dict(), trained_d_path)

        # Validation
        if (epoch + 1) % val_interval == 0:
            model_for_training.eval()
            val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

            with torch.no_grad():
                for batch in dataloader_val:
                    images = batch["image"]
                    with autocast(device_type=device.type, enabled=args.amp):
                        reconstruction, z_mu, z_sigma = dynamic_infer(val_inferer, model_for_training, images)
                        reconstruction = reconstruction.to(device)
                        images_dev = images.to(device)
                        val_epoch_losses["recons_loss"] += intensity_loss(reconstruction, images_dev).item()
                        val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                        val_epoch_losses["p_loss"] += loss_perceptual(reconstruction.float(), images_dev.float()).item()

            for key in val_epoch_losses:
                val_epoch_losses[key] /= len(dataloader_val)

            # Aggregate validation losses
            if is_distributed:
                for key in val_epoch_losses:
                    loss_tensor = torch.tensor(val_epoch_losses[key], device=device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    val_epoch_losses[key] = loss_tensor.item() / world_size

            val_loss_g = loss_weighted_sum(val_epoch_losses)

            if rank == 0:
                print(f"Epoch {epoch + 1} Validation VAE Loss: {val_loss_g:.4f}, Details: {val_epoch_losses}")
                for loss_name, loss_value in val_epoch_losses.items():
                    tensorboard_writer.add_scalar(f"val_epoch/{loss_name}", loss_value, epoch + 1)

                if val_loss_g < best_val_recon_epoch_loss:
                    best_val_recon_epoch_loss = val_loss_g
                    best_model_path = f"{trained_g_path[:-3]}_best_epoch{epoch+1}.pt"
                    torch.save(model_for_training.state_dict(), best_model_path)
                    print(f"New best validation loss! Model saved to {best_model_path}")

                # Visualize validation results
                center_loc_axis = find_label_center_loc(images[0, 0, ...])
                vis_image = get_xyz_plot(images[0, ...], center_loc_axis, mask_bool=False)
                vis_recon_image = get_xyz_plot(reconstruction[0, ...].cpu(), center_loc_axis, mask_bool=False)
                tensorboard_writer.add_image("val_original_image", vis_image.transpose([2, 0, 1]), epoch + 1)
                tensorboard_writer.add_image("val_reconstructed_image", vis_recon_image.transpose([2, 0, 1]), epoch + 1)

    if rank == 0:
        tensorboard_writer.close()
        print("Training finished.")
        print(f"Best validation VAE loss: {best_val_recon_epoch_loss:.4f}")
        print(f"Latest autoencoder saved to: {trained_g_path}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

