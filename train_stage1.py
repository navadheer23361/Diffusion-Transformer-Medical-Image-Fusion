#!/usr/bin/env python3
"""Train the Stage 1 diffusion UNets for CT and MR reconstruction."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.paired_ct_mr_dataset import PairedCTMRDataset
from models.stage1.diffusion_unet import GaussianDiffusion, Stage1DUNet
from utils.project_paths import CONFIG_PATH, STAGE1_MODELS_DIR


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(config: dict, split: str, batch_size: int, workers: int) -> DataLoader:
    """Build a dataloader for the requested split."""
    dataset = PairedCTMRDataset(config["datasets"][split], split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def batch_image(batch: dict, modality: str) -> torch.Tensor:
    """Return the modality-specific image tensor from a dataset batch."""
    if modality == "ct":
        return batch["ir"]
    if modality == "mr":
        return batch["vis"]
    raise ValueError("modality must be 'ct' or 'mr'")


def run_epoch(
    model: Stage1DUNet,
    loader: DataLoader,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    modality: str,
    max_batches: int | None = None,
) -> float:
    """Run one train or validation epoch and return mean L1 noise loss."""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        clean_image = batch_image(batch, modality).to(device)
        timestep = torch.randint(0, diffusion.timesteps, (clean_image.size(0),), device=device)
        noise = torch.randn_like(clean_image)
        noisy_image = diffusion.q_sample(clean_image, timestep, noise)

        with torch.set_grad_enabled(train_mode):
            predicted_noise = model(noisy_image, timestep)
            loss = F.l1_loss(predicted_noise, noise)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: Stage1DUNet,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    modality: str,
    epoch: int,
    val_loss: float,
) -> tuple[Path, Path]:
    """Save the full Stage 1 model and its encoder-only weights."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / f"{modality}_dunet_last.pth"
    encoder_path = out_dir / f"{modality}_encoder_last.pth"

    checkpoint = {
        "modality": modality,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_args": {
            "in_ch": 1,
            "base_ch": args.base_ch,
            "time_dim": args.time_dim,
        },
    }
    torch.save(checkpoint, full_path)
    torch.save(
        {
            "modality": modality,
            "epoch": epoch,
            "val_loss": val_loss,
            "encoder_state": model.encoder_state_dict(),
            "model_args": checkpoint["model_args"],
        },
        encoder_path,
    )
    return full_path, encoder_path


def train_modality(args: argparse.Namespace, config: dict, modality: str, device: torch.device) -> None:
    """Train one modality-specific Stage 1 diffusion UNet."""
    train_loader = make_loader(config, "train", args.batch_size, args.workers)
    val_loader = make_loader(config, "val", args.batch_size, args.workers)
    model = Stage1DUNet(base_ch=args.base_ch, time_dim=args.time_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)

    print(
        "Training {} DUNet on {} train slices, {} val slices".format(
            modality.upper(), len(train_loader.dataset), len(val_loader.dataset)
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            diffusion,
            optimizer,
            device,
            modality,
            max_batches=args.max_train_batches,
        )
        val_loss = run_epoch(
            model,
            val_loader,
            diffusion,
            None,
            device,
            modality,
            max_batches=args.max_val_batches,
        )
        full_path, encoder_path = save_checkpoint(model, optimizer, args, modality, epoch, val_loss)
        print(
            "{} epoch {:03d}/{:03d} train_l1={:.5f} val_l1={:.5f} saved {} {}".format(
                modality.upper(),
                epoch,
                args.epochs,
                train_loss,
                val_loss,
                full_path,
                encoder_path,
            )
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 1 training."""
    parser = argparse.ArgumentParser(description="Train the Stage 1 diffusion UNets.")
    parser.add_argument("-c", "--config", default=CONFIG_PATH)
    parser.add_argument("--modality", choices=["ct", "mr", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=STAGE1_MODELS_DIR)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run Stage 1 training for one or both modalities."""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    modalities = ["ct", "mr"] if args.modality == "both" else [args.modality]
    for modality in modalities:
        train_modality(args, config, modality, device)


if __name__ == "__main__":
    main()
