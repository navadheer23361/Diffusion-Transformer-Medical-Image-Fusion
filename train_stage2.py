#!/usr/bin/env python3
"""Train the Stage 2 fusion head on frozen Stage 1 diffusion encoders."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.paired_ct_mr_dataset import PairedCTMRDataset
from models.losses.fusion_loss import Fusionloss
from models.stage1.diffusion_unet import GaussianDiffusion, Stage1DUNet
from models.stage2.feature_extraction import extract_stage1_dual_features
from models.stage2.fusion_head import Fusion_Head
from utils.project_paths import CONFIG_PATH, STAGE1_MODELS_DIR, STAGE2_MODELS_DIR


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(config: dict, split: str, batch_size: int, workers: int) -> DataLoader:
    """Build a dataloader for Stage 2 training or validation."""
    dataset = PairedCTMRDataset(config["datasets"][split], split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )


def load_dunet(path: str | Path, device: torch.device) -> tuple[Stage1DUNet, dict]:
    """Load a frozen Stage 1 diffusion UNet checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model_args = checkpoint.get("model_args", {})
    model = Stage1DUNet(
        in_ch=model_args.get("in_ch", 1),
        base_ch=model_args.get("base_ch", 32),
        time_dim=model_args.get("time_dim", 128),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, model_args


def run_epoch(
    fusion: Fusion_Head,
    ct_model: Stage1DUNet,
    mr_model: Stage1DUNet,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    loss_fn: Fusionloss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    time_steps: list[int],
    max_batches: int | None = None,
) -> float:
    """Run one Stage 2 train or validation epoch."""
    train_mode = optimizer is not None
    fusion.train(train_mode)
    total = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break

        data = {
            "ir": batch["ir"].to(device),
            "vis": batch["vis"].to(device),
        }
        feats = extract_stage1_dual_features(
            ct_model, mr_model, diffusion, batch, time_steps, device
        )

        with torch.set_grad_enabled(train_mode):
            fused = fusion(data["ir"], data["vis"], feats)
            loss_in, loss_grad = loss_fn(data, fused)
            loss = loss_in + loss_grad
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total += loss.item()
        n_batches += 1

    return total / max(n_batches, 1)


def save_checkpoint(
    fusion: Fusion_Head,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
) -> Path:
    """Save the latest Stage 2 fusion checkpoint."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fusion_stage2_last.pth"
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "fusion_state": fusion.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ct_checkpoint": args.ct_checkpoint,
            "mr_checkpoint": args.mr_checkpoint,
            "model_args": {
                "inner_channel": args.base_ch,
                "channel_multiplier": [1, 2, 4, 8, 8],
                "feat_scales": [2, 5, 8, 11, 14],
                "time_steps": args.time_steps,
                "num_heads": args.num_heads,
                "max_attention_tokens": args.max_attention_tokens,
            },
        },
        path,
    )
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for Stage 2 fusion training."""
    parser = argparse.ArgumentParser(description="Train the Stage 2 fusion head.")
    parser.add_argument("-c", "--config", default=CONFIG_PATH)
    parser.add_argument("--ct-checkpoint", default=f"{STAGE1_MODELS_DIR}/ct_dunet_last.pth")
    parser.add_argument("--mr-checkpoint", default=f"{STAGE1_MODELS_DIR}/mr_dunet_last.pth")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=STAGE2_MODELS_DIR)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--time-steps", type=int, nargs="+", default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--max-attention-tokens", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run Stage 2 fusion training."""
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)

    ct_model, ct_args = load_dunet(args.ct_checkpoint, device)
    mr_model, mr_args = load_dunet(args.mr_checkpoint, device)
    if ct_args.get("base_ch") != mr_args.get("base_ch"):
        raise ValueError("CT and MR DUNets must use the same base_ch for fusion.")

    args.base_ch = int(ct_args.get("base_ch", 32))
    model_df = config["model_df"]
    args.time_steps = args.time_steps or model_df.get("t", [5, 10, 20])
    args.num_heads = args.num_heads or model_df.get("num_heads", 8)
    args.max_attention_tokens = (
        args.max_attention_tokens or model_df.get("max_attention_tokens", 1024)
    )

    train_loader = make_loader(config, "train", args.batch_size, args.workers)
    val_loader = make_loader(config, "val", args.batch_size, args.workers)

    diffusion = GaussianDiffusion(timesteps=args.timesteps, device=device)
    fusion = Fusion_Head(
        feat_scales=[2, 5, 8, 11, 14],
        out_channels=1,
        inner_channel=args.base_ch,
        channel_multiplier=[1, 2, 4, 8, 8],
        img_size=config["model_df"].get("output_cm_size", [224, 256]),
        time_steps=args.time_steps,
        num_heads=args.num_heads,
        max_tokens=args.max_attention_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    loss_fn = Fusionloss(
        alpha=config["train"].get("alpha", 1.5),
        beta=config["train"].get("beta", 0.5),
        lambda_boundary=config["train"].get("lambda_boundary", 0.3),
        lambda_adaptive=config["train"].get("lambda_adaptive", 0.2),
        gamma=config["train"].get("drgo_gamma", 0.5),
    ).to(device)

    print("Stage 2 fusion training")
    print("CT checkpoint:", args.ct_checkpoint)
    print("MR checkpoint:", args.mr_checkpoint)
    print("Train slices:", len(train_loader.dataset), "Val slices:", len(val_loader.dataset))

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            fusion,
            ct_model,
            mr_model,
            diffusion,
            train_loader,
            loss_fn,
            optimizer,
            device,
            args.time_steps,
            args.max_train_batches,
        )
        val_loss = run_epoch(
            fusion,
            ct_model,
            mr_model,
            diffusion,
            val_loader,
            loss_fn,
            None,
            device,
            args.time_steps,
            args.max_val_batches,
        )
        path = save_checkpoint(fusion, optimizer, args, epoch, val_loss)
        print(
            "Stage2 epoch {:03d}/{:03d} train_loss={:.5f} val_loss={:.5f} saved {}".format(
                epoch, args.epochs, train_loss, val_loss, path
            )
        )


if __name__ == "__main__":
    main()
