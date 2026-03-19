"""
train_dacm.py

Train the Degradation-Aware Conditioning Module (DACM).
Uses RestorationDataset which now returns dtype_label and severity.

Usage:
    python train_dacm.py --image_dir data/train --epochs 30 --batch_size 16
    python train_dacm.py --image_dir data/train --epochs 30 --log_dir logs/dacm

Monitor:
    Training prints loss every N steps.
    Loss history saved to log_dir/loss.csv for plotting.
    Checkpoints saved every save_every epochs.
"""

import os
import csv
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.restoration_dataset import RestorationDataset
from models.dacm import DACM, DACMLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",   type=str,   default="data/train")
    p.add_argument("--image_size",  type=int,   default=256)
    p.add_argument("--batch_size",  type=int,   default=16)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--lambda_s",    type=float, default=1.0,  help="severity loss weight")
    p.add_argument("--log_every",   type=int,   default=20,   help="print every N steps")
    p.add_argument("--save_every",  type=int,   default=5,    help="save ckpt every N epochs")
    p.add_argument("--log_dir",     type=str,   default="logs/dacm")
    p.add_argument("--ckpt_dir",    type=str,   default="checkpoints")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def build_severity_map_gt(severity_scalar, map_h, map_w):
    """
    Build spatial severity map ground truth from global scalar.

    Current synthetic degradations (noise, blur, fading) are all spatially
    uniform, so GT severity map is a constant map filled with the global score.
    When spatially-varying degradations (e.g. scratches) are added later,
    this function should be updated to produce non-uniform maps.

    Args:
        severity_scalar: (B,) tensor
        map_h, map_w:    spatial dims of DACM severity map output (H/16, W/16)

    Returns:
        (B, 1, map_h, map_w) tensor
    """
    return severity_scalar.view(-1, 1, 1, 1).expand(-1, 1, map_h, map_w)


def train():
    args = parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # -- data --
    dataset = RestorationDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    print(f"dataset: {len(dataset)} images, {len(loader)} batches/epoch")

    # -- model --
    model = DACM(num_classes=3, freeze_backbone=False).to(device)
    criterion = DACMLoss(lambda_s=args.lambda_s)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"DACM parameters: {param_count:,}")

    # -- severity map spatial dims (256 / 16 = 16) --
    map_h = map_w = args.image_size // 8

    # -- logging --
    csv_path = os.path.join(args.log_dir, "loss.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "step", "loss", "loss_cls", "loss_sev", "lr"])

    # -- train loop --
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            degraded   = batch["degraded"].to(device)       # (B, 3, 256, 256)
            dtype_gt   = batch["dtype_label"].to(device)    # (B, 3) multi-hot
            severity   = batch["severity"].to(device)       # (B,)

            sev_map_gt = build_severity_map_gt(severity, map_h, map_w).to(device)

            pred = model(degraded)
            loss = criterion(pred, dtype_gt, sev_map_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # -- log detail --
            if global_step % args.log_every == 0:
                # recompute individual losses for logging
                with torch.no_grad():
                    l_cls = nn.functional.binary_cross_entropy_with_logits(
                        pred["dtype_logits"], dtype_gt
                    ).item()
                    l_sev = nn.functional.mse_loss(
                        pred["severity_map"], sev_map_gt
                    ).item()
                lr_now = optimizer.param_groups[0]["lr"]
                csv_writer.writerow([
                    epoch, global_step,
                    f"{loss.item():.5f}", f"{l_cls:.5f}", f"{l_sev:.5f}", f"{lr_now:.6f}"
                ])
                csv_file.flush()
                print(
                    f"  [epoch {epoch:>3d} | step {global_step:>5d}] "
                    f"loss={loss.item():.4f}  cls={l_cls:.4f}  sev={l_sev:.4f}  "
                    f"lr={lr_now:.6f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0
        print(f"epoch {epoch:>3d}/{args.epochs}  avg_loss={avg_loss:.4f}  time={elapsed:.1f}s")

        # -- save checkpoint --
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.ckpt_dir, f"dacm_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  saved: {ckpt_path}")

    csv_file.close()

    # save final
    final_path = os.path.join(args.ckpt_dir, "dacm_last.pth")
    torch.save(model.state_dict(), final_path)
    print(f"training done. final model: {final_path}")
    print(f"loss log: {csv_path}")


if __name__ == "__main__":
    train()