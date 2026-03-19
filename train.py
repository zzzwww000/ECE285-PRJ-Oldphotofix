'''
Training script for image restoration using a conditional U-Net and diffusion model.


'''
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.restoration_dataset import RestorationDataset
from models.unet import ConditionalUNet
from models.diffusion import Diffusion


def train():
    # -------------------------
    # 1. Settings
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    image_dir = "data/train"
    image_size = 256
    batch_size = 4
    lr = 1e-4
    timesteps = 200
    # ~120 epochs for current dataset size (~1464 steps/epoch)
    target_steps = 175680
    auto_resume = True
    resume_state_path = "checkpoints/train_state_last.pth"

    # -------------------------
    # 2. Dataset & Dataloader
    # -------------------------
    dataset = RestorationDataset(
        image_dir=image_dir,
        image_size=image_size
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("No training samples found.")

    epochs = max(1, math.ceil(target_steps / steps_per_epoch))
    total_steps = steps_per_epoch * epochs
    save_every = max(1, epochs // 20)  # keep roughly 20 checkpoints
    print(f"dataset_size: {len(dataset)}")
    print(f"steps/epoch: {steps_per_epoch}")
    print(f"target_steps: {target_steps}, epochs: {epochs}, total_steps: {total_steps}")
    print(f"checkpoint_every: {save_every} epochs")

    # -------------------------
    # 3. Model, Diffusion, Optimizer, Loss
    # -------------------------
    model = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256
    ).to(device)

    diffusion = Diffusion(
        timesteps=timesteps,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    start_epoch = 0
    global_step = 0
    run_config = {
        "image_size": image_size,
        "batch_size": batch_size,
        "timesteps": timesteps,
        "target_steps": target_steps,
    }
    resume_compat_config = {
        "image_size": image_size,
        "batch_size": batch_size,
        "timesteps": timesteps,
    }
    if auto_resume and os.path.exists(resume_state_path):
        state = torch.load(resume_state_path, map_location=device)
        saved_cfg = state.get("config", {})
        saved_compat_config = {k: saved_cfg.get(k) for k in resume_compat_config.keys()}
        if saved_cfg and saved_compat_config != resume_compat_config:
            print("resume state config mismatch, skip auto-resume.")
            print(f"saved config: {saved_cfg}")
            print(f"current config: {run_config}")
        else:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state["global_step"])
            print(f"resumed from epoch={start_epoch}, global_step={global_step}")

    # -------------------------
    # 4. Checkpoint dir
    # -------------------------
    os.makedirs("checkpoints", exist_ok=True)

    # -------------------------
    # 5. Training Loop
    # -------------------------
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            clean = batch["clean"].to(device)        
            degraded = batch["degraded"].to(device)   

            t = diffusion.sample_timesteps(clean.shape[0])

            xt, noise = diffusion.noise_images(clean, t)

            noise_pred = model(xt, degraded, t)

            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            ckpt_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"checkpoint saved to: {ckpt_path}")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": run_config,
                },
                resume_state_path
            )
            print(f"resume state saved to: {resume_state_path}")

    # Save final model
    torch.save(model.state_dict(), "checkpoints/model_last.pth")
    print("final model saved to: project/checkpoints/model_last.pth")


if __name__ == "__main__":
    train()
