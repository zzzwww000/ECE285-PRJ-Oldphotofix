import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets.restoration_dataset import RestorationDataset
from models.unet import ConditionalUNet
from models.diffusion import Diffusion
from models.ddim_restoration import restore_with_ddim
from utils.scheduler import severity_to_teff
from models.dacm import DACM

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse <= 1e-12:
        return 99.0
    return (-10.0 * torch.log10(mse)).item()


def resolve_teff(teff, severity):
    if teff is not None:
        return int(teff)
    if severity is not None:
        return severity_to_teff(severity)
    return 80


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/train")
    parser.add_argument("--ckpt", type=str, default="checkpoints/model_last.pth")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--sample_index", type=int, default=798)
    parser.add_argument("--teff", type=int, default=None, help="Effective DDIM steps")
    parser.add_argument("--severity", type=float, default=None, help="Severity score in [0, 1]")
    parser.add_argument("--init_mode", type=str, default="img2img", choices=["img2img", "pure_noise"])
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="ddim_result.png")
    parser.add_argument("--dacm_ckpt", type=str, default="checkpoints/dacm_last.pth", help="DACM checkpoint path for auto severity estimation")
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = RestorationDataset(
        image_dir=args.image_dir,
        image_size=args.image_size,
    )

    sample_idx = max(0, min(args.sample_index, len(dataset) - 1))
    sample = dataset[sample_idx]
    clean = sample["clean"].unsqueeze(0).to(device)
    degraded = sample["degraded"].unsqueeze(0).to(device)

    model = ConditionalUNet(
        in_channels=6,
        out_channels=3,
        base_channels=64,
        time_emb_dim=256,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    diffusion = Diffusion(
        timesteps=args.timesteps,
        device=device,
    )
    
    dacm = None
    if args.dacm_ckpt is not None:
        dacm = DACM(num_classes=3).to(device)
        dacm.load_state_dict(torch.load(args.dacm_ckpt, map_location=device))
        dacm.eval()

    if args.teff is not None:
        teff = int(args.teff)
    elif args.severity is not None:
        teff = severity_to_teff(args.severity)
    elif dacm is not None:
        with torch.no_grad():
            dacm_out = dacm(degraded)
        severity = dacm_out["severity_score"].item()
        probs = torch.sigmoid(dacm_out["dtype_logits"][0])  # [noise, blur, fading]
        
        noise_prob = probs[0].item()
        blur_prob = probs[1].item()
        fading_prob = probs[2].item()

        adjustment = 0.0
        if blur_prob > 0.5:
            adjustment += 0.15 * blur_prob
        if noise_prob > 0.5:
            adjustment += 0.10 * noise_prob 
            teff = severity_to_teff(severity)
        adjusted_severity = min(1.0, severity + adjustment)
        teff = severity_to_teff(adjusted_severity)        
    else:
        teff = 80

    print(f"teff: {teff}, init_mode: {args.init_mode}, strength: {args.strength:.2f}")

    restored = restore_with_ddim(
        model=model,
        diffusion=diffusion,
        degraded=degraded,
        teff=teff,
        init_mode=args.init_mode,
        strength=args.strength,
        x_range=(0.0, 1.0),
    )

    degraded_psnr = psnr(degraded, clean)
    restored_psnr = psnr(restored, clean)
    print(f"PSNR(degraded, clean): {degraded_psnr:.2f} dB")
    print(f"PSNR(restored, clean): {restored_psnr:.2f} dB")

    clean_img = clean.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    degraded_img = degraded.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    restored_img = restored.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Clean")
    plt.imshow(clean_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Degraded")
    plt.imshow(degraded_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Restored (Teff={teff})")
    plt.imshow(restored_img)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150)
    plt.show()
    print(f"saved: {args.save_path}")


if __name__ == "__main__":
    main()
