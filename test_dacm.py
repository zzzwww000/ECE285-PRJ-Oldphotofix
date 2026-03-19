"""
test_dacm.py

Standalone DACM inference: load checkpoint, run on a single image,
print degradation type and severity score.

Usage:
    python test_dacm.py --image test_photo.jpg --ckpt checkpoints/dacm_last.pth
"""

import argparse
import torch
from utils.preprocessing import load_image

from models.dacm import DACM, DEGRAD_TYPES, heuristic_severity
from utils.scheduler import severity_to_teff


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      type=str, required=True)
    p.add_argument("--ckpt",       type=str, default="checkpoints/dacm_last.pth")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--heuristic",  action="store_true", help="also run heuristic fallback")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- load image --
    tensor = load_image(args.image, image_size=args.image_size, device=device)   # (1, 3, 256, 256)

    # -- load model --
    model = DACM(num_classes=3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # -- inference --
    with torch.no_grad():
        out = model(tensor)

    score = out["severity_score"].item()
    logits = out["dtype_logits"].squeeze(0)
    probs = torch.sigmoid(logits)

    # -- print results --
    print(f"\nImage: {args.image}")
    print(f"{'='*40}")

    print(f"\nDegradation types detected:")
    for name, prob in zip(DEGRAD_TYPES, probs):
        tag = " <--" if prob > 0.5 else ""
        print(f"  {name:>8s}: {prob:.3f}{tag}")

    print(f"\nSeverity score: {score:.4f}")
    teff = severity_to_teff(score)
    print(f"  -> Teff = {teff} DDIM steps (range 20-150)")

    # -- severity map stats --
    smap = out["severity_map"].squeeze()   # (16, 16)
    print(f"\nSeverity map ({smap.shape[0]}x{smap.shape[1]}):")
    print(f"  min={smap.min():.4f}  max={smap.max():.4f}  mean={smap.mean():.4f}")

    # -- optional heuristic --
    if args.heuristic:
        h_score = heuristic_severity(tensor).item()
        h_teff = severity_to_teff(h_score)
        print(f"\nHeuristic fallback:")
        print(f"  severity={h_score:.4f}  -> Teff={h_teff}")

    print()


if __name__ == "__main__":
    main()