"""
utils/preprocessing.py

Image Input module -- first row of the project interface table.

Loads an image from various sources (path / PIL / numpy / tensor),
converts it to the standard format expected by the rest of the pipeline:
    torch.Tensor, float32, [0, 1], RGB, shape (1, 3, H, W)

Matches the existing data pipeline in datasets/restoration_dataset.py:
    - Resize with default bilinear interpolation
    - ToTensor for [0, 1] normalization
"""

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------- default config (align with sample_ddim.py) ----------
DEFAULT_IMAGE_SIZE = 256


# ---------- core transform ----------
def _build_transform(image_size=DEFAULT_IMAGE_SIZE):
    """
    Build the same transform chain used in RestorationDataset
    so that preprocessing is consistent between training and inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # bilinear, same as dataset
        transforms.ToTensor(),                        # HWC uint8 -> CHW float32 [0,1]
    ])


# ---------- unified loader ----------
def _to_pil(source):
    """
    Convert various input types to an RGB PIL Image.

    Supported:
        str / os.PathLike  - file path
        PIL.Image           - pass through
        numpy.ndarray       - (H,W,3) uint8 or float, or (H,W) grayscale
        torch.Tensor        - (C,H,W) or (B,C,H,W)
    """
    # file path
    if isinstance(source, (str, os.PathLike)):
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Image not found: {source}")
        return Image.open(source).convert("RGB")

    # PIL
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    # numpy
    if isinstance(source, np.ndarray):
        arr = source
        # grayscale -> 3ch
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        # float [0,1] -> uint8
        if arr.dtype in (np.float32, np.float64):
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255)
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    # torch tensor
    if isinstance(source, torch.Tensor):
        t = source.detach().cpu()
        if t.ndim == 4:
            t = t[0]                       # (B,C,H,W) -> (C,H,W)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)         # CHW -> HWC
        arr = t.numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        return Image.fromarray(arr).convert("RGB")

    raise TypeError(f"Unsupported input type: {type(source)}")


def load_image(source, image_size=DEFAULT_IMAGE_SIZE, device="cpu"):
    """
    Load a single image and return a tensor ready for the pipeline.

    Args:
        source:     file path, PIL Image, numpy array, or torch Tensor
        image_size: target spatial size (square)
        device:     target device ("cpu" / "cuda" / torch.device)

    Returns:
        torch.Tensor of shape (1, 3, image_size, image_size),
        dtype float32, range [0.0, 1.0], RGB channel order.
    """
    pil_img = _to_pil(source)
    transform = _build_transform(image_size)
    tensor = transform(pil_img)   # (3, H, W)
    return tensor.unsqueeze(0).to(device)


def load_image_batch(sources, image_size=DEFAULT_IMAGE_SIZE, device="cpu"):
    """
    Load multiple images and stack into a single batch.

    Args:
        sources: list of any supported source types

    Returns:
        torch.Tensor of shape (B, 3, image_size, image_size)
    """
    tensors = [load_image(s, image_size, device) for s in sources]
    return torch.cat(tensors, dim=0)


# ---------- inverse: tensor -> PIL (for saving / visualization) ----------
def tensor_to_pil(tensor):
    """
    Convert a pipeline tensor back to a PIL Image.
    Accepts (3, H, W) or (1, 3, H, W), float32 [0,1].
    """
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[0]
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def save_image(tensor, path):
    """Convenience wrapper: tensor -> PIL -> save."""
    tensor_to_pil(tensor).save(path)


# ---------- quick sanity check ----------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m utils.preprocessing <image_path>")
        sys.exit(0)

    img = load_image(sys.argv[1])
    print(f"shape : {img.shape}")         # (1, 3, 256, 256)
    print(f"dtype : {img.dtype}")         # float32
    print(f"range : [{img.min():.4f}, {img.max():.4f}]")  # [0, 1]

    # round-trip test
    save_image(img, "_preprocess_check.png")
    reloaded = load_image("_preprocess_check.png")
    diff = (img - reloaded).abs().max().item()
    print(f"round-trip max diff: {diff:.6f}")
    os.remove("_preprocess_check.png")
    print("OK")