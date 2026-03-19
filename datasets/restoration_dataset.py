import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

from utils.degradations import apply_random_degradation


class RestorationDataset(Dataset):
    """
    DACM-based image restoration dataset.
    """

    def __init__(self, image_dir, image_size=64):
        """
        input:
            image_dir: the directory containing clean images for training
            image_size: size to resize images to (image_size, image_size)
        """
        self.image_dir = image_dir
        self.image_size = image_size

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        self.image_paths = []

        for fname in os.listdir(image_dir):
            if fname.lower().endswith(valid_exts):
                self.image_paths.append(os.path.join(image_dir, fname))

        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise ValueError(f" {image_dir} does not contain any valid image files.")

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        clean_img = Image.open(img_path).convert("RGB")
        clean_img = self.base_transform(clean_img)

        degraded_img, meta = apply_random_degradation(clean_img)

        clean_tensor = self.to_tensor(clean_img)
        degraded_tensor = self.to_tensor(degraded_img)

        return {
            "clean": clean_tensor,
            "degraded": degraded_tensor,
            "path": img_path,
            "dtype_label": torch.tensor(meta["applied"], dtype=torch.float32),  # (3,) multi-hot
            "severity": torch.tensor(meta["severity"], dtype=torch.float32),    # scalar
        }