"""
utils/degradations.py

Synthetic degradation pipeline for old photo simulation.
Each function returns (degraded_image, metadata_dict) so that
DACM training labels can be constructed from the known parameters.
"""

import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def add_gaussian_noise(img, sigma_range=(5, 25)):
    sigma = random.uniform(*sigma_range)
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, img_np.shape)
    noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    # severity: sigma normalized to [0, 1] by its max range
    denom = sigma_range[1] - sigma_range[0]
    severity = (sigma - sigma_range[0]) / denom if denom > 0 else 1.0
    return Image.fromarray(noisy), {"sigma": sigma, "severity": severity}


def add_blur(img, radius_range=(0.5, 2.0)):
    radius = random.uniform(*radius_range)
    blurred = img.filter(ImageFilter.GaussianBlur(radius))
    denom = radius_range[1] - radius_range[0]
    severity = (radius - radius_range[0]) / denom if denom > 0 else 1.0
    return blurred, {"radius": radius, "severity": severity}


def add_fading(img, brightness_range=(0.85, 1.0), contrast_range=(0.6, 0.95)):
    brightness = random.uniform(*brightness_range)
    contrast = random.uniform(*contrast_range)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    # lower brightness/contrast = more severe, so invert
    b_denom = brightness_range[1] - brightness_range[0]
    b_sev = 1.0 - ((brightness - brightness_range[0]) / b_denom if b_denom > 0 else 0.0) #
    c_denom = contrast_range[1] - contrast_range[0]
    c_sev = 1.0 - ((contrast - contrast_range[0]) / c_denom if c_denom > 0 else 1.0)
    severity = 0.5 * b_sev + 0.5 * c_sev
    return img, {"brightness": brightness, "contrast": contrast, "severity": severity}


def apply_random_degradation(img):
    """
    Randomly apply degradations with balanced sampling.

    Strategy:
        50% chance -> single degradation (equally split among 3 types)
        30% chance -> two degradations (random pair)
        20% chance -> all three
    """
    degraded = img.copy()

    applied = [False, False, False]
    params = [None, None, None]

    r = random.random()
    if r < 0.5:
        # single degradation
        idx = random.randint(0, 2)
        chosen = [idx]
    elif r < 0.8:
        # two degradations
        chosen = random.sample([0, 1, 2], 2)
    else:
        # all three
        chosen = [0, 1, 2]

    if 1 in chosen:
        degraded, p = add_blur(degraded)
        applied[1] = True
        params[1] = p
        
    if 2 in chosen:
        degraded, p = add_fading(degraded)
        applied[2] = True
        params[2] = p
        
    if 0 in chosen:
        degraded, p = add_gaussian_noise(degraded)
        applied[0] = True
        params[0] = p

    active_sevs = [params[i]["severity"] for i in range(3) if applied[i]]
    severity = sum(active_sevs) / len(active_sevs) if active_sevs else 0.0

    meta = {
        "applied": applied,
        "params": params,
        "severity": severity,
    }
    return degraded, meta