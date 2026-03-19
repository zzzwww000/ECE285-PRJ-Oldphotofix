"""
models/dacm.py

Degradation-Aware Conditioning Module (DACM) -- lightweight version.
Uses a frozen pretrained ResNet18 as shared backbone (Report Sec 2.3),
only trains two small heads:
    Branch 1: degradation type classification (multi-label)
    Branch 2: spatial severity map regression

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms.functional import normalize

# Degradation labels, matching utils/degradations.py
DEGRAD_TYPES = ["noise", "blur", "fading"]


class DACM(nn.Module):
    """
    Lightweight DACM with frozen ResNet18 backbone.

    Input:
        degraded image  (B, 3, H, W)  float32  [0, 1]

    Output dict:
        severity_score  (B,)                float [0, 1]
        severity_map    (B, 1, H/16, W/16)  float [0, 1]
        dtype_logits    (B, num_classes)     float
    """

    def __init__(self, num_classes=3, freeze_backbone=True):
        super().__init__()

        # -- shared backbone: ResNet18 up to layer3 --
        # layer2 output: (B, 128, H/8, W/8)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(
            resnet.conv1,    # /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # /4
            resnet.layer1,   # /4
            resnet.layer2,   # /8
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # -- branch 1: classification head --
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        # -- branch 2: severity map head --
        self.sev_head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        feat = self.backbone(x)                              # (B, 256, H/16, W/16)
        
        # branch 1
        dtype_logits = self.cls_head(
            self.cls_pool(feat).flatten(1)
        )

        # branch 2
        severity_map   = torch.sigmoid(self.sev_head(feat))  # (B, 1, H/16, W/16)
        severity_score = severity_map.mean(dim=(1, 2, 3))    # (B,)

        return {
            "severity_score": severity_score,
            "severity_map":   severity_map,
            "dtype_logits":   dtype_logits,
        }


# ------------------------------------------------------------------ #
#  Loss (unchanged)                                                    #
# ------------------------------------------------------------------ #

class DACMLoss(nn.Module):
    def __init__(self, lambda_s=1.0):
        super().__init__()
        self.lambda_s = lambda_s
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, pred, dtype_gt, sev_map_gt):
        loss_cls = self.bce(pred["dtype_logits"], dtype_gt)
        loss_sev = self.mse(pred["severity_map"], sev_map_gt)
        return loss_cls + self.lambda_s * loss_sev


# ------------------------------------------------------------------ #
#  Heuristic fallback (unchanged)                                      #
# ------------------------------------------------------------------ #

_LAP_KERNEL = torch.tensor(
    [[0, 1, 0],
     [1, -4, 1],
     [0, 1, 0]], dtype=torch.float32
).view(1, 1, 3, 3)

_REF_LAP_VAR  = 0.005
_REF_CONTRAST = 0.22


def heuristic_severity(image):
    device = image.device
    gray = (0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]).unsqueeze(1)

    lap_k = _LAP_KERNEL.to(device)
    laplacian = F.conv2d(gray, lap_k, padding=1)
    lap_var = laplacian.var(dim=(1, 2, 3))
    blur_severity = 1.0 - (lap_var / _REF_LAP_VAR).clamp(0, 1)

    contrast = gray.std(dim=(2, 3)).squeeze(1)
    fade_severity = 1.0 - (contrast / _REF_CONTRAST).clamp(0, 1)

    return (0.5 * blur_severity + 0.5 * fade_severity).clamp(0.0, 1.0)


# ------------------------------------------------------------------ #
#  Quick test                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    model = DACM(num_classes=3)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")

    dummy = torch.randn(2, 3, 256, 256).clamp(0, 1)
    out = model(dummy)
    print(f"severity_score: {out['severity_score']}")
    print(f"severity_map:   {out['severity_map'].shape}")
    print(f"dtype_logits:   {out['dtype_logits'].shape}")