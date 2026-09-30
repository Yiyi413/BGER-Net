from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_boundary_from_mask(mask: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B, 1, H, W] or [B, H, W]")
    mask = (mask.float() > 0.5).float()
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-mask, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp_(0.0, 1.0)


class DiceLossWithLogits(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(1)
        targets = targets.float().flatten(1)
        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth + self.eps
        )
        return (1.0 - dice).mean()


class BGERLoss(nn.Module):
    def __init__(self, lambda_boundary: float = 0.3):
        super().__init__()
        self.lambda_boundary = lambda_boundary
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossWithLogits()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mask_gt: torch.Tensor,
        boundary_gt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask_gt.ndim == 3:
            mask_gt = mask_gt.unsqueeze(1)
        mask_gt = mask_gt.float()
        mask_logits = outputs["mask_logits"]
        mask_loss = self.bce(mask_logits, mask_gt) + self.dice(mask_logits, mask_gt)
        if self.lambda_boundary == 0.0:
            return mask_loss

        if boundary_gt is None:
            boundary_gt = make_boundary_from_mask(mask_gt)
        elif boundary_gt.ndim == 3:
            boundary_gt = boundary_gt.unsqueeze(1)
        boundary_gt = boundary_gt.float()
        boundary_logits = outputs["boundary_logits"]
        boundary_loss = self.bce(boundary_logits, boundary_gt)
        return mask_loss + self.lambda_boundary * boundary_loss
