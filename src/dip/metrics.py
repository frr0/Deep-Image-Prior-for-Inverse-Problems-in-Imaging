"""Image quality metrics and regularization losses."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as skimage_ssim


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between two tensors in [0, 1]."""
    with torch.no_grad():
        mse = torch.mean((pred.detach().clamp(0.0, max_val) - target.detach().clamp(0.0, max_val)) ** 2).item()
        if mse <= 1e-12:
            return 100.0
        return float(10.0 * np.log10((max_val**2) / mse))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Structural Similarity Index (SSIM) between two (1, C, H, W) tensors."""
    with torch.no_grad():
        p = pred.detach().clamp(0.0, max_val).squeeze(0).cpu().numpy()
        t = target.detach().clamp(0.0, max_val).squeeze(0).cpu().numpy()

    if p.ndim == 3:
        if p.shape[0] == 1:
            return float(skimage_ssim(t[0], p[0], data_range=max_val))
        p_hwc = np.transpose(p, (1, 2, 0))
        t_hwc = np.transpose(t, (1, 2, 0))
        return float(skimage_ssim(t_hwc, p_hwc, data_range=max_val, channel_axis=-1))
    return float(skimage_ssim(t, p, data_range=max_val))


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Anisotropic Total Variation (TV) penalty for optional DIP+TV regularization."""
    dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    return dh + dw
