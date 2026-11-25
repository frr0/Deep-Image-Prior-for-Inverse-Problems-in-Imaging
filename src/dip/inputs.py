"""Fixed input tensor generators z for Deep Image Prior."""

from __future__ import annotations

import numpy as np
import torch


def create_input_tensor(
    input_type: str,
    spatial_size: tuple[int, int],
    input_depth: int = 32,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> torch.Tensor:
    """Generate the fixed network input z of shape (1, C, H, W).

    Args:
        input_type: One of 'uniform', 'normal', or 'meshgrid'.
        spatial_size: (H, W) of the target image.
        input_depth: Number of channels for random noise inputs (ignored for meshgrid, which uses 2).
        device: Target torch device.
        seed: Random seed for reproducibility.

    Returns:
        Fixed input tensor z on `device`.
    """
    h, w = spatial_size
    gen = torch.Generator(device="cpu").manual_seed(seed)
    kind = input_type.lower()

    if kind == "uniform":
        z = torch.empty((1, input_depth, h, w), dtype=torch.float32).uniform_(0.0, 0.1, generator=gen)
    elif kind == "normal":
        z = torch.empty((1, input_depth, h, w), dtype=torch.float32).normal_(0.0, 0.1, generator=gen)
    elif kind == "meshgrid":
        x_lin = np.linspace(0.0, 1.0, w, dtype=np.float32)
        y_lin = np.linspace(0.0, 1.0, h, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x_lin, y_lin)
        mesh = np.stack([grid_x, grid_y], axis=0)[None, ...]
        z = torch.from_numpy(mesh)
    else:
        raise ValueError(f"Unknown input_type '{input_type}'. Choose 'uniform', 'normal', or 'meshgrid'.")

    return z.to(device)
