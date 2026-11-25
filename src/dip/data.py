"""Sample image loaders and tensor/image conversion utilities."""

from __future__ import annotations

import io
from typing import Literal

import numpy as np
import torch
from PIL import Image
from skimage import data as skdata

SampleImageName = Literal["astronaut", "chelsea", "camera", "coffee", "shepp_logan"]


def _resize_and_format(
    arr: np.ndarray,
    size: int = 128,
    grayscale: bool = False,
) -> np.ndarray:
    """Center-crop to square, resize to (size, size), and normalize to float32 in [0, 1]."""
    if arr.ndim == 2:
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if grayscale:
        pil = pil.convert("L")
    elif pil.mode != "L":
        pil = pil.convert("RGB")

    w, h = pil.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    pil = pil.crop((left, top, left + side, top + side))
    pil = pil.resize((size, size), Image.Resampling.BICUBIC)

    out = np.asarray(pil, dtype=np.float32) / 255.0
    return out


def numpy_to_tensor(img: np.ndarray, device: torch.device | str = "cpu") -> torch.Tensor:
    """Convert HxW or HxWxC float32 array in [0, 1] to (1, C, H, W) tensor."""
    if img.ndim == 2:
        t = torch.from_numpy(img)[None, None, :, :]
    elif img.ndim == 3:
        t = torch.from_numpy(np.transpose(img, (2, 0, 1)))[None, :, :, :]
    else:
        raise ValueError(f"Expected 2D or 3D image array, got shape {img.shape}.")
    return t.float().to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, C, H, W) tensor to HxW or HxWxC float32 numpy array in [0, 1]."""
    arr = tensor.detach().clamp(0.0, 1.0).squeeze(0).cpu().numpy()
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        return np.transpose(arr, (1, 2, 0))
    return arr


def load_sample_image(
    name: SampleImageName | str = "astronaut",
    size: int = 128,
    grayscale: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load one of the built-in offline sample images as a (1, C, H, W) tensor."""
    key = name.lower()
    if key == "astronaut":
        raw = skdata.astronaut()
    elif key == "chelsea":
        raw = skdata.chelsea()
    elif key == "camera":
        raw = skdata.camera()
    elif key == "coffee":
        raw = skdata.coffee()
    elif key == "shepp_logan":
        raw = (skdata.shepp_logan_phantom() * 255.0).astype(np.uint8)
        grayscale = True
    else:
        raise ValueError(
            f"Unknown sample image '{name}'. Choose from: astronaut, chelsea, camera, coffee, shepp_logan."
        )

    arr = _resize_and_format(raw, size=size, grayscale=grayscale)
    return numpy_to_tensor(arr, device=device)


def load_image_from_bytes(
    data: bytes,
    size: int = 128,
    grayscale: bool = False,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load an uploaded image from bytes as a (1, C, H, W) tensor."""
    pil = Image.open(io.BytesIO(data))
    arr = np.asarray(pil)
    formatted = _resize_and_format(arr, size=size, grayscale=grayscale)
    return numpy_to_tensor(formatted, device=device)
