"""Differentiable forward operators A(x) for inverse problems in imaging.

Supports:
1. Denoising:               A(x) = x,  y = x + noise
2. Inpainting:              A(x) = M * x
3. Deblurring (+ noise):    A(x) = k * x
4. Super-Resolution:        A(x) = Downsample_s(x)
5. X-Ray Tomography (CT):   A(x) = Radon(x)  (differentiable parallel-beam sinogram)
6. Compressed Sensing:      A(x) = Phi x  (block random Gaussian matrix or subsampled 2D FFT)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardOperator(nn.Module, ABC):
    """Base class for differentiable imaging forward operators."""

    name: str = "base"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply forward operator A(x)."""

    def corrupt(
        self,
        x: torch.Tensor,
        noise_std: float = 0.0,
        seed: int = 42,
        clamp_output: bool = False,
    ) -> torch.Tensor:
        """Simulate measurement y = A(x) + epsilon."""
        with torch.no_grad():
            y = self.forward(x)
            if noise_std > 0.0:
                gen = torch.Generator(device="cpu").manual_seed(seed)
                noise = torch.empty(y.shape, dtype=y.dtype).normal_(0.0, noise_std, generator=gen).to(y.device)
                y = y + noise
            if clamp_output:
                y = y.clamp(0.0, 1.0)
            return y

    def visualize_measurement(self, y: torch.Tensor) -> np.ndarray:
        """Convert measurement tensor into a displayable [0, 1] HxW or HxWxC numpy array."""
        arr = y.detach().cpu().squeeze(0).numpy()
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = np.transpose(arr, (1, 2, 0))
        return np.clip(arr, 0.0, 1.0)


class DenoisingOperator(ForwardOperator):
    """Identity forward operator A(x) = x for additive noise removal."""

    name = "denoising"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def corrupt(
        self,
        x: torch.Tensor,
        noise_std: float = 0.1,
        seed: int = 42,
        clamp_output: bool = True,
    ) -> torch.Tensor:
        return super().corrupt(x, noise_std=noise_std, seed=seed, clamp_output=clamp_output)


class InpaintingOperator(ForwardOperator):
    """Element-wise binary mask operator A(x) = M * x."""

    name = "inpainting"

    def __init__(self, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask.float())

    @classmethod
    def from_strategy(
        cls,
        shape: tuple[int, int, int, int],
        strategy: str = "random",
        keep_ratio: float = 0.5,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> "InpaintingOperator":
        """Create an InpaintingOperator with a generated binary mask.

        Args:
            shape: (1, C, H, W) tensor shape.
            strategy: 'random' (Bernoulli pixels), 'block' (center hole), or 'stripes' (vertical missing lines).
            keep_ratio: Fraction of observed pixels (used for 'random' and 'stripes').
            seed: Random seed.
            device: Target torch device.
        """
        _, _, h, w = shape
        rng = np.random.default_rng(seed)

        if strategy == "random":
            mask_2d = (rng.random((h, w)) < keep_ratio).astype(np.float32)
        elif strategy == "block":
            mask_2d = np.ones((h, w), dtype=np.float32)
            hole_h = max(8, int(h * math.sqrt(max(0.05, 1.0 - keep_ratio))))
            hole_w = max(8, int(w * math.sqrt(max(0.05, 1.0 - keep_ratio))))
            y0 = (h - hole_h) // 2
            x0 = (w - hole_w) // 2
            mask_2d[y0 : y0 + hole_h, x0 : x0 + hole_w] = 0.0
        elif strategy == "stripes":
            mask_2d = np.ones((h, w), dtype=np.float32)
            drop_cols = rng.random(w) > keep_ratio
            mask_2d[:, drop_cols] = 0.0
        else:
            raise ValueError(f"Unknown mask strategy '{strategy}'. Choose 'random', 'block', or 'stripes'.")

        mask = torch.from_numpy(mask_2d)[None, None, :, :].to(device)
        return cls(mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.mask


class DeblurringOperator(ForwardOperator):
    """2D convolution forward operator A(x) = k * x (Gaussian or linear motion blur)."""

    name = "deblurring"

    def __init__(
        self,
        kernel_type: str = "gaussian",
        kernel_size: int = 11,
        sigma: float = 2.0,
        motion_angle_deg: float = 30.0,
        channels: int = 3,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.channels = channels

        if kernel_type == "gaussian":
            k2d = self._make_gaussian_kernel(kernel_size, sigma)
        elif kernel_type == "motion":
            k2d = self._make_motion_kernel(kernel_size, motion_angle_deg)
        else:
            raise ValueError(f"Unknown kernel_type '{kernel_type}'. Choose 'gaussian' or 'motion'.")

        # Depthwise convolution weight: (C, 1, K, K)
        weight = torch.from_numpy(k2d)[None, None, :, :].repeat(channels, 1, 1, 1).to(device)
        self.register_buffer("weight", weight)

    @staticmethod
    def _make_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
        ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * max(sigma, 1e-3) ** 2))
        kernel /= np.sum(kernel)
        return kernel.astype(np.float32)

    @staticmethod
    def _make_motion_kernel(size: int, angle_deg: float) -> np.ndarray:
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        for t in np.linspace(-center, center, size * 4):
            x = int(round(center + t * dx))
            y = int(round(center + t * dy))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        total = np.sum(kernel)
        if total == 0:
            kernel[center, center] = 1.0
        else:
            kernel /= total
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="reflect")
        return F.conv2d(x_pad, self.weight, groups=x.shape[1])


class SuperResolutionOperator(ForwardOperator):
    """Anti-aliased differentiable downsampling operator A(x) = Downsample_s(x)."""

    name = "super_resolution"

    def __init__(self, scale_factor: int = 4) -> None:
        super().__init__()
        if scale_factor < 2:
            raise ValueError("scale_factor must be >= 2.")
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            scale_factor=1.0 / self.scale_factor,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )


class RadonTransformOperator(ForwardOperator):
    """Differentiable 2D parallel-beam Radon transform for X-Ray Tomography (CT).

    Rotates the input image across `num_angles` uniformly spaced projection angles
    in [0, pi) using bilinear `grid_sample` and integrates along detector rays,
    producing a sinogram of shape (B, C, num_angles, W).
    """

    name = "tomography"

    def __init__(
        self,
        num_angles: int = 30,
        image_size: int = 128,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.num_angles = num_angles
        self.image_size = image_size

        angles = torch.linspace(0.0, math.pi, num_angles + 1, dtype=torch.float32)[:-1]
        cos_a = torch.cos(angles)
        sin_a = torch.sin(angles)

        # Affine rotation matrices: shape (num_angles, 2, 3)
        theta = torch.zeros((num_angles, 2, 3), dtype=torch.float32)
        theta[:, 0, 0] = cos_a
        theta[:, 0, 1] = -sin_a
        theta[:, 1, 0] = sin_a
        theta[:, 1, 1] = cos_a

        grid = F.affine_grid(theta, size=(num_angles, 1, image_size, image_size), align_corners=False)
        self.register_buffer("grids", grid.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != self.image_size or w != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        # Expand x across all projection angles: (B * num_angles, C, H, W)
        x_rep = x.unsqueeze(1).expand(b, self.num_angles, c, self.image_size, self.image_size)
        x_rep = x_rep.reshape(b * self.num_angles, c, self.image_size, self.image_size)

        grids = self.grids.unsqueeze(0).expand(b, -1, -1, -1, -1).reshape(
            b * self.num_angles, self.image_size, self.image_size, 2
        )
        rotated = F.grid_sample(x_rep, grids, mode="bilinear", padding_mode="zeros", align_corners=False)
        # Integrate along the ray dimension (dim=-2) and normalize by image_size
        sinogram = rotated.mean(dim=-2)  # (B * num_angles, C, W)
        sinogram = sinogram.view(b, self.num_angles, c, self.image_size).permute(0, 2, 1, 3)
        return sinogram

    def visualize_measurement(self, y: torch.Tensor) -> np.ndarray:
        """Normalize sinogram for visual display."""
        sino = y.detach().cpu().squeeze(0).numpy()
        if sino.ndim == 3:
            sino = sino.mean(axis=0)
        smin, smax = float(sino.min()), float(sino.max())
        if smax - smin < 1e-8:
            return np.zeros_like(sino)
        return (sino - smin) / (smax - smin)


class CompressedSensingOperator(ForwardOperator):
    """Compressed sensing forward operator A(x) = Phi x.

    Supports two modes:
    - 'random_matrix': Block-wise random Gaussian sensing matrix Phi in R^{m x n}
      with m = round(measurement_ratio * n) applied to non-overlapping B x B patches.
    - 'fourier': Subsampled 2D orthonormal FFT measurements (as in compressed-sensing MRI).
    """

    name = "compressed_sensing"

    def __init__(
        self,
        image_shape: tuple[int, int, int, int],
        measurement_ratio: float = 0.25,
        mode: str = "random_matrix",
        block_size: int = 16,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.measurement_ratio = float(np.clip(measurement_ratio, 0.05, 0.95))
        self.block_size = block_size
        _, c, h, w = image_shape

        gen = torch.Generator(device="cpu").manual_seed(seed)

        if self.mode == "random_matrix":
            if h % block_size != 0 or w % block_size != 0:
                raise ValueError(
                    f"Image spatial size ({h}, {w}) must be divisible by block_size ({block_size})."
                )
            n = block_size * block_size
            m = max(1, int(round(self.measurement_ratio * n)))
            phi = torch.empty((m, n), dtype=torch.float32).normal_(0.0, 1.0 / math.sqrt(m), generator=gen)
            self.register_buffer("phi", phi.to(device))
            self.num_measurements = m
            self.patch_dim = n
        elif self.mode == "fourier":
            # Random k-space mask with guaranteed DC component
            mask = (torch.rand((h, w), generator=gen) < self.measurement_ratio).float()
            mask[0, 0] = 1.0
            self.register_buffer("kspace_mask", mask[None, None, :, :].to(device))
        else:
            raise ValueError(f"Unknown compressed sensing mode '{mode}'. Choose 'random_matrix' or 'fourier'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "random_matrix":
            b, c, h, w = x.shape
            bs = self.block_size
            nh, nw = h // bs, w // bs
            # Rearrange into patches: (B, C, nh, nw, bs * bs)
            patches = x.view(b, c, nh, bs, nw, bs).permute(0, 1, 2, 4, 3, 5).contiguous()
            patches = patches.view(b, c, nh * nw, bs * bs)
            # Multiply by Phi^T -> (B, C, num_patches, m)
            return torch.matmul(patches, self.phi.t())

        # Fourier mode: return stacked real & imag parts of masked 2D FFT
        fft_x = torch.fft.fft2(x, norm="ortho")
        masked = fft_x * self.kspace_mask
        return torch.stack([masked.real, masked.imag], dim=-1)

    def visualize_measurement(self, y: torch.Tensor) -> np.ndarray:
        """Visualize the adjoint (zero-filled / back-projected) reconstruction A^T(y)."""
        with torch.no_grad():
            if self.mode == "random_matrix":
                b, c, num_patches, _ = y.shape
                side = int(round(math.sqrt(num_patches)))
                bs = self.block_size
                # Adjoint Phi^T y -> (B, C, num_patches, bs*bs)
                bp = torch.matmul(y, self.phi)
                bp = bp.view(b, c, side, side, bs, bs).permute(0, 1, 2, 4, 3, 5).contiguous()
                bp = bp.view(b, c, side * bs, side * bs)
                img = bp.squeeze(0).cpu().numpy()
            else:
                complex_y = torch.complex(y[..., 0], y[..., 1])
                zf = torch.fft.ifft2(complex_y, norm="ortho").real
                img = zf.squeeze(0).cpu().numpy()

        if img.ndim == 3:
            if img.shape[0] == 1:
                img = img[0]
            else:
                img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0.0, 1.0)
