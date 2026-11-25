"""Unit and integration tests for Deep Image Prior."""

from __future__ import annotations

import pytest
import torch

from dip import (
    CompressedSensingOperator,
    DeblurringOperator,
    DeepImagePriorSolver,
    DenoisingOperator,
    DIPConfig,
    InpaintingOperator,
    RadonTransformOperator,
    SuperResolutionOperator,
    build_model,
    compute_psnr,
    compute_ssim,
    create_input_tensor,
    load_sample_image,
)


@pytest.mark.parametrize("arch", ["skipnet", "unet"])
@pytest.mark.parametrize("input_type", ["uniform", "normal", "meshgrid"])
def test_models_and_inputs_forward(arch: str, input_type: str) -> None:
    z = create_input_tensor(input_type=input_type, spatial_size=(64, 64), input_depth=16, seed=0)
    model = build_model(arch=arch, num_input_channels=z.shape[1], num_output_channels=3, base_channels=16, num_scales=3)
    out = model(z).detach()
    assert out.shape == (1, 3, 64, 64)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_all_forward_operators_differentiable() -> None:
    x = torch.rand((1, 3, 64, 64), requires_grad=True)

    operators = [
        DenoisingOperator(),
        InpaintingOperator.from_strategy((1, 3, 64, 64), strategy="random", keep_ratio=0.5),
        InpaintingOperator.from_strategy((1, 3, 64, 64), strategy="block", keep_ratio=0.7),
        InpaintingOperator.from_strategy((1, 3, 64, 64), strategy="stripes", keep_ratio=0.6),
        DeblurringOperator(kernel_type="gaussian", kernel_size=7, sigma=1.5, channels=3),
        DeblurringOperator(kernel_type="motion", kernel_size=7, motion_angle_deg=45.0, channels=3),
        SuperResolutionOperator(scale_factor=4),
        RadonTransformOperator(num_angles=18, image_size=64),
        CompressedSensingOperator((1, 3, 64, 64), measurement_ratio=0.3, mode="random_matrix", block_size=16),
        CompressedSensingOperator((1, 3, 64, 64), measurement_ratio=0.3, mode="fourier"),
    ]

    for op in operators:
        x_in = x.detach().clone().requires_grad_(True)
        y = op(x_in)
        loss = (y**2).mean()
        loss.backward()
        assert x_in.grad is not None, f"No gradient for {op.name}"
        assert torch.isfinite(x_in.grad).all(), f"Non-finite gradient for {op.name}"
        vis = op.visualize_measurement(y.detach())
        assert vis.ndim in (2, 3)


def test_dip_denoising_improves_psnr() -> None:
    gt = load_sample_image("camera", size=64, grayscale=True)
    op = DenoisingOperator()
    noisy = op.corrupt(gt, noise_std=0.15, seed=7, clamp_output=True)
    baseline_psnr = compute_psnr(noisy, gt)

    cfg = DIPConfig(
        arch="skipnet",
        base_channels=24,
        num_scales=3,
        num_iter=120,
        lr=0.01,
        reg_noise_std=0.03,
        exp_weight=0.95,
        log_every=30,
        seed=7,
        device="cpu",
    )
    result = DeepImagePriorSolver(cfg).solve(noisy, op, target_shape=tuple(gt.shape), ground_truth=gt)
    final_psnr = compute_psnr(result.reconstruction, gt)
    final_ssim = compute_ssim(result.reconstruction, gt)

    assert final_psnr > baseline_psnr + 2.0, f"Expected DIP to improve PSNR: {final_psnr:.2f} vs {baseline_psnr:.2f}"
    assert final_ssim > 0.5


def test_dip_tomography_end_to_end() -> None:
    gt = load_sample_image("shepp_logan", size=64, grayscale=True)
    op = RadonTransformOperator(num_angles=24, image_size=64)
    sinogram = op.corrupt(gt, noise_std=0.0, seed=1)

    cfg = DIPConfig(
        arch="skipnet",
        base_channels=24,
        num_scales=3,
        num_iter=150,
        lr=0.01,
        reg_noise_std=0.01,
        exp_weight=0.95,
        tv_weight=1e-3,
        log_every=30,
        seed=1,
        device="cpu",
    )
    result = DeepImagePriorSolver(cfg).solve(sinogram, op, target_shape=tuple(gt.shape), ground_truth=gt)
    assert result.history.losses[-1] < result.history.losses[0]
    assert compute_psnr(result.reconstruction, gt) > 15.0
