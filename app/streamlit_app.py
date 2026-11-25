"""Interactive Streamlit dashboard for Deep Image Prior on Inverse Problems in Imaging."""

from __future__ import annotations

import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from dip import (
    CompressedSensingOperator,
    DeblurringOperator,
    DeepImagePriorSolver,
    DenoisingOperator,
    DIPConfig,
    InpaintingOperator,
    RadonTransformOperator,
    SuperResolutionOperator,
    compute_psnr,
    compute_ssim,
    load_image_from_bytes,
    load_sample_image,
    tensor_to_numpy,
)

st.set_page_config(
    page_title="Deep Image Prior — Inverse Problems",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Deep Image Prior for Inverse Problems in Imaging")
st.markdown(
    """
Solve ill-posed imaging inverse problems **without any training data or pre-training**
by optimizing the weights of a convolutional generator $f_\\theta(z)$ directly on a single corrupted measurement:
$$\\theta^* = \\arg\\min_\\theta \\; \\mathcal{L}\\big(A(f_\\theta(z)),\\; y\\big) + \\lambda_{\\mathrm{TV}} \\mathrm{TV}(f_\\theta(z))$$
"""
)


def _to_png_bytes(img_float: np.ndarray) -> bytes:
    arr = (np.clip(img_float, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _build_operator(
    problem: str,
    gt: torch.Tensor,
    params: dict[str, Any],
    seed: int,
):
    if problem == "Denoising":
        op = DenoisingOperator()
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=True)
        return op, y, "Noisy Observation"

    if problem == "Inpainting":
        op = InpaintingOperator.from_strategy(
            shape=tuple(gt.shape),
            strategy=params["mask_strategy"],
            keep_ratio=params["keep_ratio"],
            seed=seed,
        )
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=True)
        return op, y, f"Masked Observation ({params['mask_strategy']})"

    if problem == "Deblurring (+ Denoising)":
        op = DeblurringOperator(
            kernel_type=params["kernel_type"],
            kernel_size=params["kernel_size"],
            sigma=params["blur_sigma"],
            motion_angle_deg=params["motion_angle"],
            channels=gt.shape[1],
        )
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=True)
        return op, y, "Blurred (+ Noisy) Observation"

    if problem == "Super-Resolution":
        op = SuperResolutionOperator(scale_factor=params["scale_factor"])
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=True)
        return op, y, f"Low-Res Observation (1/{params['scale_factor']}x)"

    if problem == "X-Ray Tomography (Radon)":
        op = RadonTransformOperator(num_angles=params["num_angles"], image_size=gt.shape[-1])
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=False)
        return op, y, f"Sinogram ({params['num_angles']} angles)"

    if problem == "Compressed Sensing":
        op = CompressedSensingOperator(
            image_shape=tuple(gt.shape),
            measurement_ratio=params["cs_ratio"],
            mode=params["cs_mode"],
            block_size=16,
            seed=seed,
        )
        y = op.corrupt(gt, noise_std=params["noise_std"], seed=seed, clamp_output=False)
        label = (
            f"Adjoint Back-Projection ({int(params['cs_ratio'] * 100)}% meas.)"
            if params["cs_mode"] == "random_matrix"
            else f"Zero-Filled IFFT ({int(params['cs_ratio'] * 100)}% k-space)"
        )
        return op, y, label

    raise ValueError(f"Unsupported problem: {problem}")


with st.sidebar:
    st.header("1. Inverse Problem")
    problem = st.selectbox(
        "Task",
        [
            "Denoising",
            "Inpainting",
            "Deblurring (+ Denoising)",
            "Super-Resolution",
            "X-Ray Tomography (Radon)",
            "Compressed Sensing",
        ],
    )

    st.header("2. Input Image")
    image_source = st.radio("Source", ["Built-in Sample", "Upload Image"], horizontal=True)
    default_sample = "shepp_logan" if problem == "X-Ray Tomography (Radon)" else "astronaut"
    sample_options = ["astronaut", "chelsea", "camera", "coffee", "shepp_logan"]
    sample_name = st.selectbox(
        "Sample",
        sample_options,
        index=sample_options.index(default_sample),
        disabled=(image_source != "Built-in Sample"),
    )
    uploaded = st.file_uploader("Upload PNG/JPG", type=["png", "jpg", "jpeg"]) if image_source == "Upload Image" else None

    img_size = st.select_slider("Resolution (px)", options=[64, 96, 128, 192, 256], value=128)
    force_gray = problem == "X-Ray Tomography (Radon)"
    grayscale = st.checkbox("Grayscale", value=force_gray, disabled=force_gray)

    st.header("3. Forward Operator")
    op_params: dict[str, Any] = {"noise_std": 0.0}
    if problem == "Denoising":
        op_params["noise_std"] = st.slider("Gaussian Noise σ", 0.02, 0.40, 0.12, 0.01)
    elif problem == "Inpainting":
        op_params["mask_strategy"] = st.selectbox("Mask Type", ["random", "block", "stripes"])
        op_params["keep_ratio"] = st.slider("Observed Pixel Ratio", 0.15, 0.85, 0.50, 0.05)
        op_params["noise_std"] = st.slider("Additive Noise σ", 0.0, 0.10, 0.0, 0.01)
    elif problem == "Deblurring (+ Denoising)":
        op_params["kernel_type"] = st.selectbox("Blur Kernel", ["gaussian", "motion"])
        op_params["kernel_size"] = st.select_slider("Kernel Size", options=[5, 7, 9, 11, 15], value=9)
        op_params["blur_sigma"] = st.slider("Gaussian σ", 0.5, 4.0, 1.8, 0.1)
        op_params["motion_angle"] = st.slider("Motion Angle (°)", 0, 180, 30, 5)
        op_params["noise_std"] = st.slider("Additive Noise σ", 0.0, 0.10, 0.02, 0.005)
    elif problem == "Super-Resolution":
        op_params["scale_factor"] = st.select_slider("Downsampling Factor", options=[2, 4], value=4)
        op_params["noise_std"] = st.slider("Additive Noise σ", 0.0, 0.05, 0.0, 0.005)
    elif problem == "X-Ray Tomography (Radon)":
        op_params["num_angles"] = st.slider("Projection Angles", 12, 90, 30, 6)
        op_params["noise_std"] = st.slider("Sinogram Noise σ", 0.0, 0.03, 0.002, 0.001)
    elif problem == "Compressed Sensing":
        op_params["cs_mode"] = st.selectbox("Sensing Model", ["random_matrix", "fourier"])
        op_params["cs_ratio"] = st.slider("Measurement Ratio (m/n)", 0.10, 0.70, 0.30, 0.05)
        op_params["noise_std"] = st.slider("Measurement Noise σ", 0.0, 0.05, 0.0, 0.005)

    st.header("4. Network & Optimization")
    arch = st.selectbox("Architecture", ["skipnet", "unet"])
    input_type = st.selectbox("Fixed Input z", ["uniform", "normal", "meshgrid"])
    base_channels = st.select_slider("Base Channels", options=[32, 64, 96, 128], value=64)
    num_scales = st.select_slider("Network Depth (Scales)", options=[3, 4, 5], value=4)
    num_iter = st.slider("Iterations", 100, 3000, 500, 50)
    lr = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.02], value=0.01)
    reg_noise_std = st.slider("Input Noise Perturbation (reg_noise_std)", 0.0, 0.10, 0.03, 0.005)
    exp_weight = st.slider("EMA Smoothing Weight", 0.0, 0.995, 0.98, 0.005)
    tv_weight = st.select_slider("TV Regularization Weight", options=[0.0, 1e-4, 5e-4, 1e-3, 5e-3], value=0.0)
    seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)

# Load ground truth image
if image_source == "Upload Image" and uploaded is not None:
    gt = load_image_from_bytes(uploaded.getvalue(), size=img_size, grayscale=grayscale)
else:
    gt = load_sample_image(sample_name, size=img_size, grayscale=grayscale)

operator, measurement, meas_label = _build_operator(problem, gt, op_params, seed=int(seed))
gt_np = tensor_to_numpy(gt)
meas_vis = operator.visualize_measurement(measurement)

# Baseline PSNR when measurement lives in image space with same shape
baseline_psnr = None
if measurement.shape == gt.shape:
    baseline_psnr = compute_psnr(measurement, gt)

preview_cols = st.columns(3)
with preview_cols[0]:
    st.subheader("Ground Truth")
    st.image(gt_np, clamp=True, use_container_width=True)
with preview_cols[1]:
    st.subheader(meas_label)
    st.image(meas_vis, clamp=True, use_container_width=True)
    if baseline_psnr is not None:
        st.caption(f"Corrupted PSNR: **{baseline_psnr:.2f} dB**")
with preview_cols[2]:
    st.subheader("DIP Reconstruction")
    recon_placeholder = st.empty()
    recon_placeholder.info("Click **Run Deep Image Prior** below to start optimization.")

run_btn = st.button("🚀 Run Deep Image Prior", type="primary", use_container_width=True)

if run_btn:
    cfg = DIPConfig(
        arch=arch,
        input_type=input_type,
        base_channels=base_channels,
        num_scales=num_scales,
        num_iter=num_iter,
        lr=lr,
        reg_noise_std=reg_noise_std,
        exp_weight=exp_weight,
        tv_weight=tv_weight,
        log_every=max(10, num_iter // 25),
        seed=int(seed),
    )

    progress_bar = st.progress(0.0, text="Initializing network...")
    status_box = st.empty()

    def on_step(step: int, total: int, loss_val: float, psnr_val: float | None, img_t: torch.Tensor) -> None:
        frac = step / total
        psnr_str = f" | PSNR: {psnr_val:.2f} dB" if psnr_val is not None else ""
        progress_bar.progress(frac, text=f"Iteration {step}/{total} — Loss: {loss_val:.5f}{psnr_str}")
        recon_placeholder.image(tensor_to_numpy(img_t), clamp=True, use_container_width=True)

    solver = DeepImagePriorSolver(cfg)
    result = solver.solve(
        measurement=measurement,
        operator=operator,
        target_shape=tuple(gt.shape),
        ground_truth=gt,
        callback=on_step,
    )

    recon_np = tensor_to_numpy(result.reconstruction)
    final_psnr = compute_psnr(result.reconstruction, gt)
    final_ssim = compute_ssim(result.reconstruction, gt)

    progress_bar.progress(1.0, text=f"Completed in {result.elapsed_seconds:.1f}s on {result.device.upper()}")
    recon_placeholder.image(recon_np, clamp=True, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    delta_str = f"{final_psnr - baseline_psnr:+.2f} dB" if baseline_psnr is not None else None
    m1.metric("Final PSNR (EMA)", f"{final_psnr:.2f} dB", delta=delta_str)
    m2.metric("Final SSIM (EMA)", f"{final_ssim:.4f}")
    m3.metric("Elapsed Time", f"{result.elapsed_seconds:.1f} s")
    m4.metric("Trainable Params", f"{result.num_parameters:,}")

    # Plots: Loss, PSNR, and Absolute Residual
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(result.history.steps, result.history.losses, color="#2563eb", lw=2)
    axes[0].set_title("Data Fidelity Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result.history.steps, result.history.psnr_raw, label="Raw output", alpha=0.55, color="#f59e0b")
    axes[1].plot(result.history.steps, result.history.psnr_ema, label="EMA output", lw=2, color="#10b981")
    if baseline_psnr is not None:
        axes[1].axhline(baseline_psnr, linestyle="--", color="#ef4444", label="Corrupted input")
    axes[1].set_title("PSNR (dB) over Iterations")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    residual = np.abs(gt_np - recon_np)
    if residual.ndim == 3:
        residual = residual.mean(axis=-1)
    im = axes[2].imshow(residual, cmap="magma")
    axes[2].set_title("Absolute Residual |x - x_hat|")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    st.pyplot(fig)

    st.download_button(
        label="⬇️ Download Reconstructed Image (PNG)",
        data=_to_png_bytes(recon_np),
        file_name=f"dip_{operator.name}_{arch}.png",
        mime="image/png",
    )
