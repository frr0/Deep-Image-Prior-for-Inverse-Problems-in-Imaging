"""CLI runner for reproducible Deep Image Prior inverse-problem experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dip import (
    CompressedSensingOperator,
    DeblurringOperator,
    DeepImagePriorSolver,
    DIPConfig,
    InpaintingOperator,
    RadonTransformOperator,
    SuperResolutionOperator,
    DenoisingOperator,
    compute_psnr,
    compute_ssim,
    load_sample_image,
    tensor_to_numpy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Deep Image Prior inverse problem experiment.")
    parser.add_argument(
        "--problem",
        type=str,
        default="denoising",
        choices=["denoising", "inpainting", "deblurring", "super_resolution", "tomography", "compressed_sensing"],
    )
    parser.add_argument("--image", type=str, default="astronaut", choices=["astronaut", "chelsea", "camera", "coffee", "shepp_logan"])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--arch", type=str, default="skipnet", choices=["skipnet", "unet"])
    parser.add_argument("--input-type", type=str, default="uniform", choices=["uniform", "normal", "meshgrid"])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-scales", type=int, default=4)
    parser.add_argument("--reg-noise-std", type=float, default=0.03)
    parser.add_argument("--exp-weight", type=float, default=0.98)
    parser.add_argument("--tv-weight", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--mask-strategy", type=str, default="random", choices=["random", "block", "stripes"])
    parser.add_argument("--blur-kernel", type=str, default="gaussian", choices=["gaussian", "motion"])
    parser.add_argument("--scale-factor", type=int, default=4)
    parser.add_argument("--num-angles", type=int, default=30)
    parser.add_argument("--cs-ratio", type=float, default=0.3)
    parser.add_argument("--cs-mode", type=str, default="random_matrix", choices=["random_matrix", "fourier"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grayscale = args.grayscale or (args.problem == "tomography")
    gt = load_sample_image(args.image, size=args.size, grayscale=grayscale)

    if args.problem == "denoising":
        op = DenoisingOperator()
        y = op.corrupt(gt, noise_std=args.noise_std, seed=args.seed, clamp_output=True)
    elif args.problem == "inpainting":
        op = InpaintingOperator.from_strategy(tuple(gt.shape), strategy=args.mask_strategy, keep_ratio=args.keep_ratio, seed=args.seed)
        y = op.corrupt(gt, noise_std=0.0, seed=args.seed, clamp_output=True)
    elif args.problem == "deblurring":
        op = DeblurringOperator(kernel_type=args.blur_kernel, kernel_size=9, sigma=1.8, channels=gt.shape[1])
        y = op.corrupt(gt, noise_std=args.noise_std, seed=args.seed, clamp_output=True)
    elif args.problem == "super_resolution":
        op = SuperResolutionOperator(scale_factor=args.scale_factor)
        y = op.corrupt(gt, noise_std=0.0, seed=args.seed, clamp_output=True)
    elif args.problem == "tomography":
        op = RadonTransformOperator(num_angles=args.num_angles, image_size=args.size)
        y = op.corrupt(gt, noise_std=0.002, seed=args.seed, clamp_output=False)
    else:
        op = CompressedSensingOperator(tuple(gt.shape), measurement_ratio=args.cs_ratio, mode=args.cs_mode, seed=args.seed)
        y = op.corrupt(gt, noise_std=0.0, seed=args.seed, clamp_output=False)

    cfg = DIPConfig(
        arch=args.arch,
        input_type=args.input_type,
        base_channels=args.base_channels,
        num_scales=args.num_scales,
        num_iter=args.iterations,
        lr=args.lr,
        reg_noise_std=args.reg_noise_std,
        exp_weight=args.exp_weight,
        tv_weight=args.tv_weight,
        log_every=max(10, args.iterations // 20),
        seed=args.seed,
    )

    def on_step(step: int, total: int, loss_val: float, psnr_val: float | None, _img) -> None:
        psnr_str = f" | PSNR={psnr_val:.2f} dB" if psnr_val is not None else ""
        print(f"[{step:4d}/{total}] loss={loss_val:.6f}{psnr_str}", flush=True)

    solver = DeepImagePriorSolver(cfg)
    result = solver.solve(y, op, target_shape=tuple(gt.shape), ground_truth=gt, callback=on_step)

    final_psnr = compute_psnr(result.reconstruction, gt)
    final_ssim = compute_ssim(result.reconstruction, gt)
    baseline_psnr = compute_psnr(y, gt) if y.shape == gt.shape else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.problem}_{args.image}_{args.arch}"

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    cmap = "gray" if grayscale else None
    axes[0].imshow(tensor_to_numpy(gt), cmap=cmap)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    meas_vis = op.visualize_measurement(y)
    axes[1].imshow(meas_vis, cmap="gray" if meas_vis.ndim == 2 else None)
    axes[1].set_title(f"Measurement ({args.problem})")
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(result.reconstruction), cmap=cmap)
    axes[2].set_title(f"DIP ({final_psnr:.2f} dB / SSIM {final_ssim:.3f})")
    axes[2].axis("off")

    axes[3].plot(result.history.steps, result.history.psnr_ema, label="EMA PSNR", lw=2)
    axes[3].plot(result.history.steps, result.history.psnr_raw, label="Raw PSNR", alpha=0.5)
    if baseline_psnr is not None:
        axes[3].axhline(baseline_psnr, linestyle="--", color="red", label="Baseline")
    axes[3].set_title("PSNR Progression")
    axes[3].set_xlabel("Iteration")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / f"{stem}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    summary = {
        "problem": args.problem,
        "image": args.image,
        "arch": args.arch,
        "input_type": args.input_type,
        "iterations": args.iterations,
        "baseline_psnr": baseline_psnr,
        "final_psnr": final_psnr,
        "final_ssim": final_ssim,
        "elapsed_seconds": result.elapsed_seconds,
        "device": result.device,
        "num_parameters": result.num_parameters,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved figure to {fig_path} and metrics to {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
