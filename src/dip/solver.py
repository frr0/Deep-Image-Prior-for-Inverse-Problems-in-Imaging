"""Deep Image Prior optimization engine."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from .inputs import create_input_tensor
from .metrics import compute_psnr, compute_ssim, total_variation_loss
from .models import build_model
from .operators import ForwardOperator


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve compute device ('auto', 'cpu', 'cuda')."""
    env_override = os.getenv("DIP_DEVICE")
    choice = (requested if requested != "auto" else (env_override or "auto")).lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


@dataclass
class DIPConfig:
    """Configuration for a Deep Image Prior optimization run."""

    arch: str = "skipnet"
    input_type: str = "uniform"
    input_depth: int = 32
    base_channels: int = 64
    num_scales: int = 4
    skip_channels: int = 4
    upsample_mode: str = "bilinear"
    num_iter: int = 600
    lr: float = 0.01
    reg_noise_std: float = 0.03
    exp_weight: float = 0.99
    tv_weight: float = 0.0
    loss_type: str = "mse"
    log_every: int = 20
    seed: int = 42
    device: str = "auto"


@dataclass
class DIPHistory:
    """Per-checkpoint metrics collected during DIP optimization."""

    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    psnr_raw: list[float] = field(default_factory=list)
    psnr_ema: list[float] = field(default_factory=list)
    ssim_ema: list[float] = field(default_factory=list)


@dataclass
class DIPResult:
    """Final output and diagnostics of a DIP optimization run."""

    reconstruction: torch.Tensor
    reconstruction_raw: torch.Tensor
    measurement: torch.Tensor
    history: DIPHistory
    elapsed_seconds: float
    device: str
    num_parameters: int


ProgressCallback = Callable[[int, int, float, float | None, torch.Tensor], None]


class DeepImagePriorSolver:
    """Solves min_theta || A(f_theta(z)) - y || + lambda_TV * TV(f_theta(z))."""

    def __init__(self, config: DIPConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)

    def solve(
        self,
        measurement: torch.Tensor,
        operator: ForwardOperator,
        target_shape: tuple[int, int, int, int],
        ground_truth: torch.Tensor | None = None,
        callback: ProgressCallback | None = None,
    ) -> DIPResult:
        """Run Deep Image Prior optimization from scratch (zero training data)."""
        cfg = self.config
        torch.manual_seed(cfg.seed)

        _, out_channels, h, w = target_shape
        measurement = measurement.to(self.device)
        operator = operator.to(self.device)
        gt = ground_truth.to(self.device) if ground_truth is not None else None

        z_fixed = create_input_tensor(
            input_type=cfg.input_type,
            spatial_size=(h, w),
            input_depth=cfg.input_depth,
            device=self.device,
            seed=cfg.seed,
        )
        in_channels = z_fixed.shape[1]

        model = build_model(
            arch=cfg.arch,
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            base_channels=cfg.base_channels,
            num_scales=cfg.num_scales,
            skip_channels=cfg.skip_channels,
            upsample_mode=cfg.upsample_mode,
        ).to(self.device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        criterion: nn.Module = nn.L1Loss() if cfg.loss_type.lower() == "l1" else nn.MSELoss()

        out_ema: torch.Tensor | None = None
        history = DIPHistory()
        noise_buf = torch.empty_like(z_fixed)

        t0 = time.perf_counter()
        model.train()

        for step in range(1, cfg.num_iter + 1):
            optimizer.zero_grad(set_to_none=True)

            if cfg.reg_noise_std > 0.0:
                noise_buf.normal_(0.0, cfg.reg_noise_std)
                net_input = z_fixed + noise_buf
            else:
                net_input = z_fixed

            out = model(net_input)
            pred_measurement = operator(out)
            loss = criterion(pred_measurement, measurement)

            if cfg.tv_weight > 0.0:
                loss = loss + cfg.tv_weight * total_variation_loss(out)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                out_detached = out.detach()
                if out_ema is None or cfg.exp_weight <= 0.0:
                    out_ema = out_detached.clone()
                else:
                    # Adaptive warmup from cumulative average to exponential moving average
                    # so early random initializations do not bias short runs.
                    alpha = max(1.0 - cfg.exp_weight, 1.0 / step)
                    out_ema.mul_(1.0 - alpha).add_(out_detached, alpha=alpha)

            is_log_step = (step % cfg.log_every == 0) or (step == 1) or (step == cfg.num_iter)
            if is_log_step:
                loss_val = float(loss.item())
                history.steps.append(step)
                history.losses.append(loss_val)

                psnr_ema_val: float | None = None
                if gt is not None:
                    psnr_raw_val = compute_psnr(out_detached, gt)
                    psnr_ema_val = compute_psnr(out_ema, gt)
                    ssim_ema_val = compute_ssim(out_ema, gt)
                    history.psnr_raw.append(psnr_raw_val)
                    history.psnr_ema.append(psnr_ema_val)
                    history.ssim_ema.append(ssim_ema_val)

                if callback is not None:
                    callback(step, cfg.num_iter, loss_val, psnr_ema_val, out_ema.detach().cpu())

        elapsed = time.perf_counter() - t0
        assert out_ema is not None

        return DIPResult(
            reconstruction=out_ema.detach().cpu(),
            reconstruction_raw=out_detached.detach().cpu(),
            measurement=measurement.detach().cpu(),
            history=history,
            elapsed_seconds=elapsed,
            device=str(self.device),
            num_parameters=num_params,
        )
