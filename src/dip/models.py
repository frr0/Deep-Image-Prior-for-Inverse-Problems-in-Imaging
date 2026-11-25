"""Neural network architectures for Deep Image Prior.

Implements:
- SkipNet: The exact encoder-decoder hourglass architecture with skip connections
  from Ulyanov et al., "Deep Image Prior" (CVPR 2018).
- UNet: Classic symmetric U-Net architecture for comparative experiments.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    bias: bool = True,
    pad: str = "reflection",
) -> nn.Sequential:
    pad_amount = (kernel_size - 1) // 2
    layers: list[nn.Module] = []
    if pad_amount > 0:
        if pad == "reflection":
            layers.append(nn.ReflectionPad2d(pad_amount))
        elif pad == "zero":
            layers.append(nn.ZeroPad2d(pad_amount))
        else:
            raise ValueError(f"Unsupported pad mode: {pad}")

    layers.append(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )
    )
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _match_spatial_size(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Resize `src` to match the HxW spatial dimensions of `target` if needed."""
    if src.shape[-2:] != target.shape[-2:]:
        src = F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return src


class SkipNet(nn.Module):
    """Hourglass encoder-decoder with skip connections (Ulyanov et al., CVPR 2018).

    Each scale downsamples via strided convolution, optionally passes features
    through a narrow 1x1 skip connection, and upsamples via bilinear/nearest
    interpolation followed by convolutions.
    """

    def __init__(
        self,
        num_input_channels: int = 32,
        num_output_channels: int = 3,
        num_channels_down: Sequence[int] = (128, 128, 128, 128, 128),
        num_channels_up: Sequence[int] = (128, 128, 128, 128, 128),
        num_channels_skip: Sequence[int] = (4, 4, 4, 4, 4),
        filter_size_down: int = 3,
        filter_size_up: int = 3,
        filter_size_skip: int = 1,
        upsample_mode: str = "bilinear",
        pad: str = "reflection",
        need_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if not (len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)):
            raise ValueError("num_channels_down, num_channels_up, and num_channels_skip must have equal length.")

        self.num_scales = len(num_channels_down)
        self.upsample_mode = upsample_mode
        self.need_sigmoid = need_sigmoid

        self.down_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_ch = num_input_channels
        for i in range(self.num_scales):
            down_ch = num_channels_down[i]
            skip_ch = num_channels_skip[i]

            down_block = nn.Sequential(
                _conv_block(in_ch, down_ch, kernel_size=filter_size_down, stride=2, pad=pad),
                _conv_block(down_ch, down_ch, kernel_size=filter_size_down, stride=1, pad=pad),
            )
            self.down_blocks.append(down_block)

            if skip_ch > 0:
                self.skip_blocks.append(
                    _conv_block(in_ch, skip_ch, kernel_size=filter_size_skip, stride=1, pad=pad)
                )
            else:
                self.skip_blocks.append(nn.Identity())

            in_ch = down_ch

        for i in reversed(range(self.num_scales)):
            up_ch = num_channels_up[i]
            skip_ch = num_channels_skip[i]
            deeper_ch = num_channels_down[i] if i == self.num_scales - 1 else num_channels_up[i + 1]

            concat_ch = deeper_ch + skip_ch
            up_block = nn.Sequential(
                nn.BatchNorm2d(concat_ch),
                _conv_block(concat_ch, up_ch, kernel_size=filter_size_up, stride=1, pad=pad),
                _conv_block(up_ch, up_ch, kernel_size=1, stride=1, pad=pad),
            )
            self.up_blocks.append(up_block)

        self.final_conv = nn.Conv2d(num_channels_up[0], num_output_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor | None] = []
        spatial_sizes: list[tuple[int, int]] = []
        h = x

        for down_block, skip_block in zip(self.down_blocks, self.skip_blocks):
            spatial_sizes.append((h.shape[-2], h.shape[-1]))
            if isinstance(skip_block, nn.Identity):
                skips.append(None)
            else:
                skips.append(skip_block(h))
            h = down_block(h)

        for idx, up_block in enumerate(self.up_blocks):
            scale_idx = self.num_scales - 1 - idx
            target_size = spatial_sizes[scale_idx]
            if self.upsample_mode == "bilinear":
                h = F.interpolate(h, size=target_size, mode="bilinear", align_corners=False)
            else:
                h = F.interpolate(h, size=target_size, mode="nearest")

            skip = skips[scale_idx]
            if skip is not None:
                h = _match_spatial_size(h, skip)
                h = torch.cat([skip, h], dim=1)

            h = up_block(h)

        out = self.final_conv(h)
        if self.need_sigmoid:
            out = torch.sigmoid(out)
        return out


class UNet(nn.Module):
    """Classic symmetric U-Net architecture adapted for Deep Image Prior."""

    def __init__(
        self,
        num_input_channels: int = 32,
        num_output_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        need_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.need_sigmoid = need_sigmoid

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = num_input_channels
        enc_channels: list[int] = []

        for d in range(depth):
            out_ch = base_channels * (2**d)
            self.encoders.append(
                nn.Sequential(
                    _conv_block(in_ch, out_ch, kernel_size=3, pad="reflection"),
                    _conv_block(out_ch, out_ch, kernel_size=3, pad="reflection"),
                )
            )
            self.pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
            enc_channels.append(out_ch)
            in_ch = out_ch

        bottleneck_ch = base_channels * (2**depth)
        self.bottleneck = nn.Sequential(
            _conv_block(in_ch, bottleneck_ch, kernel_size=3, pad="reflection"),
            _conv_block(bottleneck_ch, bottleneck_ch, kernel_size=3, pad="reflection"),
        )

        self.decoders = nn.ModuleList()
        prev_ch = bottleneck_ch
        for skip_ch in reversed(enc_channels):
            self.decoders.append(
                nn.Sequential(
                    _conv_block(prev_ch + skip_ch, skip_ch, kernel_size=3, pad="reflection"),
                    _conv_block(skip_ch, skip_ch, kernel_size=3, pad="reflection"),
                )
            )
            prev_ch = skip_ch

        self.final_conv = nn.Conv2d(base_channels, num_output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        h = x

        for enc, pool in zip(self.encoders, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoders, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([skip, h], dim=1)
            h = dec(h)

        out = self.final_conv(h)
        if self.need_sigmoid:
            out = torch.sigmoid(out)
        return out


def build_model(
    arch: str,
    num_input_channels: int,
    num_output_channels: int,
    base_channels: int = 64,
    num_scales: int = 4,
    skip_channels: int = 4,
    upsample_mode: str = "bilinear",
) -> nn.Module:
    """Factory function for constructing DIP generator networks."""
    arch_lower = arch.lower()
    if arch_lower == "skipnet":
        ch_down = tuple([base_channels] * num_scales)
        ch_up = tuple([base_channels] * num_scales)
        ch_skip = tuple([skip_channels] * num_scales)
        return SkipNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            num_channels_down=ch_down,
            num_channels_up=ch_up,
            num_channels_skip=ch_skip,
            upsample_mode=upsample_mode,
        )
    if arch_lower == "unet":
        return UNet(
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            base_channels=max(16, base_channels // 2),
            depth=num_scales,
        )
    raise ValueError(f"Unknown architecture '{arch}'. Choose 'skipnet' or 'unet'.")
