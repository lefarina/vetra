"""The Vetra model: two 3D U-Net wind encoders and a spectral decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vetra.unet3d import UNet3d


def _make_spatial_decoder(
    merged_channels: int,
    freq_num: int,
    direction_num: int,
    decoder_width: int,
    decoder_dropout: float,
) -> nn.Sequential:
    """Map the merged 8x8 feature map to one value per frequency and direction (log(1+E))."""
    layers: list[nn.Module] = [
        nn.Conv2d(merged_channels, decoder_width, kernel_size=3, padding=1),
        nn.BatchNorm2d(decoder_width),
        nn.SiLU(),
    ]
    if decoder_dropout > 0:
        layers.append(nn.Dropout2d(p=decoder_dropout))
    layers.extend([
        nn.AdaptiveAvgPool2d((freq_num, direction_num)),
        nn.Conv2d(decoder_width, decoder_width, kernel_size=3, padding=1),
        nn.BatchNorm2d(decoder_width),
        nn.SiLU(),
    ])
    if decoder_dropout > 0:
        layers.append(nn.Dropout2d(p=decoder_dropout))
    layers.extend([
        nn.Conv2d(decoder_width, decoder_width // 2, kernel_size=3, padding=1),
        nn.BatchNorm2d(decoder_width // 2),
        nn.SiLU(),
    ])
    if decoder_dropout > 0:
        layers.append(nn.Dropout2d(p=decoder_dropout))
    layers.append(nn.Conv2d(decoder_width // 2, 1, kernel_size=1))
    return nn.Sequential(*layers)


class VetraEncoderBranch(nn.Module):
    """UNet3d encoder that collapses (T, H, W) to a (feature_hw, feature_hw) feature map."""

    def __init__(self, unet_out: int = 32, feature_hw: int = 8) -> None:
        super().__init__()
        self.unet = UNet3d(in_channels=2, out_channels=unet_out)
        self.feature_hw = feature_hw
        self.out_channels = self.unet.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)   # (B, 2, T, H, W)
        x = self.unet(x)                # (B, unet_out, T, H, W)
        x = F.adaptive_avg_pool3d(x, (x.shape[2], self.feature_hw, self.feature_hw))
        return x.mean(dim=2)             # (B, unet_out, feature_hw, feature_hw)


class Vetra(nn.Module):
    """Basin and local wind histories -> directional spectrum in log(1+E)."""

    def __init__(
        self,
        freq_num: int = 36,
        direction_num: int = 24,
        unet_out: int = 32,
        feature_hw: int = 8,
        decoder_width: int = 64,
        decoder_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.freq_num = freq_num
        self.direction_num = direction_num
        self.large_branch = VetraEncoderBranch(unet_out=unet_out, feature_hw=feature_hw)
        self.local_branch = VetraEncoderBranch(unet_out=unet_out, feature_hw=feature_hw)
        merged = self.large_branch.out_channels + self.local_branch.out_channels
        self.decoder = _make_spatial_decoder(merged, freq_num, direction_num, decoder_width, decoder_dropout)

    def forward(self, large: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([self.large_branch(large), self.local_branch(local)], dim=1)
        return self.decoder(merged).squeeze(1)


def build_model(config: dict, freq_num: int = 36, direction_num: int = 24) -> Vetra:
    """Build Vetra from a checkpoint or command-line configuration."""
    return Vetra(
        freq_num=freq_num,
        direction_num=direction_num,
        unet_out=config.get("unet_out", 32),
        feature_hw=config.get("feature_hw", 8),
        decoder_width=config.get("decoder_width", 64),
        decoder_dropout=config.get("decoder_dropout", 0.05),
    )
