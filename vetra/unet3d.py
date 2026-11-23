from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dz = x2.size(2) - x1.size(2)
        dy = x2.size(3) - x1.size(3)
        dx = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2, dz // 2, dz - dz // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet3d(nn.Module):
    """3D U-Net with skip connections. Output has the same (T, H, W) as the input."""

    def __init__(self, in_channels: int = 2, out_channels: int = 16) -> None:
        super().__init__()
        c = [out_channels * s for s in (1, 2, 4, 8, 16)]
        self.inc = DoubleConv(in_channels, c[0])
        self.down1 = Down(c[0], c[1])
        self.down2 = Down(c[1], c[2])
        self.down3 = Down(c[2], c[3])
        self.down4 = Down(c[3], c[4] // 2)
        self.up1 = Up(c[4], c[3] // 2)
        self.up2 = Up(c[3], c[2] // 2)
        self.up3 = Up(c[2], c[1] // 2)
        self.up4 = Up(c[1], c[0])
        self.out_channels = c[0]

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x
