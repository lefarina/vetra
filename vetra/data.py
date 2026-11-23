"""Wind inputs, spectrum targets and the training dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindStats:
    """Per-component mean and standard deviation of a wind field (training portion only)."""

    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class SpectrumTransform:
    """Target transform. Vetra is trained on log(1 + E)."""

    name: str = "log1p"

    def forward(self, y: np.ndarray) -> np.ndarray:
        if self.name == "log1p":
            return np.log1p(y).astype(np.float32)
        if self.name == "raw":
            return y.astype(np.float32)
        raise ValueError(f"Unknown target transform: {self.name}")

    def inverse(self, y: np.ndarray) -> np.ndarray:
        if self.name == "log1p":
            return np.expm1(y)
        if self.name == "raw":
            return y
        raise ValueError(f"Unknown target transform: {self.name}")


def load_year_arrays(site: str, year: int | list[int], data_root: str | Path = "data"
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate whole years of basin winds, local winds and target spectra.

    Expected layout under ``data_root``, one file per month::

        <site>/<YYYY>/large_<YYYYMM>.npy      (T, 2, H, W)   basin winds
        <site>/<YYYY>/local_<YYYYMM>.npy      (T, 2, h, w)   local winds
        <site>/<YYYY>/spectrum_<YYYYMM>.npy   (T, 36, 24)    target spectra

    The three files of a month share the same T time steps, every 3 h. Winds
    are the signed squares (u|u|, v|v|) of the 10-m wind components, on any
    grids of at least 16 x 16 points. Spectra are on the grid of ``vetra.grid``.
    Returns ``(large, local, target)`` concatenated in ascending year order.
    """
    years = [year] if isinstance(year, int) else sorted(year)
    if not years:
        raise ValueError("year list must contain at least one year")

    large_chunks, local_chunks, target_chunks = [], [], []
    for y in years:
        year_dir = Path(data_root) / site / f"{y:04d}"
        stamps = [f"{y:04d}{m:02d}" for m in range(1, 13)]
        large_files = [year_dir / f"large_{s}.npy" for s in stamps]
        local_files = [year_dir / f"local_{s}.npy" for s in stamps]
        target_files = [year_dir / f"spectrum_{s}.npy" for s in stamps]
        for path in [*large_files, *local_files, *target_files]:
            if not path.exists():
                raise FileNotFoundError(path)
        large_y = np.concatenate([np.load(p) for p in large_files], axis=0)
        local_y = np.concatenate([np.load(p) for p in local_files], axis=0)
        target_y = np.concatenate([np.load(p) for p in target_files], axis=0)
        n = min(len(large_y), len(local_y), len(target_y))
        large_chunks.append(large_y[:n])
        local_chunks.append(local_y[:n])
        target_chunks.append(target_y[:n])

    return (np.concatenate(large_chunks, axis=0), np.concatenate(local_chunks, axis=0),
            np.concatenate(target_chunks, axis=0))


def make_sample_indices(n_times: int, large_window: int = 80, max_samples: int | None = None) -> np.ndarray:
    """Start index of every sample; a sample needs ``large_window`` times of wind history."""
    indices = np.arange(0, n_times - large_window + 1, dtype=np.int64)
    return indices if max_samples is None else indices[:max_samples]


def split_indices(indices: np.ndarray, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split: the first ``train_fraction`` of samples train, the rest validate."""
    split = int(len(indices) * train_fraction)
    return indices[:split], indices[split:]


def wind_stats(wind: np.ndarray, end_time: int) -> WindStats:
    sample = wind[:end_time]
    mean = sample.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = np.maximum(sample.std(axis=(0, 2, 3), keepdims=True).astype(np.float32), 1e-6)
    return WindStats(mean=mean, std=std)


class WindSpectrumDataset(Dataset):
    """One sample = basin and local wind windows ending at the target time, and the target spectrum."""

    def __init__(
        self,
        large_wind: np.ndarray,
        local_wind: np.ndarray,
        target: np.ndarray,
        sample_indices: np.ndarray,
        large_stats: WindStats,
        local_stats: WindStats,
        target_transform: SpectrumTransform,
        large_window: int = 80,
        local_window: int = 24,
        normalize_wind: bool = True,
    ) -> None:
        self.large_wind = large_wind
        self.local_wind = local_wind
        self.target = target
        self.sample_indices = sample_indices
        self.large_stats = large_stats
        self.local_stats = local_stats
        self.target_transform = target_transform
        self.large_window = large_window
        self.local_window = local_window
        self.normalize_wind = normalize_wind

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(self.sample_indices[item])
        end = start + self.large_window
        large = self.large_wind[start:end]
        local = self.local_wind[end - self.local_window:end]
        if self.normalize_wind:
            large = (large - self.large_stats.mean) / self.large_stats.std
            local = (local - self.local_stats.mean) / self.local_stats.std
        target = self.target_transform.forward(self.target[end - 1])
        return (torch.from_numpy(large.astype(np.float32)), torch.from_numpy(local.astype(np.float32)),
                torch.from_numpy(target.astype(np.float32)))
