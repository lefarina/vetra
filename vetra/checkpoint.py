"""Portable Vetra checkpoints: tensors and plain values only, loadable with weights_only=True."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from vetra.data import SpectrumTransform, WindStats
from vetra.hs_gain import HsGain
from vetra.models import Vetra, build_model

MODEL_KEYS = ("unet_out", "feature_hw", "decoder_width", "decoder_dropout", "large_window", "local_window")


@dataclass
class Pretrained:
    model: Vetra
    large_stats: WindStats
    local_stats: WindStats
    target_transform: SpectrumTransform
    config: dict
    site: dict = field(default_factory=dict)
    hs_gain: HsGain | None = None

    @torch.no_grad()
    def predict(self, large: np.ndarray, local: np.ndarray, apply_hs_gain: bool = True,
                batch_size: int = 16) -> np.ndarray:
        """Spectra (N, 36, 24) in physical units from N pairs of wind windows.

        ``large`` is (N, large_window, 2, H, W) and ``local`` is (N, local_window, 2, h, w),
        signed-square winds as in ``vetra.data``, each window ending at the time of the spectrum.
        Negative values are set to zero; Hs-Gain is applied when the checkpoint has one.
        """
        device = next(self.model.parameters()).device
        spectra = []
        for start in range(0, len(large), batch_size):
            batch_large = (large[start:start + batch_size] - self.large_stats.mean) / self.large_stats.std
            batch_local = (local[start:start + batch_size] - self.local_stats.mean) / self.local_stats.std
            output = self.model(torch.from_numpy(batch_large.astype(np.float32)).to(device),
                                torch.from_numpy(batch_local.astype(np.float32)).to(device))
            spectra.append(np.maximum(self.target_transform.inverse(output.float().cpu().numpy()), 0.0))
        spectra = np.concatenate(spectra)
        if apply_hs_gain and self.hs_gain is not None:
            spectra = self.hs_gain.apply(spectra)
        return spectra


def _stats_to_tensors(stats: WindStats) -> dict:
    return {"mean": torch.as_tensor(np.asarray(stats.mean)), "std": torch.as_tensor(np.asarray(stats.std))}


def _tensors_to_stats(values: dict) -> WindStats:
    return WindStats(mean=values["mean"].numpy(), std=values["std"].numpy())


def save_checkpoint(path: str | Path, model: Vetra, config: dict, large_stats: WindStats, local_stats: WindStats,
                    target_transform: SpectrumTransform, site: dict | None = None,
                    hs_gain: HsGain | None = None, extra: dict | None = None) -> None:
    payload = {
        "format": "vetra-checkpoint-v1",
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {**{k: config[k] for k in MODEL_KEYS if k in config},
                   "freq_num": model.freq_num, "direction_num": model.direction_num},
        "large_stats": _stats_to_tensors(large_stats),
        "local_stats": _stats_to_tensors(local_stats),
        "target_transform": target_transform.name,
        "site": dict(site or {}),
        "hs_gain": None if hs_gain is None else {"predicted_hs": hs_gain.predicted_hs.tolist(),
                                                 "energy_gain": hs_gain.energy_gain.tolist()},
        "extra": dict(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> Pretrained:
    """Load a Vetra checkpoint: the model in eval mode plus everything needed to prepare inputs."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") not in ("vetra-checkpoint-v1", "njord-checkpoint-v1"):  # v1.0.0 files used the old name
        raise ValueError(f"{path} is not a Vetra checkpoint")
    config = payload["config"]
    model = build_model(config, freq_num=config["freq_num"], direction_num=config["direction_num"])
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    gain = payload.get("hs_gain")
    return Pretrained(
        model=model,
        large_stats=_tensors_to_stats(payload["large_stats"]),
        local_stats=_tensors_to_stats(payload["local_stats"]),
        target_transform=SpectrumTransform(payload["target_transform"]),
        config=config,
        site=payload.get("site", {}),
        hs_gain=None if gain is None else HsGain(np.asarray(gain["predicted_hs"]), np.asarray(gain["energy_gain"])),
    )
