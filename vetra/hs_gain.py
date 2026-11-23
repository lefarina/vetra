"""Hs-Gain: optional post-training amplitude correction of Vetra spectra.

Each predicted spectrum is multiplied by an energy gain looked up from its
predicted Hs, so the spectral shape is unchanged. The gain is the ratio of true
to predicted energy in bins of predicted Hs (finer bins at the top), fitted on
validation predictions, made monotone non-decreasing in predicted Hs by
isotonic regression and held constant beyond the fitted range.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vetra.grid import IOWAGA_FREQUENCIES


def significant_wave_height(spectra: np.ndarray) -> np.ndarray:
    """Hs = 4 sqrt(m0) for (N, freq, dir) spectra on the IOWAGA grid."""
    df = np.gradient(IOWAGA_FREQUENCIES)
    dtheta = 2 * np.pi / spectra.shape[2]
    m0 = (spectra.astype(np.float64) * df[None, :, None]).sum((1, 2)) * dtheta
    return 4 * np.sqrt(np.maximum(m0, 0))


@dataclass(frozen=True)
class HsGain:
    predicted_hs: np.ndarray   # bin centres (m)
    energy_gain: np.ndarray    # gain at each centre

    def apply(self, spectra: np.ndarray) -> np.ndarray:
        gain = np.interp(significant_wave_height(spectra), self.predicted_hs, self.energy_gain)
        return (spectra * gain[:, None, None]).astype(spectra.dtype)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"predicted_hs": self.predicted_hs.tolist(),
                                          "energy_gain": self.energy_gain.tolist()}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "HsGain":
        data = json.loads(Path(path).read_text())
        return cls(np.asarray(data["predicted_hs"]), np.asarray(data["energy_gain"]))


def bin_energies(prediction: np.ndarray, target: np.ndarray, min_count: int = 5):
    """Mean predicted Hs and summed true / predicted energy (Hs^2) per predicted-Hs bin."""
    hs_pred = significant_wave_height(prediction)
    energy_pred, energy_true = hs_pred ** 2, significant_wave_height(target) ** 2
    quantiles = np.r_[np.linspace(0, 0.9, 10), np.linspace(0.92, 1, 16)]
    edges = np.quantile(hs_pred, quantiles)
    centres, true_sums, pred_sums = [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (hs_pred >= low) & (hs_pred <= high)
        if in_bin.sum() >= min_count:
            centres.append(hs_pred[in_bin].mean())
            true_sums.append(energy_true[in_bin].sum())
            pred_sums.append(energy_pred[in_bin].sum())
    return np.array(centres), np.array(true_sums), np.array(pred_sums)


def isotonic_increasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = []   # [weighted mean, weight, size]
    for value, weight in zip(values, weights):
        blocks.append([value, weight, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, n2 = blocks.pop()
            v1, w1, n1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([np.full(n, v) for v, _, n in blocks])


def fit_hs_gain(prediction: np.ndarray, target: np.ndarray, min_count: int = 5) -> HsGain:
    """Fit on physical-space (N, freq, dir) validation predictions and targets.

    Bins that violate monotonicity are pooled, so a pooled gain is again the
    energy ratio of the merged bins.
    """
    centres, true_sums, pred_sums = bin_energies(prediction, target, min_count)
    return HsGain(centres, isotonic_increasing(true_sums / pred_sums, pred_sums))
