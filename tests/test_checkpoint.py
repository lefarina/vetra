import numpy as np
import torch

from vetra.checkpoint import load_checkpoint, save_checkpoint
from vetra.data import SpectrumTransform, WindStats
from vetra.hs_gain import HsGain
from vetra.models import Vetra


def test_round_trip_and_predict_on_any_grid(tmp_path):
    torch.manual_seed(0)
    model = Vetra(unet_out=4).eval()
    stats = WindStats(mean=np.full((1, 2, 1, 1), 3.0, np.float32), std=np.full((1, 2, 1, 1), 20.0, np.float32))
    gain = HsGain(np.array([0.5, 5.0]), np.array([1.0, 1.2]))
    path = tmp_path / "vetra.pt"
    save_checkpoint(path, model, {"unet_out": 4, "large_window": 80, "local_window": 24}, stats, stats,
                    SpectrumTransform("log1p"), site={"name": "Anywhere"}, hs_gain=gain)

    pretrained = load_checkpoint(path)
    large = np.random.default_rng(0).normal(size=(3, 80, 2, 23, 37)).astype(np.float32)
    local = np.random.default_rng(1).normal(size=(3, 24, 2, 41, 41)).astype(np.float32)
    raw = pretrained.predict(large, local, apply_hs_gain=False, batch_size=2)
    with torch.no_grad():
        expected = model(torch.from_numpy((large - 3) / 20), torch.from_numpy((local - 3) / 20)).numpy()
    np.testing.assert_allclose(raw, np.maximum(np.expm1(expected), 0), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(pretrained.predict(large, local), gain.apply(raw), rtol=1e-6)
    assert pretrained.site == {"name": "Anywhere"}
