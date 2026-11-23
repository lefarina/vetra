import torch

from vetra.models import Vetra, build_model


def test_output_shape_and_parameter_count():
    model = build_model({"unet_out": 32})
    large = torch.zeros(2, 80, 2, 72, 60)
    local = torch.zeros(2, 24, 2, 81, 81)
    model.eval()
    with torch.no_grad():
        out = model(large, local)
    assert out.shape == (2, 36, 24)
    assert sum(p.numel() for p in model.parameters()) > 25_000_000


def test_small_model_runs_on_cpu():
    model = Vetra(unet_out=4).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 80, 2, 16, 16), torch.randn(1, 24, 2, 16, 16))
    assert torch.isfinite(out).all()
