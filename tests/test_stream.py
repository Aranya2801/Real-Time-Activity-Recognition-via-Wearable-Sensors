import numpy as np
import pytest
import torch

from wearhar.models import WithNormalization, build_model
from wearhar.stream import OnlinePredictor, Smoother, causal_window_probs, stream_metrics


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return WithNormalization(build_model("cnn", 6, 4), np.zeros(6), np.ones(6)).eval()


def test_online_matches_batched(model):
    sig = np.random.default_rng(1).normal(size=(500, 6)).astype(np.float32)
    online = OnlinePredictor(model, window=64, hop=16, n_channels=6)
    got = [r[1] for s in sig if (r := online.push(s)) is not None]
    at, probs = causal_window_probs(model, sig, 64, 16)
    assert len(got) == len(probs)
    np.testing.assert_allclose(np.stack(got), probs, atol=1e-5)
    assert at[0] == 63


def test_hmm_with_uniform_transitions_equals_argmax():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(5), size=200)
    flat = Smoother(5, "hmm", stay=0.2).run(p)  # stay = 1/n -> no memory
    np.testing.assert_array_equal(flat, p.argmax(1))


def test_smoothers_reduce_flicker():
    rng = np.random.default_rng(0)
    true = np.repeat([0, 1, 2], 100)
    p = np.full((300, 3), 0.1)
    p[np.arange(300), true] = 0.8
    noisy = rng.random(300) < 0.25  # 25% of decisions point to a random wrong class
    p[noisy] = rng.dirichlet(np.ones(3) * 0.3, size=noisy.sum())
    p /= p.sum(1, keepdims=True)

    raw = Smoother(3, "none").run(p)
    for method in ("ema", "vote", "hmm"):
        sm = Smoother(3, method).run(p)
        assert np.count_nonzero(np.diff(sm)) < np.count_nonzero(np.diff(raw))
        assert (sm == true).mean() >= (raw == true).mean()


def test_stream_metrics_perfect():
    y = np.repeat([0, 1, 0], 40)
    m = stream_metrics(y, y, hop=25)
    assert m["accuracy"] == 1.0
    assert m["median_delay_s"] == 0.0
    assert m["switches_per_min"] == m["true_switches_per_min"]


def test_onnx_export_roundtrip(tmp_path, model):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    from wearhar.export import check_onnx, export_onnx

    path = export_onnx(model, 6, 64, tmp_path / "m.onnx")
    assert check_onnx(model, path, 6, 64) < 1e-4
