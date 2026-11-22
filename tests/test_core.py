import numpy as np
import pytest
import torch

from wearhar.augment import Augmenter, random_rotation_matrices
from wearhar.data import Recording, sliding_windows
from wearhar.features import window_features
from wearhar.metrics import classification_metrics
from wearhar.models import MODELS, WithNormalization, build_model
from wearhar.splits import holdout_val, subject_folds, subject_split


def fake_recordings(n_subjects=6, T=400, C=9, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for s in range(1, n_subjects + 1):
        for label in range(3):
            recs.append(Recording(s, label, 1, rng.normal(size=(T, C)).astype(np.float32)))
    return recs


def test_sliding_windows_shapes_and_content():
    recs = fake_recordings()
    W = sliding_windows(recs, win=128, hop=64, class_names=["a", "b", "c"], channels=list("abcdefghi"))
    per_rec = (400 - 128) // 64 + 1
    assert W.X.shape == (len(recs) * per_rec, 9, 128)
    # window k of recording 0 must equal the raw slice
    np.testing.assert_array_equal(W.X[1], recs[0].signal[64:192].T)
    assert set(np.unique(W.subject)) == set(range(1, 7))


def test_short_recordings_are_skipped():
    recs = [Recording(1, 0, 1, np.zeros((50, 3), np.float32)),
            Recording(1, 1, 1, np.zeros((200, 3), np.float32))]
    W = sliding_windows(recs, 128, 64, ["a", "b"], ["x", "y", "z"])
    assert (W.y == 1).all()


def test_subject_folds_are_disjoint_and_complete():
    subjects = np.repeat(np.arange(1, 25), 10)
    folds = subject_folds(subjects, 5)
    allsub = np.concatenate(folds)
    assert sorted(allsub.tolist()) == list(range(1, 25))
    for f in folds:
        tr, te = subject_split(subjects, f)
        assert not set(subjects[tr]) & set(subjects[te])
        tr2, va = holdout_val(subjects, tr)
        assert not set(subjects[tr2]) & set(subjects[va])
        assert len(np.unique(subjects[va])) == 2


@pytest.mark.parametrize("name", list(MODELS))
def test_models_forward_and_backward(name):
    model = build_model(name, 9, 6)
    x = torch.randn(4, 9, 128, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 6)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_tcn_is_causal():
    # changing the last sample must not change the TCN features at earlier steps
    model = build_model("tcn", 3, 2).eval()
    x = torch.randn(1, 3, 64)
    x2 = x.clone()
    x2[..., -1] += 10
    with torch.no_grad():
        h1 = model.blocks(model.inp(x))
        h2 = model.blocks(model.inp(x2))
    assert torch.allclose(h1[..., :-1], h2[..., :-1], atol=1e-5)


def test_normalization_wrapper():
    net = build_model("cnn", 3, 2)
    mean, std = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])
    wrapped = WithNormalization(net, mean, std).eval()
    x = torch.randn(2, 3, 32)
    with torch.no_grad():
        net.eval()
        expected = net((x - torch.tensor(mean).float().view(1, 3, 1)) / 2.0)
        assert torch.allclose(wrapped(x), expected, atol=1e-6)


def test_rotation_matrices_are_orthonormal():
    R = random_rotation_matrices(16, 45)
    eye = torch.eye(3).expand(16, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-5)
    assert torch.allclose(torch.det(R), torch.ones(16), atol=1e-5)


def test_rotation_preserves_vector_norm():
    aug = Augmenter(rotate_deg=90, jitter=0, scale=0, p=1.0)
    x = torch.randn(8, 9, 50)
    y = aug(x)
    n_before = x.view(8, 3, 3, 50).norm(dim=2)
    n_after = y.view(8, 3, 3, 50).norm(dim=2)
    assert torch.allclose(n_before, n_after, atol=1e-4)
    assert not torch.allclose(x, y)


def test_features_finite():
    X = np.random.default_rng(0).normal(size=(10, 9, 128)).astype(np.float32)
    F = window_features(X)
    assert F.shape[0] == 10 and np.isfinite(F).all()
    # constant signal should not produce nans either
    assert np.isfinite(window_features(np.zeros((2, 9, 128), np.float32))).all()


def test_metrics():
    m = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert m["accuracy"] == 0.75
    assert m["confusion"] == [[1, 1], [0, 2]]
    assert abs(m["macro_f1"] - np.mean([2 / 3, 0.8])) < 1e-9
