"""Online (streaming) inference.

In a real app samples arrive one at a time. `OnlinePredictor` keeps a ring
buffer of the last `window` samples and runs the model every `hop`
samples, so with window=128, hop=25 you get a decision every 0.5 s based
on the last 2.56 s. Smoothing lives in smoothing.py.
"""

from __future__ import annotations

import numpy as np
import torch

from .smoothing import Smoother

__all__ = ["OnlinePredictor", "Smoother", "build_stream", "causal_window_probs", "stream_metrics"]


class OnlinePredictor:
    """Sample-by-sample wrapper around a trained (normalisation-included) model."""

    def __init__(self, model: torch.nn.Module, window: int, hop: int, n_channels: int,
                 smoother: Smoother | None = None):
        self.model = model.eval()
        self.window = window
        self.hop = hop
        self.buf = np.zeros((window, n_channels), dtype=np.float32)
        self.filled = 0
        self.since_last = 0
        self.smoother = smoother

    def reset(self):
        self.buf[:] = 0
        self.filled = 0
        self.since_last = 0
        if self.smoother:
            self.smoother.reset()

    @torch.no_grad()
    def push(self, sample) -> tuple[int, np.ndarray] | None:
        """Add one sample (n_channels,). Returns (label, probs) when a decision is made."""
        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = sample
        self.filled = min(self.filled + 1, self.window)
        self.since_last += 1
        if self.filled < self.window or self.since_last < self.hop:
            return None
        self.since_last = 0
        x = torch.from_numpy(np.ascontiguousarray(self.buf.T[None]))
        probs = torch.softmax(self.model(x), -1)[0].numpy()
        if self.smoother is None:
            return int(probs.argmax()), probs
        return self.smoother.update(probs)


# ---------------------------------------------------------------------------
# helpers for offline evaluation of the streaming setting
# ---------------------------------------------------------------------------

def build_stream(recordings, rng: np.random.Generator):
    """Concatenate one subject's recordings in random order -> (signal, per-sample labels)."""
    order = rng.permutation(len(recordings))
    sig = np.concatenate([recordings[i].signal for i in order])
    lab = np.concatenate([np.full(len(recordings[i].signal), recordings[i].label) for i in order])
    return sig, lab


@torch.no_grad()
def causal_window_probs(model, signal: np.ndarray, window: int, hop: int, batch: int = 1024):
    """Equivalent to pushing every sample through OnlinePredictor, but batched.

    Returns (decision_sample_index, probs). Decision i is made at sample
    end_i - 1 using samples [end_i - window, end_i).
    """
    ends = np.arange(window, len(signal) + 1, hop)
    view = np.lib.stride_tricks.sliding_window_view(signal, window, axis=0)  # (T-w+1, C, w)
    out = []
    for i in range(0, len(ends), batch):
        idx = ends[i:i + batch] - window
        xb = torch.from_numpy(np.ascontiguousarray(view[idx], dtype=np.float32))
        out.append(torch.softmax(model(xb), -1).numpy())
    return ends - 1, np.concatenate(out)


def stream_metrics(pred: np.ndarray, true: np.ndarray, hop: int, fs: int = 50) -> dict:
    """pred/true are per-decision labels."""
    minutes = len(true) * hop / fs / 60
    true_switch = np.flatnonzero(np.diff(true) != 0) + 1
    pred_switch = np.count_nonzero(np.diff(pred) != 0)

    # delay: decisions until the prediction first matches the new activity
    delays = []
    bounds = list(true_switch) + [len(true)]
    for s, e in zip(bounds[:-1], bounds[1:]):
        hit = np.flatnonzero(pred[s:e] == true[s])
        delays.append(hit[0] * hop / fs if len(hit) else (e - s) * hop / fs)
    return {
        "accuracy": float((pred == true).mean()),
        "switches_per_min": pred_switch / minutes,
        "true_switches_per_min": len(true_switch) / minutes,
        "median_delay_s": float(np.median(delays)) if delays else 0.0,
    }
