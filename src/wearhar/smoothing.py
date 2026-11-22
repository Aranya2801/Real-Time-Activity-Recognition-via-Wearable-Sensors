"""Causal smoothing of per-window class probabilities (numpy only).

Raw per-window predictions flicker, especially around transitions and
between sitting/standing. Everything has to stay causal (a live app can't
look into the future), so the smoothers only use the past:

* ``ema``   exponential moving average of the class probabilities
* ``vote``  majority vote over the last k decisions
* ``hmm``   forward filtering with a "sticky" transition matrix - a
            discrete Bayes filter that treats the model's softmax as a
            (scaled) likelihood. One parameter: the probability of staying
            in the same activity between two decisions.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class Smoother:
    def __init__(self, n_classes: int, method: str = "hmm", alpha: float = 0.3, k: int = 5,
                 stay: float = 0.97, prior=None):
        if method not in ("none", "ema", "vote", "hmm"):
            raise ValueError(method)
        self.n = n_classes
        self.method = method
        self.alpha = alpha
        self.k = k
        self.stay = stay
        self.prior = np.full(n_classes, 1.0 / n_classes) if prior is None else np.asarray(prior, float)
        A = np.full((n_classes, n_classes), (1 - stay) / (n_classes - 1))
        np.fill_diagonal(A, stay)
        self.A = A
        self.reset()

    def reset(self):
        self.state = None
        self.history = deque(maxlen=self.k)

    def update(self, probs: np.ndarray) -> tuple[int, np.ndarray]:
        """Feed one softmax vector, get back (label, smoothed belief)."""
        p = np.asarray(probs, dtype=np.float64)
        if self.method == "none":
            return int(p.argmax()), p
        if self.method == "ema":
            self.state = p if self.state is None else self.alpha * p + (1 - self.alpha) * self.state
            return int(self.state.argmax()), self.state
        if self.method == "vote":
            self.history.append(int(p.argmax()))
            counts = np.bincount(self.history, minlength=self.n)
            # ties go to the most recent prediction
            best = np.flatnonzero(counts == counts.max())
            label = self.history[-1] if self.history[-1] in best else int(best[0])
            return label, counts / counts.sum()
        # hmm forward step
        lik = p / self.prior
        pred = self.prior if self.state is None else self.A.T @ self.state
        b = pred * lik
        b /= b.sum()
        self.state = b
        return int(b.argmax()), b

    def run(self, probs_seq: np.ndarray) -> np.ndarray:
        self.reset()
        return np.array([self.update(p)[0] for p in probs_seq])
