"""Train/test splitting.

A lot of HAR papers shuffle windows and split them randomly. With 50%
overlapping windows that puts half of every test window into the training
set, and the same person is on both sides. `random_split` is kept here
only so we can measure how much that inflates the numbers
(see scripts/leakage_experiment.py).
"""

from __future__ import annotations

import numpy as np


def subject_folds(subjects: np.ndarray, n_folds: int = 5, seed: int = 0) -> list[np.ndarray]:
    """Split the unique subject ids into `n_folds` disjoint groups."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(subjects)
    rng.shuffle(uniq)
    return [np.sort(g) for g in np.array_split(uniq, n_folds)]


def subject_split(subjects: np.ndarray, test_subjects) -> tuple[np.ndarray, np.ndarray]:
    test_mask = np.isin(subjects, test_subjects)
    return np.where(~test_mask)[0], np.where(test_mask)[0]


def random_split(n: int, test_frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * test_frac))
    return np.sort(perm[k:]), np.sort(perm[:k])


def holdout_val(subjects: np.ndarray, train_idx: np.ndarray, n_val_subjects: int = 2,
                seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Carve a few whole subjects out of the training set for early stopping."""
    rng = np.random.default_rng(seed + 1000)
    tr_subj = np.unique(subjects[train_idx])
    val_subj = rng.choice(tr_subj, size=min(n_val_subjects, len(tr_subj) - 1), replace=False)
    is_val = np.isin(subjects[train_idx], val_subj)
    return train_idx[~is_val], train_idx[is_val]
