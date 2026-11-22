from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_metrics(y_true, y_pred, n_classes: int) -> dict:
    cm = confusion(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    present = cm.sum(1) > 0  # don't average over classes missing from y_true
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_f1": f1.tolist(),
        "confusion": cm.tolist(),
    }
