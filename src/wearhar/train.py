"""Training loop for the deep models + the classical baseline."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .augment import Augmenter
from .metrics import classification_metrics
from .models import WithNormalization, build_model


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-3
    label_smoothing: float = 0.05
    patience: int = 8
    augment: bool = True
    rotate_deg: float = 30.0
    jitter: float = 0.05
    scale: float = 0.1
    seed: int = 0
    balance: bool = True  # class-weighted loss (sit/std have fewer windows than wlk)
    verbose: bool = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size])
        out.append(torch.softmax(model(xb), -1).numpy())
    return np.concatenate(out)


def train_deep(model_name: str, X_tr, y_tr, X_val, y_val, n_classes: int,
               cfg: TrainConfig | None = None, model_kw: dict | None = None):
    """Train one model. X_* are raw (un-normalised) windows (N, C, L).

    Returns the best model (by validation macro-F1), wrapped with its
    normaliser, and a small history dict.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    mean = X_tr.mean(axis=(0, 2))
    std = X_tr.std(axis=(0, 2)) + 1e-6
    net = build_model(model_name, X_tr.shape[1], n_classes, **(model_kw or {}))
    model = WithNormalization(net, mean, std)

    aug = Augmenter(cfg.rotate_deg, cfg.jitter, cfg.scale, channel_std=std) if cfg.augment else None

    if cfg.balance:
        counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
        w = counts.sum() / (n_classes * np.maximum(counts, 1))
        weight = torch.tensor(w)
    else:
        weight = None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(X_tr) / cfg.batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.15)

    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    best_f1, best_state, bad_epochs = -1.0, None, 0
    history = {"train_loss": [], "val_f1": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        running = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            xb, yb = Xt[idx], yt[idx]
            if aug is not None:
                xb = aug(xb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            running += loss.item() * len(idx)

        val_pred = predict_proba(model, X_val).argmax(1)
        m = classification_metrics(y_val, val_pred, n_classes)
        history["train_loss"].append(running / len(Xt))
        history["val_f1"].append(m["macro_f1"])
        history["val_acc"].append(m["accuracy"])
        if cfg.verbose:
            print(f"  epoch {epoch + 1:3d}  loss {running / len(Xt):.4f}  "
                  f"val acc {m['accuracy']:.4f}  val F1 {m['macro_f1']:.4f}")

        if m["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_state, bad_epochs = m["macro_f1"], copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    history.update(best_val_f1=best_f1, epochs_run=epoch + 1, train_seconds=time.time() - t0,
                   config=asdict(cfg))
    return model, history


def train_classical(X_tr, y_tr, seed: int = 0, n_estimators: int = 300):
    """Random forest on hand-crafted features. Returns (fitted pipeline, feature fn)."""
    from sklearn.ensemble import RandomForestClassifier

    from .features import window_features

    clf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
                                 class_weight="balanced", random_state=seed)
    clf.fit(window_features(X_tr), y_tr)
    return clf
