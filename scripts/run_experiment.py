"""Cross-validated experiment runner.

Every number in the README comes from this script (plus stream_eval.py and
benchmark_latency.py). Examples:

    # main table: all models, 5-fold leave-subjects-out
    python scripts/run_experiment.py --models rf cnn convgru tcn transformer --tag main --save-models

    # same thing but with the (leaky) random window split
    python scripts/run_experiment.py --models rf cnn --split random --tag leak_random

    # window length / sensors / augmentation ablations
    python scripts/run_experiment.py --models cnn --window 64 --hop 32 --tag win64
    python scripts/run_experiment.py --models cnn --sensors acc --tag sens_acc
    python scripts/run_experiment.py --models cnn --no-augment --tag noaug
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearhar.data import load_windows  # noqa: E402
from wearhar.features import window_features  # noqa: E402
from wearhar.metrics import classification_metrics  # noqa: E402
from wearhar.models import n_params  # noqa: E402
from wearhar.splits import holdout_val, random_split, subject_folds, subject_split  # noqa: E402
from wearhar.train import TrainConfig, predict_proba, train_classical, train_deep  # noqa: E402


def make_folds(W, split: str, n_folds: int, seed: int):
    if split == "subject":
        for k, test_subj in enumerate(subject_folds(W.subject, n_folds, seed)):
            tr, te = subject_split(W.subject, test_subj)
            tr, va = holdout_val(W.subject, tr, seed=seed + k)
            yield k, tr, va, te, test_subj.tolist()
    else:
        for k in range(n_folds):
            tr, te = random_split(len(W), 0.2, seed=seed + k)
            # validation is also random here, which is the "usual" (leaky) protocol
            rng = np.random.default_rng(seed + k + 100)
            va = rng.choice(tr, size=len(tr) // 10, replace=False)
            tr = np.setdiff1d(tr, va)
            yield k, tr, va, te, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="motionsense", choices=["motionsense", "uci_har"])
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--models", nargs="+", default=["cnn"])
    ap.add_argument("--split", default="subject", choices=["subject", "random"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--window", type=int, default=128, help="samples (50 Hz)")
    ap.add_argument("--hop", type=int, default=64)
    ap.add_argument("--sensors", nargs="+", default=None,
                    help="motionsense: acc gyro grav att   uci_har: acc gyro total")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--rotate-deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    kw = {}
    if args.sensors:
        kw["sensors"] = tuple(args.sensors)
    if args.dataset == "motionsense":
        kw.update(win=args.window, hop=args.hop)
    W = load_windows(args.dataset, args.data_dir, **kw)
    n_classes = len(W.class_names)
    print(f"{args.dataset}: {len(W)} windows, {W.X.shape[1]} channels, "
          f"{len(np.unique(W.subject))} subjects")

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    ckpt_dir = ROOT / "checkpoints" / args.tag
    runs = []

    for k, tr, va, te, test_subj in make_folds(W, args.split, args.folds, args.seed):
        for name in args.models:
            t0 = time.time()
            if name == "rf":
                clf = train_classical(W.X[np.concatenate([tr, va])], W.y[np.concatenate([tr, va])],
                                      seed=args.seed + k)
                pred = clf.predict(window_features(W.X[te]))
                extra = {"params": None}
            else:
                cfg = TrainConfig(epochs=args.epochs, augment=not args.no_augment,
                                  rotate_deg=args.rotate_deg, seed=args.seed + k)
                model, hist = train_deep(name, W.X[tr], W.y[tr], W.X[va], W.y[va], n_classes, cfg)
                pred = predict_proba(model, W.X[te]).argmax(1)
                extra = {"params": n_params(model.net), "epochs_run": hist["epochs_run"],
                         "best_val_f1": hist["best_val_f1"]}
                if args.save_models:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": name, "state_dict": model.state_dict(),
                                "in_ch": W.X.shape[1], "n_classes": n_classes,
                                "window": W.X.shape[2], "channels": W.channels,
                                "class_names": W.class_names, "test_subjects": test_subj},
                               ckpt_dir / f"{name}_fold{k}.pt")
            m = classification_metrics(W.y[te], pred, n_classes)
            run = {"model": name, "fold": k, "test_subjects": test_subj, "n_test": int(len(te)),
                   "seconds": round(time.time() - t0, 1), **extra, **m}
            runs.append(run)
            print(f"fold {k}  {name:12s} acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
                  f"({run['seconds']}s)", flush=True)

    summary = {}
    for name in args.models:
        r = [x for x in runs if x["model"] == name]
        acc = np.array([x["accuracy"] for x in r])
        f1 = np.array([x["macro_f1"] for x in r])
        summary[name] = {"acc_mean": acc.mean(), "acc_std": acc.std(), "f1_mean": f1.mean(),
                         "f1_std": f1.std(), "params": r[0].get("params"),
                         "per_class_f1_mean": np.mean([x["per_class_f1"] for x in r], 0).tolist(),
                         "confusion_sum": np.sum([x["confusion"] for x in r], 0).tolist()}
        print(f"{name:12s} acc {acc.mean():.4f} ± {acc.std():.4f}   F1 {f1.mean():.4f} ± {f1.std():.4f}")

    result = {"args": vars(args), "class_names": W.class_names, "channels": W.channels,
              "n_windows": len(W), "summary": summary, "runs": runs}
    with open(out_dir / f"{args.tag}.json", "w") as f:
        json.dump(result, f, indent=1, default=float)


if __name__ == "__main__":
    main()
