"""Replay a held-out subject through the online predictor in the terminal.

    python scripts/replay.py --subject 3 --speed 10

The subject must be in the test fold of the checkpoint, otherwise you are
looking at training data. The script picks the right fold automatically.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stream_eval import load_ckpt  # noqa: E402

from wearhar.data import FS, MS_ACTIVITY_NAMES, load_motionsense  # noqa: E402
from wearhar.stream import OnlinePredictor, Smoother, build_stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--speed", type=float, default=10.0, help="1 = real time")
    ap.add_argument("--smoother", default="hmm", choices=["none", "ema", "vote", "hmm"])
    ap.add_argument("--max-seconds", type=float, default=120)
    args = ap.parse_args()
    torch.set_num_threads(1)

    ckpt = None
    for f in sorted((ROOT / "checkpoints" / args.tag).glob(f"{args.model}_fold*.pt")):
        model, ck = load_ckpt(f)
        if args.subject in ck["test_subjects"]:
            ckpt = (model, ck)
            break
    if ckpt is None:
        sys.exit(f"subject {args.subject} not found in any test fold of checkpoints/{args.tag}")
    model, ck = ckpt

    recs, _ = load_motionsense(ROOT / "data" / "motionsense")
    sig, lab = build_stream([r for r in recs if r.subject == args.subject],
                            np.random.default_rng(args.subject))
    sig, lab = sig[: int(args.max_seconds * FS)], lab[: int(args.max_seconds * FS)]

    pred = OnlinePredictor(model, ck["window"], 25, sig.shape[1],
                           Smoother(len(MS_ACTIVITY_NAMES), args.smoother))
    correct = total = 0
    for i, s in enumerate(sig):
        out = pred.push(s)
        if out is not None:
            label, belief = out
            ok = label == lab[i]
            correct += ok
            total += 1
            bar = "#" * int(belief[label] * 20)
            print(f"{i / FS:7.1f}s  true {MS_ACTIVITY_NAMES[lab[i]]:10s} pred {MS_ACTIVITY_NAMES[label]:10s} "
                  f"{bar:20s} {'' if ok else '  <-- wrong'}")
            time.sleep(25 / FS / args.speed)
    print(f"\n{correct}/{total} decisions correct ({correct / max(total, 1):.1%})")


if __name__ == "__main__":
    main()
