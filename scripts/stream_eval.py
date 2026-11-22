"""Evaluate the models as they would run live.

For every held-out subject we glue their recordings together in random
order (so the stream contains activity changes), push it through the
model causally with a decision every `hop` samples, and compare the
smoothing strategies.

Uses the fold checkpoints written by
    python scripts/run_experiment.py --models ... --tag main --save-models

Smoother parameters are fixed up front (not tuned on the test subjects);
the --sweep option just shows how sensitive the HMM is to its one knob.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearhar.data import MS_ACTIVITY_NAMES, load_motionsense  # noqa: E402
from wearhar.metrics import classification_metrics  # noqa: E402
from wearhar.models import WithNormalization, build_model  # noqa: E402
from wearhar.stream import Smoother, build_stream, causal_window_probs, stream_metrics  # noqa: E402

SMOOTHERS = {
    "none": dict(method="none"),
    "ema": dict(method="ema", alpha=0.3),
    "vote": dict(method="vote", k=5),
    "hmm": dict(method="hmm", stay=0.97),
}


def load_ckpt(path: Path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build_model(ck["model"], ck["in_ch"], ck["n_classes"])
    model = WithNormalization(net, np.zeros(ck["in_ch"]), np.ones(ck["in_ch"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="+", default=["cnn", "convgru", "tcn", "transformer"])
    ap.add_argument("--hop", type=int, default=25, help="decision every hop samples (25 = 0.5 s)")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)

    recs, _ = load_motionsense(ROOT / "data" / "motionsense")
    ckpt_dir = ROOT / "checkpoints" / args.tag
    n_classes = len(MS_ACTIVITY_NAMES)
    results = {"hop": args.hop, "models": {}}

    for name in args.models:
        files = sorted(ckpt_dir.glob(f"{name}_fold*.pt"))
        if not files:
            print(f"no checkpoints for {name} in {ckpt_dir}, skipping")
            continue
        per_subject = []
        for f in files:
            model, ck = load_ckpt(f)
            for subj in ck["test_subjects"]:
                rs = [r for r in recs if r.subject == subj]
                sig, lab = build_stream(rs, np.random.default_rng(subj))
                at, probs = causal_window_probs(model, sig, ck["window"], args.hop)
                per_subject.append({"subject": subj, "true": lab[at], "probs": probs})
        print(f"{name}: {len(per_subject)} subject streams, "
              f"{sum(len(p['true']) for p in per_subject)} decisions")

        res = {}
        for sname, params in SMOOTHERS.items():
            ms, all_pred, all_true = [], [], []
            for p in per_subject:
                pred = Smoother(n_classes, **params).run(p["probs"])
                ms.append(stream_metrics(pred, p["true"], args.hop))
                all_pred.append(pred)
                all_true.append(p["true"])
            cm = classification_metrics(np.concatenate(all_true), np.concatenate(all_pred), n_classes)
            res[sname] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            res[sname]["macro_f1"] = cm["macro_f1"]
            res[sname]["per_subject_acc"] = {int(p["subject"]): m["accuracy"]
                                             for p, m in zip(per_subject, ms)}
            r = res[sname]
            print(f"  {sname:5s} acc {r['accuracy']:.4f}  F1 {r['macro_f1']:.4f}  "
                  f"switches/min {r['switches_per_min']:.2f} (true {r['true_switches_per_min']:.2f})  "
                  f"median delay {r['median_delay_s']:.2f}s")

        if args.sweep:
            res["hmm_sweep"] = []
            for stay in [0.5, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995]:
                ms = [stream_metrics(Smoother(n_classes, "hmm", stay=stay).run(p["probs"]),
                                     p["true"], args.hop) for p in per_subject]
                row = {"stay": stay, **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]}}
                res["hmm_sweep"].append(row)
                print(f"  hmm stay={stay:<6} acc {row['accuracy']:.4f}  "
                      f"switches/min {row['switches_per_min']:.2f}  delay {row['median_delay_s']:.2f}s")
        results["models"][name] = res

        if name == args.models[0]:
            # dump one subject's first 10 minutes for the timeline figure
            p = per_subject[0]
            n = min(len(p["true"]), int(600 * 50 / args.hop))
            with open(ROOT / "results" / "timeline.json", "w") as f:
                json.dump({"subject": int(p["subject"]), "hop": args.hop, "class_names": MS_ACTIVITY_NAMES,
                           "true": p["true"][:n].tolist(),
                           "raw": Smoother(n_classes, "none").run(p["probs"][:n]).tolist(),
                           "hmm": Smoother(n_classes, **SMOOTHERS["hmm"]).run(p["probs"][:n]).tolist()},
                          f)

    with open(ROOT / "results" / f"stream_{args.tag}.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
