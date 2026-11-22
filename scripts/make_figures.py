"""Turn results/*.json into the figures in figures/ and the markdown tables
in results/tables.md (which is what the README tables are copied from)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# categorical palette (fixed order) + a single blue ramp for magnitudes
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})

NICE = {"rf": "Random forest\n(features)", "cnn": "1D-CNN", "convgru": "Conv-GRU",
        "tcn": "TCN", "transformer": "Patch\nTransformer"}


def load(tag):
    p = RES / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("wrote", FIG / name)


def fmt(m, s):
    return f"{100 * m:.1f} ± {100 * s:.1f}"


def main():
    lines = []
    main_res = load("main")
    lat = json.loads((RES / "latency.json").read_text()) if (RES / "latency.json").exists() else None

    # ---------------- main table + figure ----------------
    if main_res:
        S = main_res["summary"]
        order = sorted(S, key=lambda k: S[k]["f1_mean"])
        lines += ["## Main results (5-fold leave-subjects-out, MotionSense)", "",
                  "| Model | Params | Accuracy (%) | Macro-F1 (%) | CPU latency, 1 window (ms) |",
                  "|---|---:|---:|---:|---:|"]
        for k in reversed(order):
            s = S[k]
            p = f"{s['params'] / 1000:.0f}k" if s["params"] else "-"
            if lat and k in lat["rows"]:
                r = lat["rows"][k]
                ms = f"{r['onnx_ms']:.2f}" if "onnx_ms" in r else f"{r['total_ms']:.2f}"
            else:
                ms = "-"
            lines.append(f"| {k} | {p} | {fmt(s['acc_mean'], s['acc_std'])} | "
                         f"{fmt(s['f1_mean'], s['f1_std'])} | {ms} |")
        lines.append("")

        fig, ax = plt.subplots(figsize=(6.4, 3.2))
        y = np.arange(len(order))
        f1 = [100 * S[k]["f1_mean"] for k in order]
        sd = [100 * S[k]["f1_std"] for k in order]
        ax.barh(y, f1, xerr=sd, color=C[0], height=0.55, error_kw=dict(ecolor=INK2, lw=1, capsize=3))
        for yi, v, e in zip(y, f1, sd):
            ax.text(v + e + 0.4, yi, f"{v:.1f}", va="center", color=INK, fontsize=9)
        ax.set_yticks(y, [NICE[k] for k in order])
        lo = max(0, min(f1) - max(sd) - 6)
        ax.set_xlim(lo, 100)
        ax.set_xlabel("macro-F1 on unseen subjects (%), mean ± std over 5 folds")
        ax.grid(axis="y", visible=False)
        ax.set_title("Model comparison")
        save(fig, "models.png")

        # confusion for the best deep model
        best = max((k for k in S if k != "rf"), key=lambda k: S[k]["f1_mean"])
        cm = np.array(S[best]["confusion_sum"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        names = main_res["class_names"]
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"]), vmin=0, vmax=1)
        for i in range(len(names)):
            for j in range(len(names)):
                if cmn[i, j] >= 0.005:
                    ax.text(j, i, f"{100 * cmn[i, j]:.0f}", ha="center", va="center", fontsize=9,
                            color="white" if cmn[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.grid(False)
        ax.set_title(f"{NICE[best].replace(chr(10), ' ')} - row-normalised confusion (%)")
        save(fig, "confusion.png")

        lines += [f"Per-class F1 ({best}): " + ", ".join(
            f"{n} {100 * f:.1f}" for n, f in zip(names, S[best]["per_class_f1_mean"])), ""]

    # ---------------- leakage ----------------
    leak = load("leak_random")
    if leak and main_res:
        lines += ["## Random-window split vs subject split", "",
                  "| Model | Random split F1 (%) | Subject split F1 (%) | Gap (pts) |",
                  "|---|---:|---:|---:|"]
        models = list(leak["summary"])
        for k in models:
            a, b = leak["summary"][k], main_res["summary"][k]
            lines.append(f"| {k} | {fmt(a['f1_mean'], a['f1_std'])} | {fmt(b['f1_mean'], b['f1_std'])} | "
                         f"{100 * (a['f1_mean'] - b['f1_mean']):.1f} |")
        lines.append("")
        fig, ax = plt.subplots(figsize=(5.2, 3.0))
        x = np.arange(len(models))
        w = 0.36
        for i, (res, lab) in enumerate([(leak, "random window split"), (main_res, "leave-subjects-out")]):
            v = [100 * res["summary"][k]["f1_mean"] for k in models]
            ax.bar(x + (i - 0.5) * (w + 0.02), v, w, color=C[i], label=lab)
            for xi, vi in zip(x, v):
                ax.text(xi + (i - 0.5) * (w + 0.02), vi + 0.3, f"{vi:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x, [NICE[k].replace("\n", " ") for k in models])
        allv = [100 * r["summary"][k]["f1_mean"] for r in (leak, main_res) for k in models]
        ax.set_ylim(min(allv) - 5, 101)
        ax.set_ylabel("macro-F1 (%)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2)
        ax.set_title("How much a random split overestimates", pad=24)
        save(fig, "leakage.png")

    # ---------------- window length ----------------
    wins = []
    for tag, w in [("win32", 32), ("win64", 64), ("main", 128), ("win256", 256)]:
        r = load(tag)
        if r and "cnn" in r["summary"]:
            wins.append((w, r["summary"]["cnn"]))
    if len(wins) > 1:
        lines += ["## Window length (1D-CNN)", "", "| Window | Seconds | Macro-F1 (%) |", "|---:|---:|---:|"]
        for w, s in wins:
            lines.append(f"| {w} | {w / 50:.2f} | {fmt(s['f1_mean'], s['f1_std'])} |")
        lines.append("")
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        xs = [w / 50 for w, _ in wins]
        ys = [100 * s["f1_mean"] for _, s in wins]
        es = [100 * s["f1_std"] for _, s in wins]
        ax.errorbar(xs, ys, yerr=es, color=C[0], lw=2, marker="o", ms=7, capsize=3, ecolor=INK2)
        for xv, yv in zip(xs, ys):
            ax.annotate(f"{yv:.1f}", (xv, yv), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [f"{x:.2f}" for x in xs])
        ax.minorticks_off()
        ax.set_xlabel("window length (s) = minimum delay before a decision")
        ax.set_ylabel("macro-F1 (%)")
        ax.set_title("Longer windows help, but you wait longer")
        save(fig, "window_length.png")

    # ---------------- sensors / augmentation ----------------
    abl = [("sens_acc", "acc only"), ("sens_acc_gyro", "acc + gyro"), ("main", "acc + gyro + gravity"),
           ("sens_all", "acc + gyro + gravity + attitude"), ("aug_none", "no augmentation"),
           ("aug_norot", "jitter + scale, no rotation"), ("main", "jitter + scale + rotation (default)")]
    rows = [(lab, load(t)["summary"]["cnn"]) for t, lab in abl if load(t) and "cnn" in load(t)["summary"]]
    if rows:
        lines += ["## Ablations (1D-CNN, 5-fold leave-subjects-out)", "", "| Setting | Macro-F1 (%) |",
                  "|---|---:|"]
        lines += [f"| {lab} | {fmt(s['f1_mean'], s['f1_std'])} |" for lab, s in rows]
        lines.append("")

    # ---------------- streaming ----------------
    st = load("stream_main")
    if st:
        lines += ["## Streaming (decision every 0.5 s, causal smoothing)", "",
                  "| Model | Smoother | Accuracy (%) | Macro-F1 (%) | Label switches / min | "
                  "Median delay after a change (s) |", "|---|---|---:|---:|---:|---:|"]
        true_sw = None
        for m, res in st["models"].items():
            for sm in ("none", "ema", "vote", "hmm"):
                r = res[sm]
                true_sw = r["true_switches_per_min"]
                lines.append(f"| {m} | {sm} | {100 * r['accuracy']:.1f} | {100 * r['macro_f1']:.1f} | "
                             f"{r['switches_per_min']:.2f} | {r['median_delay_s']:.1f} |")
        lines += ["", f"(ground truth has {true_sw:.2f} activity changes per minute in these streams)", ""]

        m0 = "cnn" if "cnn" in st["models"] else next(iter(st["models"]))
        sweep = st["models"][m0].get("hmm_sweep")
        if sweep:
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
            stay = [r["stay"] for r in sweep]
            xlab = [str(s) for s in stay]
            axes[0].plot(range(len(stay)), [100 * r["accuracy"] for r in sweep], color=C[0], lw=2,
                         marker="o", ms=7)
            axes[0].set_ylabel("stream accuracy (%)")
            axes[1].plot(range(len(stay)), [r["switches_per_min"] for r in sweep], color=C[0], lw=2,
                         marker="o", ms=7)
            axes[1].axhline(sweep[0]["true_switches_per_min"], color=INK2, lw=1, ls="--")
            axes[1].text(0, sweep[0]["true_switches_per_min"], " true rate", va="bottom", fontsize=8.5,
                         color=INK2)
            axes[1].set_ylabel("predicted label switches / min")
            axes[1].set_ylim(0, None)
            for a in axes:
                a.set_xticks(range(len(stay)), xlab)
                a.set_xlabel("HMM self-transition probability")
            fig.suptitle(f"HMM smoothing trade-off ({NICE[m0]})", fontweight="bold", fontsize=11)
            save(fig, "hmm_sweep.png")

        ps = st["models"][m0]["hmm"].get("per_subject_acc")
        if ps:
            subj = sorted(ps, key=lambda k: ps[k])
            v = [100 * ps[k] for k in subj]
            fig, ax = plt.subplots(figsize=(8.0, 2.8))
            ax.bar(range(len(subj)), v, color=C[0], width=0.7)
            ax.axhline(np.mean(v), color=INK2, lw=1, ls="--")
            ax.text(-0.4, np.mean(v) + 0.6, f"mean {np.mean(v):.1f}", ha="left", fontsize=8.5,
                    color=INK2)
            ax.set_xticks(range(len(subj)), subj)
            ax.set_xlabel("held-out subject id")
            ax.set_ylabel("stream accuracy (%)")
            ax.set_ylim(min(v) - 5, 100.5)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"Per-subject accuracy ({NICE[m0]} + HMM), every subject unseen in training")
            save(fig, "per_subject.png")
            lines += [f"Per-subject stream accuracy ({m0} + hmm): min {min(v):.1f} (subject {subj[0]}), "
                      f"max {max(v):.1f}, median {np.median(v):.1f}", ""]

    tl = RES / "timeline.json"
    if tl.exists():
        d = json.loads(tl.read_text())
        names = d["class_names"]
        fig, ax = plt.subplots(figsize=(9, 2.3))
        rows_ = [("ground truth", d["true"]), ("raw model", d["raw"]), ("+ HMM smoothing", d["hmm"])]
        t = np.arange(len(d["true"])) * d["hop"] / 50 / 60
        dt = t[1] - t[0]
        for i, (_, seq) in enumerate(rows_):
            seq = np.asarray(seq)
            starts = np.r_[0, np.flatnonzero(np.diff(seq)) + 1]
            ends = np.r_[starts[1:], len(seq)]
            ax.broken_barh([(t[s], (e - s) * dt) for s, e in zip(starts, ends)], (2 - i - 0.35, 0.7),
                           facecolors=[C[seq[s]] for s in starts])
        ax.set_yticks([2, 1, 0], [r[0] for r in rows_])
        ax.set_xlabel("time (min)")
        ax.grid(False)
        ax.set_xlim(0, t[-1] + dt)
        handles = [plt.Rectangle((0, 0), 1, 1, color=C[i]) for i in range(len(names))]
        ax.legend(handles, names, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=8.5)
        ax.set_title(f"Held-out subject {d['subject']}, first {t[-1]:.0f} minutes", pad=26)
        save(fig, "stream_timeline.png")

    (RES / "tables.md").write_text("\n".join(lines) + "\n")
    print("wrote", RES / "tables.md")


if __name__ == "__main__":
    main()
