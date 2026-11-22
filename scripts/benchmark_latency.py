"""Single-window CPU latency (batch size 1, one thread) - the number that
matters for on-device use. Exports each model to ONNX along the way.

    python scripts/benchmark_latency.py --tag main
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wearhar.export import check_onnx, export_onnx, quantize_int8  # noqa: E402
from wearhar.features import window_features  # noqa: E402
from wearhar.models import WithNormalization, build_model, n_params  # noqa: E402


def timeit(fn, n=300, warmup=30):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--models", nargs="+", default=["cnn", "convgru", "tcn", "transformer"])
    ap.add_argument("--channels", type=int, default=9)
    ap.add_argument("--window", type=int, default=128)
    args = ap.parse_args()

    import onnxruntime as ort

    torch.set_num_threads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1

    x = np.random.default_rng(0).normal(size=(1, args.channels, args.window)).astype(np.float32)
    xt = torch.from_numpy(x)
    export_dir = ROOT / "checkpoints" / "onnx"
    rows = {}

    for name in args.models:
        ck_path = ROOT / "checkpoints" / args.tag / f"{name}_fold0.pt"
        net = build_model(name, args.channels, 6)
        model = WithNormalization(net, np.zeros(args.channels), np.ones(args.channels))
        if ck_path.exists():
            model.load_state_dict(torch.load(ck_path, map_location="cpu", weights_only=False)["state_dict"])
        model.eval()

        with torch.no_grad():
            t_med, t_p95 = timeit(lambda m=model: m(xt))

        fp32 = export_onnx(model, args.channels, args.window, export_dir / f"{name}.onnx")
        err = check_onnx(model, fp32, args.channels, args.window)
        sess = ort.InferenceSession(str(fp32), so, providers=["CPUExecutionProvider"])
        o_med, o_p95 = timeit(lambda s=sess: s.run(None, {"imu_window": x}))

        int8 = quantize_int8(fp32, export_dir / f"{name}.int8.onnx")
        sess8 = ort.InferenceSession(str(int8), so, providers=["CPUExecutionProvider"])
        q_med, _ = timeit(lambda s=sess8: s.run(None, {"imu_window": x}))

        rows[name] = {
            "params": n_params(net),
            "torch_ms": t_med, "torch_p95_ms": t_p95,
            "onnx_ms": o_med, "onnx_p95_ms": o_p95, "onnx_int8_ms": q_med,
            "onnx_kb": fp32.stat().st_size / 1024, "onnx_int8_kb": int8.stat().st_size / 1024,
            "onnx_max_abs_err": err,
        }
        r = rows[name]
        print(f"{name:12s} {r['params']:>8d} params  torch {t_med:.2f} ms  onnx {o_med:.2f} ms  "
              f"int8 {q_med:.2f} ms  {r['onnx_kb']:.0f} KB -> {r['onnx_int8_kb']:.0f} KB  (err {err:.1e})")

    # random forest: feature extraction + predict on one window
    # (a forest fitted on random data has a similar size/depth to the real one; only timing matters)
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    Xf = window_features(rng.normal(size=(600, args.channels, args.window)).astype(np.float32))
    rf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=1, random_state=0)
    rf.fit(Xf, rng.integers(0, 6, 600))
    f_med, _ = timeit(lambda: window_features(x), n=200)
    p_med, _ = timeit(lambda: rf.predict(window_features(x)), n=100)
    rows["rf"] = {"features_ms": f_med, "total_ms": p_med}
    print(f"rf           features {f_med:.2f} ms, features+predict {p_med:.2f} ms")

    out = {"machine": platform.processor() or platform.machine(), "python": platform.python_version(),
           "torch": torch.__version__, "onnxruntime": ort.__version__, "threads": 1, "rows": rows}
    with open(ROOT / "results" / "latency.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
