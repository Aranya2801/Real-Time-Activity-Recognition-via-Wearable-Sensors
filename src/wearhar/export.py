"""Export a trained model to ONNX (fp32, optionally int8 dynamic-quantised)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def export_onnx(model: torch.nn.Module, n_channels: int, window: int, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, n_channels, window)
    torch.onnx.export(
        model.eval(), (dummy,), str(path),
        input_names=["imu_window"], output_names=["logits"],
        dynamic_axes={"imu_window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    return path


def quantize_int8(src: str | Path, dst: str | Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return Path(dst)


def check_onnx(model: torch.nn.Module, path: str | Path, n_channels: int, window: int,
               n: int = 64) -> float:
    """Max abs difference between torch and onnxruntime logits on random input."""
    import onnxruntime as ort

    x = np.random.default_rng(0).normal(size=(n, n_channels, window)).astype(np.float32)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"imu_window": x})[0]
    with torch.no_grad():
        torch_out = model.eval()(torch.from_numpy(x)).numpy()
    return float(np.abs(onnx_out - torch_out).max())
