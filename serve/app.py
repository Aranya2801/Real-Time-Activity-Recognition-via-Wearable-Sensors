"""Minimal WebSocket inference server.

    uvicorn serve.app:app --port 8000
    # env: WEARHAR_MODEL=checkpoints/onnx/cnn.onnx (default)

The client streams raw samples; the server keeps one ring buffer + HMM
smoother per connection and answers every `hop` samples.

Message from client (one or more samples, 9 values each, in the training
channel order: userAcc xyz, rotationRate xyz, gravity xyz):

    {"samples": [[ax, ay, az, gx, gy, gz, grx, gry, grz], ...]}

Reply (only when a decision was made):

    {"activity": "walking", "confidence": 0.97, "probs": {...}, "t": 1234}

Uses onnxruntime only, so it does not need torch at serve time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wearhar.data import MS_ACTIVITY_NAMES  # noqa: E402
from wearhar.smoothing import Smoother  # noqa: E402

MODEL_PATH = os.environ.get("WEARHAR_MODEL", str(ROOT / "checkpoints" / "onnx" / "cnn.onnx"))
WINDOW = int(os.environ.get("WEARHAR_WINDOW", 128))
HOP = int(os.environ.get("WEARHAR_HOP", 25))
N_CH = 9

app = FastAPI(title="wearhar")
_session: ort.InferenceSession | None = None


def session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


@app.get("/health")
def health():
    return {"status": "ok", "model": Path(MODEL_PATH).name, "window": WINDOW, "hop": HOP}


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    buf = np.zeros((WINDOW, N_CH), np.float32)
    filled = since = t = 0
    smoother = Smoother(len(MS_ACTIVITY_NAMES), "hmm", stay=0.97)
    sess = session()
    try:
        while True:
            msg = await ws.receive_json()
            samples = np.asarray(msg.get("samples", []), np.float32).reshape(-1, N_CH)
            for s in samples:
                buf = np.roll(buf, -1, axis=0)
                buf[-1] = s
                filled = min(filled + 1, WINDOW)
                since += 1
                t += 1
                if filled == WINDOW and since >= HOP:
                    since = 0
                    logits = sess.run(None, {"imu_window": buf.T[None].copy()})[0][0]
                    label, belief = smoother.update(softmax(logits))
                    await ws.send_json({
                        "t": t,
                        "activity": MS_ACTIVITY_NAMES[label],
                        "confidence": round(float(belief[label]), 4),
                        "probs": {n: round(float(p), 4) for n, p in zip(MS_ACTIVITY_NAMES, belief)},
                    })
    except WebSocketDisconnect:
        pass
