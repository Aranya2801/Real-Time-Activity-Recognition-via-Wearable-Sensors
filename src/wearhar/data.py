"""Dataset loading and windowing.

Two datasets are supported:

* MotionSense (Malekzadeh et al., 2019) - iPhone 6s in the front trouser
  pocket, 24 subjects, 50 Hz, 6 activities. Each (activity, trial, subject)
  is one continuous recording, which is what makes it useful for the
  streaming experiments.
* UCI-HAR (Anguita et al., 2013) - Galaxy S2 on the waist, 30 subjects,
  50 Hz, already cut into 2.56 s windows with 50% overlap. We use the raw
  "Inertial Signals" and not the 561 pre-computed features.

Everything downstream works on a `Windows` object: X with shape
(n_windows, n_channels, window_len), integer labels, and per-window
subject / recording ids so we can do subject-wise splits and rebuild streams.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

MS_ACTIVITIES = ["dws", "ups", "wlk", "jog", "sit", "std"]
MS_ACTIVITY_NAMES = ["downstairs", "upstairs", "walking", "jogging", "sitting", "standing"]

# channel groups for MotionSense DeviceMotion csvs
MS_CHANNELS = {
    "acc": ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"],
    "gyro": ["rotationRate.x", "rotationRate.y", "rotationRate.z"],
    "grav": ["gravity.x", "gravity.y", "gravity.z"],
    "att": ["attitude.roll", "attitude.pitch", "attitude.yaw"],
}

UCI_ACTIVITY_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]
UCI_SIGNALS = {
    "acc": ["body_acc_x", "body_acc_y", "body_acc_z"],
    "gyro": ["body_gyro_x", "body_gyro_y", "body_gyro_z"],
    "total": ["total_acc_x", "total_acc_y", "total_acc_z"],
}

FS = 50  # both datasets are sampled at 50 Hz


@dataclass
class Recording:
    subject: int
    label: int
    trial: int
    signal: np.ndarray  # (T, C)


@dataclass
class Windows:
    X: np.ndarray  # (N, C, L) float32
    y: np.ndarray  # (N,) int64
    subject: np.ndarray  # (N,)
    recording: np.ndarray  # (N,) id of the source recording
    start: np.ndarray  # (N,) start sample inside the recording
    class_names: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, idx: np.ndarray) -> Windows:
        return Windows(
            self.X[idx], self.y[idx], self.subject[idx], self.recording[idx],
            self.start[idx], self.class_names, self.channels,
        )


# --------------------------------------------------------------------------
# MotionSense
# --------------------------------------------------------------------------

def load_motionsense(root: str | Path, sensors=("acc", "gyro", "grav")) -> tuple[list[Recording], list[str]]:
    """Read every DeviceMotion csv into a list of recordings."""
    root = Path(root)
    motion_dir = root / "A_DeviceMotion_data"
    if not motion_dir.exists():
        raise FileNotFoundError(
            f"{motion_dir} not found - run `python scripts/download_data.py motionsense` first"
        )

    cols = [c for s in sensors for c in MS_CHANNELS[s]]
    recordings = []
    for folder in sorted(motion_dir.iterdir()):
        m = re.fullmatch(r"([a-z]{3})_(\d+)", folder.name)
        if not m:
            continue
        act, trial = m.group(1), int(m.group(2))
        label = MS_ACTIVITIES.index(act)
        for csv in sorted(folder.glob("sub_*.csv")):
            subject = int(csv.stem.split("_")[1])
            df = pd.read_csv(csv, usecols=cols)
            recordings.append(Recording(subject, label, trial, df[cols].to_numpy(np.float32)))
    if not recordings:
        raise RuntimeError(f"no recordings found under {motion_dir}")
    return recordings, cols


def sliding_windows(recordings: list[Recording], win: int, hop: int,
                    class_names: list[str], channels: list[str]) -> Windows:
    Xs, ys, subs, recs, starts = [], [], [], [], []
    for rid, r in enumerate(recordings):
        T = len(r.signal)
        if T < win:
            continue
        idx = np.arange(0, T - win + 1, hop)
        # (n, win, C) via stride tricks, then copy into channels-first
        view = np.lib.stride_tricks.sliding_window_view(r.signal, win, axis=0)[idx]
        Xs.append(view.astype(np.float32))  # already (n, C, win)
        ys.append(np.full(len(idx), r.label))
        subs.append(np.full(len(idx), r.subject))
        recs.append(np.full(len(idx), rid))
        starts.append(idx)
    return Windows(
        np.concatenate(Xs), np.concatenate(ys).astype(np.int64), np.concatenate(subs),
        np.concatenate(recs), np.concatenate(starts), class_names, channels,
    )


def motionsense_windows(root, win: int = 128, hop: int = 64,
                        sensors=("acc", "gyro", "grav")) -> tuple[Windows, list[Recording]]:
    recs, cols = load_motionsense(root, sensors)
    return sliding_windows(recs, win, hop, MS_ACTIVITY_NAMES, cols), recs


# --------------------------------------------------------------------------
# UCI-HAR
# --------------------------------------------------------------------------

def _read_txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float32)


def uci_har_windows(root: str | Path, sensors=("acc", "gyro", "total")) -> Windows:
    """UCI-HAR comes pre-windowed (128 samples, 50% overlap).

    The official train/test split is already subject-disjoint (21 vs 9
    subjects), so we keep both parts together and let splits.py decide.
    There is no recording id in the release, so subject is used for it.
    """
    root = Path(root)
    base = root / "UCI HAR Dataset"
    if not base.exists():
        raise FileNotFoundError(f"{base} not found - run `python scripts/download_data.py uci_har`")

    names = [s for g in sensors for s in UCI_SIGNALS[g]]
    parts = []
    for split in ("train", "test"):
        sig = np.stack(
            [_read_txt(base / split / "Inertial Signals" / f"{n}_{split}.txt") for n in names], axis=1
        )
        y = _read_txt(base / split / f"y_{split}.txt").astype(np.int64) - 1
        s = _read_txt(base / split / f"subject_{split}.txt").astype(np.int64)
        parts.append((sig, y, s))
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    s = np.concatenate([p[2] for p in parts])
    return Windows(X, y, s, s.copy(), np.zeros_like(y), UCI_ACTIVITY_NAMES, names)


def load_windows(dataset: str, data_dir: str | Path = "data", **kw) -> Windows:
    data_dir = Path(data_dir)
    if dataset == "motionsense":
        return motionsense_windows(data_dir / "motionsense", **kw)[0]
    if dataset == "uci_har":
        return uci_har_windows(data_dir / "uci_har", **{k: v for k, v in kw.items() if k == "sensors"})
    raise ValueError(f"unknown dataset {dataset!r}")


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

class ChannelScaler:
    """Per-channel z-score. Fit on training windows only."""

    def fit(self, X: np.ndarray) -> ChannelScaler:
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean_.ravel().tolist(), "std": self.std_.ravel().tolist()}

    @classmethod
    def from_state(cls, d: dict) -> ChannelScaler:
        s = cls()
        s.mean_ = np.asarray(d["mean"], np.float32).reshape(1, -1, 1)
        s.std_ = np.asarray(d["std"], np.float32).reshape(1, -1, 1)
        return s
