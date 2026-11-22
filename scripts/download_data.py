"""Download the datasets into ./data

    python scripts/download_data.py motionsense
    python scripts/download_data.py uci_har

MotionSense comes from the authors' GitHub repo, UCI-HAR from the UCI
archive. If a download is blocked (corporate proxy etc.) grab the zip
manually and pass it with --zip.
"""

from __future__ import annotations

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MS_BASE = "https://github.com/mmalekzadeh/motion-sense/raw/master/data/"
UCI_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

SOURCES = {
    "motionsense": [
        (MS_BASE + "A_DeviceMotion_data.zip", True),
        (MS_BASE + "data_subjects_info.csv", False),
    ],
    "uci_har": [(UCI_URL, True)],
}


def fetch(url: str) -> bytes:
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for m in zf.infolist():
        if m.filename.startswith("__MACOSX"):
            continue
        target = (dest / m.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f"unsafe path in zip: {m.filename}")
        zf.extract(m, dest)


def extract_bytes(data: bytes, dest: Path) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        safe_extract(zf, dest)
        # UCI-HAR is a zip containing another zip
        for name in zf.namelist():
            if name.endswith(".zip") and not name.startswith("__MACOSX"):
                inner = dest / name
                with zipfile.ZipFile(inner) as z2:
                    safe_extract(z2, dest)
                inner.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(SOURCES))
    ap.add_argument("--zip", help="use a manually downloaded zip instead")
    args = ap.parse_args()

    dest = ROOT / "data" / args.dataset
    dest.mkdir(parents=True, exist_ok=True)

    if args.zip:
        extract_bytes(Path(args.zip).read_bytes(), dest)
    else:
        for url, is_zip in SOURCES[args.dataset]:
            data = fetch(url)
            if is_zip:
                extract_bytes(data, dest)
            else:
                (dest / url.rsplit("/", 1)[1]).write_bytes(data)
    shutil.rmtree(dest / "__MACOSX", ignore_errors=True)
    print(f"done -> {dest}")


if __name__ == "__main__":
    main()
