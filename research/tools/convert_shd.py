"""Convert the SHD .h5 files into research/data/shd.npz (binned counts, see cvsnn/data/shd.py).

    cd research && .venv/bin/python -I tools/convert_shd.py data/shd_raw data/shd.npz
    cd research && .venv/bin/python -I tools/convert_shd.py data/shd_raw data/shd140.npz --channels 140 --dt 0.01 --steps 100 --align left
                                        (166: standard 20-class SHD, 140 channels, 10 ms bins from t = 0)

The raw folder holds shd_train.h5 and shd_test.h5 (unzipped from zenkelab.org/datasets/shd/, MD5 checked against its
md5sums.txt). Needs h5py (dependency group "data" in pyproject.toml).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from cvsnn.data.shd import C, DT, T, bin_spikes


def main(raw: Path, out: Path, channels: int = C, dt: float = DT, steps: int = T, align: str = "right") -> None:
    arrays = {}
    for split in ("train", "test"):
        with h5py.File(raw / f"shd_{split}.h5", "r") as f:
            times, units = f["spikes/times"], f["spikes/units"]
            arrays[f"x_{split}"] = np.stack([bin_spikes(times[i], units[i], steps, channels, dt, align=align) for i in range(len(times))])
            arrays[f"y_{split}"] = f["labels"][:].astype(np.int64)
            arrays[f"speaker_{split}"] = f["extra/speaker"][:].astype(np.int64)
            arrays["keys"] = np.array([k.decode() for k in f["extra/keys"][:]])
        print(split, arrays[f"x_{split}"].shape, np.bincount(arrays[f"y_{split}"]))
    np.savez_compressed(out, **arrays, T=steps, C=channels, dt=dt, align=align)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--channels", type=int, default=C)
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--steps", type=int, default=T)
    ap.add_argument("--align", default="right", choices=("right", "left"))
    a = ap.parse_args()
    main(Path(a.raw), Path(a.out), a.channels, a.dt, a.steps, a.align)
