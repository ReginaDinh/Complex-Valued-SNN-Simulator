"""Reproducibility and bookkeeping helpers."""

from __future__ import annotations

import json
import math
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "cpu") -> torch.device:
    """'auto' prefers cuda > mps > cpu. Small sequential models are usually fastest on cpu."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def git_state(repo: Path) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform()}


# Two-sided 95% Student-t critical values, df = 1..30
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145,
        2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci95(values: list[float]) -> tuple[float, float, float]:
    """(mean, sample std, half-width of the 95% confidence interval of the mean)."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0, float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    t = _T95[min(n - 1, 30) - 1] if n - 1 <= 30 else 1.96
    return m, sd, t * sd / math.sqrt(n)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
