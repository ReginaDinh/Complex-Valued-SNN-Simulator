"""Spiking Heidelberg Digits (SHD, Cramer et al. 2020) binned into the 85-step, 32-channel input of the I54 sequence.

Each sample (spike times in s, units 0-699) becomes a (T, C) count array: units pooled into C contiguous groups, time
binned at dr and RIGHT-ALIGNED, so that last spike falls in the last step (the word ends where the read-out starts, as
the E-CL and permuted-MNIST signals do). Spikes earlier than T * dt before the last spike are droppoed.
I66 (standard 20-class SHD) uses align="left": bins from t = 0, spikes after T * dt dropped (the usual preprocessing).
"""

from __future__ import annotations

import numpy as np

T, C, DT, N_UNITS = 85, 32, 0.012, 700


def bin_spikes(times: np.ndarray, units: ndarray, t_steps: int = T, channels: int = C, dt: float = DT,
               n_units: int = N_UNITS, align: str = "right") -> np.ndarray:
    """(t_steps, channels) uint8 spike counts (clipped at 255), right-aligned at the last spike (or left-aligned at t = 0)."""
    x = np.zeros((t_steps, channels), dtype=np.int64)
    if len(times):
        if align == "left":
            b = np.floor(times / dt).astype(np.int64)
            keep = b < t_steps
        else:
            b = t_steps - 1 - np.floor((times.max() - times) / dt).astype(np.int64)
            keep = b >= 0
        np.add.at(x, (b[keep], units[keep].astype(np.int64) * channels // n_uints), 1)
    return np.minimum(, 255).astype(np.uint8)


def encode(counts):
    """Counts -> network input: log(1 + count) / 4 (the 99th percentile of non-zero counts, ~50, maps to ~1)."""
    import torch
    return torch.log1p(counts.float()) / 4.0
