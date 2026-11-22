"""H6d: quantitative properties of oscillator frequency sets, without training.

    cd research && .venv/bin/python -m analysis.freq_metrics

Random continuous frequencies are almost surely independent over Q, so "independent over Q" alone cannot
separate good from bad sets. These metrics are finite-horizon / finite-precision versions of it, for N
oscillators with frequencies omega_k in [lo, hi] (the range used by the networks, rad/step):

* distinct           : number of different frequencies (rounding to a hardware grid can merge neurons)
* exact relations    : integer vectors n with |n|_1 <= 4 and sum n_k omega_k = 0 (mod 2 pi) exactly, i.e. exact
                       low-order resonances (beyond the trivial ones); with duplicates, omega_a - omega_b = 0
                       counts too
* resonance gap M    : min over |n|_1 = M (M = 2, 3, 4) of the distance of sum n_k omega_k to the nearest
                       multiple of 2 pi: how close the set comes to a resonance of order M
* additive energy    : #{(a, b, c, d): omega_a - omega_b = omega_c - omega_d}, divided by its minimum N(2N - 1)
                       (= 1 for a Sidon set / Golomb ruler)
* crosstalk s(D)     : |mean_k e^{j D omega_k}|, similarity of the codes of two events D steps apart (1 =
                       indistinguishable); max over D bands, max inside one period (b-bit grid: 100 <= D <= 2^b - 100;
                       continuous sets: the 12-bit window), and the first D >= 100 with s(D) >= 0.9 / 0.99
                       (near-recurrence time; a b-bit grid repeats exactly at D = 2^b)

Limits: s(D) weights all oscillators equally and ignores decay (|beta| < 1) and learned read-out weights; it
is a proxy. Whether these properties matter for a trained network is tested in experiments.eh6 (suite freq).
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from cvsnn.frequencies import additive_energy, frequency_set, grid_range

OUT = Path(__file__).resolve().parents[1] / "results" / "freq_metrics"
TWO_PI = 2 * math.pi
N, LO, HI = 32, 0.01, 1.5
MAX_D = 1_000_000
TOL = 1e-9


def _dist_2pi(x: np.ndarray) -> np.ndarray:
    return np.abs(x - TWO_PI * np.round(x / TWO_PI))


def _signed(vals: list[np.ndarray]) -> list[np.ndarray]:
    """All sign patterns of a sum of terms, the first term's sign fixed to + (n and -n are the same relation)."""
    out = []
    for signs in itertools.product((1, -1), repeat=len(vals) - 1):
        out.append(vals[0] + sum(s * v for s, v in zip(signs, vals[1:])))
    return out


def combos(w: np.ndarray, order: int) -> np.ndarray:
    """All sums sum n_k w_k over integer vectors with |n|_1 = order on distinct neurons (one per +-n)."""
    n = len(w)
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    ordered = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    triples = np.array(list(itertools.combinations(range(n), 3)))
    out = [order * w]                                                       # coefficient pattern (M,)
    if order == 2:
        out += _signed([w[pairs[:, 0]], w[pairs[:, 1]]])                    # (1, 1)
    elif order == 3:
        out += _signed([2 * w[ordered[:, 0]], w[ordered[:, 1]]])            # (2, 1)
        out += _signed([w[triples[:, 0]], w[triples[:, 1]], w[triples[:, 2]]])  # (1, 1, 1)
    elif order == 4:
        out += _signed([3 * w[ordered[:, 0]], w[ordered[:, 1]]])            # (3, 1)
        out += _signed([2 * w[pairs[:, 0]], 2 * w[pairs[:, 1]]])            # (2, 2)
        for d in range(3):                                                  # (2, 1, 1): which neuron is doubled
            rest = [c for c in range(3) if c != d]
            out += _signed([2 * w[triples[:, d]], w[triples[:, rest[0]]], w[triples[:, rest[1]]]])
        quads = np.array(list(itertools.combinations(range(n), 4)))
        out += _signed([w[quads[:, c]] for c in range(4)])                  # (1, 1, 1, 1)
    else:
        raise ValueError("order must be 2, 3 or 4")
    return np.concatenate(out)


def crosstalk(w: np.ndarray, max_d: int = MAX_D) -> np.ndarray:
    s = np.empty(max_d)
    chunk = 100_000
    for a in range(0, max_d, chunk):
        d = np.arange(a + 1, min(max_d, a + chunk) + 1)[:, None]
        s[a:a + d.shape[0]] = np.abs(np.exp(1j * d * w[None, :]).mean(1))
    return s


def additive_energy_real(w: np.ndarray) -> int:
    """Additive energy with a tolerance (exact repeated differences among real frequencies)."""
    d = np.round((w[:, None] - w[None, :]).ravel() / TOL).astype(np.int64)
    _, counts = np.unique(d, return_counts=True)
    return int((counts.astype(np.int64) ** 2).sum())


def metrics(w: np.ndarray, bits: int | None = None) -> dict:
    w = np.asarray(w, dtype=np.float64)
    res = {"distinct": int(len(np.unique(np.round(w / TOL)))), "min": float(w.min()), "max": float(w.max())}
    exact = 0
    for M in (2, 3, 4):
        d = _dist_2pi(combos(w, M))
        res[f"gap_order{M}"] = float(d.min())
        exact += int((d < 1e-7).sum())
    res["exact_relations_order_le4"] = exact
    res["additive_energy_ratio"] = additive_energy_real(w) / (len(w) * (2 * len(w) - 1))
    s = crosstalk(w)
    for lo, hi in ((100, 10_000), (10_000, MAX_D)):
        res[f"crosstalk_max_D{lo}-{hi}"] = float(s[lo - 1:hi].max())
    # inside one period of a b-bit grid (it repeats exactly at D = 2^b); continuous sets: the 12-bit window
    hi_d = (2**bits if bits else 2**12) - 100
    res["crosstalk_max_within_period"] = float(s[99:hi_d].max())
    for thr in (0.9, 0.99):   # from D = 100 on: below that every set in this range is similar (s(1) ~ 0.9)
        hit = np.nonzero(s[99:] >= thr)[0]
        res[f"first_D_crosstalk>={thr}"] = int(hit[0] + 100) if len(hit) else None
    return res


def main() -> None:
    import torch

    OUT.mkdir(parents=True, exist_ok=True)
    sets: dict[str, list[np.ndarray]] = {}
    for kind in ("golden", "independent", "geometric", "harmonic"):
        sets[kind] = [frequency_set(kind, N, LO, HI).numpy()]
    for kind in ("uniform", "log"):
        sets[kind] = [frequency_set(kind, N, LO, HI, generator=torch.Generator().manual_seed(s)).numpy() for s in range(10)]
    for b in (8, 10, 12):
        sets[f"quantized_b{b}"] = [frequency_set("quantized", N, LO, HI, bits=b, generator=torch.Generator().manual_seed(s)).numpy() for s in range(10)]
        sets[f"sidon_b{b}"] = [frequency_set("sidon", N, LO, HI, bits=b).numpy()]
        # random distinct grid points: separates "distinct" from "few repeated differences"
        m_lo, m_hi = grid_range(LO, HI, b)
        rng = np.random.default_rng(0)
        sets[f"random_grid_b{b}"] = [TWO_PI * rng.choice(np.arange(m_lo, m_hi + 1), N, replace=False) / 2**b for _ in range(10)]
    results = {}
    for name, ws in sets.items():
        b = int(name.split("_b")[-1]) if "_b" in name else None
        rows = [metrics(w, b) for w in ws]
        agg = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            if all(v is None for v in vals):
                agg[k] = None
            elif any(v is None for v in vals):
                agg[k] = f"{sum(v is not None for v in vals)}/{len(vals)} draws hit, median {sorted(v for v in vals if v is not None)[len([v for v in vals if v is not None]) // 2]}"
            else:
                agg[k] = float(np.mean(vals)) if len(vals) > 1 else vals[0]
        agg["draws"] = len(rows)
        results[name] = agg
        print(name, json.dumps(agg), flush=True)
    meta = {"N": N, "range_rad_per_step": [LO, HI], "max_D": MAX_D,
            "note": "random sets: mean over draws (first-D columns: how many draws reach the level, median D)"}
    (OUT / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    cols = ["distinct", "exact_relations_order_le4", "gap_order2", "gap_order3", "gap_order4", "additive_energy_ratio",
            "crosstalk_max_within_period", "crosstalk_max_D100-10000", "crosstalk_max_D10000-1000000",
            "first_D_crosstalk>=0.9", "first_D_crosstalk>=0.99"]
    lines = [f"# Frequency-set metrics (N = {N}, omega in [{LO}, {HI}] rad/step, D up to {MAX_D:,})", "",
             "| set | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]

    def fmt(v):
        if v is None:
            return "never"
        if isinstance(v, str):
            return v
        if isinstance(v, int) or (isinstance(v, float) and v.is_integer() and v > 10):
            return f"{int(v):,}"
        return f"{v:.3g}"

    for name, agg in results.items():
        lines.append(f"| {name} | " + " | ".join(fmt(agg[c]) for c in cols) + " |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
                                                                                             
  
