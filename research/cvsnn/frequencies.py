"""Frequency sets for oscillator neurons (hypothesis H6): how the rotation angles omega_k (rad/step) of
N neurons are chosen. Every set lies in [lo, hi].

* uniform      : random, U(lo, hi)
* log          : random, log-uniform (as in REACT; our default heterogeneous initialisation)
* golden       : lo + (hi - lo) frac(k / phi), k = 1..N: the golden-angle sequence, built from one irrational
                 number. frac(k / phi) = k / phi - floor(k / phi), so any integer vector n with sum n_k = 0 and
                 sum n_k k = 0 gives sum n_k omega_k = -(hi - lo) sum n_k floor(k / phi): an exact relation, e.g.
                 omega_1 - omega_2 - omega_4 + omega_5 = 0 (a repeated pairwise difference).
* independent  : lo + (hi - lo) frac(sqrt(p_k)), p_k the k-th prime. Square roots of distinct primes are linearly
                 independent over Q (Besicovitch 1940); with rational lo and hi - lo there is no integer relation
                 among the omega_k and 2 pi (pi is transcendental).
* geometric    : hi (lo / hi)^{k / (N - 1)}: a RoPE-style geometric ladder
* harmonic     : equally spaced from lo to hi: rationally dependent (omega_1 - 2 omega_2 + omega_3 = 0) and the
                 joint code repeats after about 2 pi (N - 1) / (hi - lo) steps
* quantized    : log-uniform, rounded to a b-bit phase register omega = 2 pi m / 2^b. On such hardware every
                 frequency is rational, so the joint code repeats exactly after 2^b steps; rounding can also
                 make neurons share a frequency.
* zero         : omega = 0 for every neuron (no rotation: a leaky integrator; outside [lo, hi] on purpose)
* sidon        : N distinct grid points m in the same range chosen to minimise the additive energy
                 #{(a, b, c, d): m_a - m_b = m_c - m_d}. A set where all pairwise differences differ (a Sidon set,
                 Golomb ruler) has no exact resonance omega_a - omega_b = omega_c - omega_d or omega_a + omega_b =
                 2 omega_c, and minimising the additive energy flattens the code crosstalk spectrum (it equals
                 the 4th moment of |mean_k e^{j D omega_k}| over D, by Parseval).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

PHI = (1 + 5**0.5) / 2
FREQ_SETS = ("uniform", "log", "golden", "independent", "geometric", "harmonic", "quantized", "sidon", "zero", "phi_ladder", "ladder2", "banded_log", "brain5", "ladder5", "phi_jitter", "banded_strat")


def primes(n: int) -> list[int]:
    out, c = [], 2
    while len(out) < n:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out


def grid_range(lo: float, hi: float, bits: int) -> tuple[int, int]:
    """Smallest and largest m with 2 pi m / 2^bits inside [lo, hi] (m >= 1: m = 0 does not rotate)."""
    P = 2**bits
    m_lo = max(1, math.ceil(lo * P / (2 * math.pi) - 1e-9))
    m_hi = math.floor(hi * P / (2 * math.pi) + 1e-9)
    if m_hi < m_lo:
        raise ValueError(f"no {bits}-bit grid frequency in [{lo}, {hi}]")
    return m_lo, m_hi


def additive_energy(m: np.ndarray) -> int:
    """#{(a, b, c, d): m_a - m_b = m_c - m_d} (ordered quadruples, a = b allowed). Minimum N(2N - 1) for N
    distinct points, reached exactly by Sidon sets."""
    d = (m[:, None] - m[None, :]).ravel()
    _, counts = np.unique(d, return_counts=True)
    return int((counts.astype(np.int64) ** 2).sum())


@lru_cache(maxsize=None)
def sidon_grid(n: int, m_lo: int, m_hi: int, seed: int = 0, sweeps: int = 30) -> tuple[int, ...]:
    """n distinct integers in [m_lo, m_hi] with (locally) minimal additive energy: greedy construction that adds
    the point creating the fewest repeated differences, then single-point swaps while they lower the energy."""
    cand = np.arange(m_lo, m_hi + 1)
    if len(cand) < n:
        raise ValueError(f"only {len(cand)} grid points in range for {n} neurons")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.choice(cand))]
    while len(chosen) < n:
        cur = np.array(chosen)
        best, best_e = [], None
        for c in cand:
            if c in chosen:
                continue
            e = additive_energy(np.append(cur, c))
            if best_e is None or e < best_e:
                best, best_e = [int(c)], e
            elif e == best_e:
                best.append(int(c))
        chosen.append(int(rng.choice(best)))
    cur = np.array(sorted(chosen))
    e_cur = additive_energy(cur)
    for _ in range(sweeps):
        improved = False
        for i in rng.permutation(n):
            for c in rng.permutation(cand):
                if c in cur:
                    continue
                trial = cur.copy()
                trial[i] = c
                e = additive_energy(trial)
                if e < e_cur:
                    cur, e_cur, improved = np.sort(trial), e, True
                    break
        if not improved:
            break
    return tuple(int(v) for v in cur)


def frequency_set(kind: str, n: int, lo: float, hi: float, *, bits: int | None = None,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    """n frequencies (rad/step, float64) of the given kind in [lo, hi]; see the module docstring."""
    if kind == "uniform":
        return lo + (hi - lo) * torch.rand(n, generator=generator, dtype=torch.float64)
    if kind == "log":
        u = torch.rand(n, generator=generator, dtype=torch.float64)
        return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    if kind == "zero":   # no rotation: the neuron is a leaky integrator (control for "does rotation help memory?")
        return torch.zeros(n, dtype=torch.float64)
    k = torch.arange(1, n + 1, dtype=torch.float64)
    if kind == "golden":
        return lo + (hi - lo) * torch.remainder(k / PHI, 1.0)
    if kind == "independent":
        r = torch.tensor([math.sqrt(p) for p in primes(n)], dtype=torch.float64)
        return lo + (hi - lo) * torch.remainder(r, 1.0)
    if kind == "geometric":
        return hi * (lo / hi) ** ((k - 1) / max(1, n - 1))
    if kind == "phi_ladder":   # I23: brain-rhythm style bands, each rung PHI times the one below (lo, lo PHI, ...
        # <= hi), the n neurons shared out over the rungs in turn; any three neighbouring rungs add exactly:
        # w_{r+1} = w_r + w_{r-1}, because PHI^2 = PHI + 1
        rungs = int(math.floor(math.log(hi / lo) / math.log(PHI) + 1e-12)) + 1
        return lo * PHI ** torch.remainder(k - 1, rungs)
    if kind == "ladder2":      # I23b control: the same kind of ladder with ratio 2 (2 w_r = w_{r+1}), fewer rungs
        rungs = int(math.floor(math.log2(hi / lo) + 1e-12)) + 1
        return lo * 2.0 ** torch.remainder(k - 1, rungs)
    if kind == "phi_jitter":   # I23c: the phi ladder with each neuron nudged by up to +-5% (log) around its rung: bands
        # (clusters) without exactly shared frequencies
        rungs = int(math.floor(math.log(hi / lo) / math.log(PHI) + 1e-12)) + 1
        base = lo * PHI ** torch.remainder(k - 1, rungs)
        u = torch.rand(n, generator=generator, dtype=torch.float64) * 2 - 1
        return (base * torch.exp(0.05 * u)).clamp(lo, hi)
    if kind == "banded_strat":  # I23c: as many shared random bands as the phi ladder has rungs, one drawn log-uniformly
        # inside each of that many equal log-strata of [lo, hi] (even coverage, irregular spacing)
        rungs = int(math.floor(math.log(hi / lo) / math.log(PHI) + 1e-12)) + 1
        u = (torch.arange(rungs, dtype=torch.float64) + torch.rand(rungs, generator=generator, dtype=torch.float64)) / rungs
        bands = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        return bands[torch.remainder(k - 1, rungs).long()]
    if kind == "brain5":       # I28a: five bands at the classic brain-wave centres, delta 2, theta 6, alpha 10, beta 20,
        # gamma 40 Hz, read with 1 step = 1 ms (w = 2 pi f / 1000); the n neurons shared out over the bands in turn
        bands = torch.tensor([2.0, 6.0, 10.0, 20.0, 40.0], dtype=torch.float64) * 2 * math.pi / 1000
        return bands[torch.remainder(k - 1, 5).long()]
    if kind == "ladder5":      # I28a: five bands spaced evenly in log over the whole range [lo, hi]
        return lo * (hi / lo) ** (torch.remainder(k - 1, 5) / 4)
    if kind == "banded_log":   # I23b control: as many shared frequencies as the phi ladder has rungs, but drawn
        # log-uniformly at random (shared frequencies without the regular phi spacing)
        rungs = int(math.floor(math.log(hi / lo) / math.log(PHI) + 1e-12)) + 1
        bands = frequency_set("log", rungs, lo, hi, generator=generator)
        return bands[torch.remainder(k - 1, rungs).long()]
    if kind == "harmonic":
        return lo + (hi - lo) * (k - 1) / max(1, n - 1)
    if kind in ("quantized", "sidon"):
        if bits is None:
            raise ValueError(f"{kind} frequencies need bits")
        P = 2**bits
        m_lo, m_hi = grid_range(lo, hi, bits)
        if kind == "quantized":
            w = frequency_set("log", n, lo, hi, generator=generator)
            m = torch.round(w * P / (2 * math.pi)).clamp(m_lo, m_hi)
        else:
            m = torch.tensor(sidon_grid(n, m_lo, m_hi), dtype=torch.float64)
        return 2 * math.pi * m / P
    raise ValueError(f"unknown frequency set {kind!r}; expected one of {FREQ_SETS}")
