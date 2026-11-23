"""Numerical checks of mathematical claims (see docs/verification/claims_2026-11.md).

    cd research && .venv/bin/python -m analysis.verify_claims

N1  gradient through a window of k steps (cLIF and LIF, detached reset) = |beta|^k
N2  golden-angle phases: never repeat, most uniform gaps (three-gap), but non-zero crosstalk
N3  multi-oscillator position codes: crosstalk vs distance for RoPE bases, phi-geometric,
    golden-angle-spaced, random and rational-grid frequency sets
N4  "golden ratio is the worst-approximable irrational" (Hurwitz): min_q q * ||q alpha||
N5  identities: cos(pi/5) = phi/2, golden angle, continuous Binet formula
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

PHI = (1 + 5**0.5) / 2
OUT = Path(__file__).resolve().parents[1] / "results" / "verify_claims"


def n1_gradient_window() -> dict:
    from cvsnn.layers import CLIF, LIF

    res = {}
    for kind in ("clif", "lif"):
        for mag in (0.9, 0.99, 0.999):
            for T in (10, 100, 1000):
                if kind == "clif":
                    n = CLIF(1, threshold=1e9, leak_mag=mag, leak_phase=0.3, learn_leak=False).double()
                    x = torch.zeros(T, 1, 1, dtype=torch.complex128, requires_grad=True)
                else:
                    n = LIF(1, threshold=1e9, leak=mag, learn_leak=False).double()
                    x = torch.zeros(T, 1, 1, dtype=torch.float64, requires_grad=True)
                with torch.no_grad():  # exact |beta| (sigmoid of logit round trip in float64)
                    if kind == "clif":
                        n.leak_mag_raw.fill_(math.log(mag / (1 - mag)))
                    else:
                        n.leak_raw.fill_(math.log(mag / (1 - mag)))
                _, pot, _ = n(x)
                target = pot[-1, 0, 0]
                (target.real if kind == "clif" else target).backward()
                g0 = x.grad[0, 0, 0].abs().item()          # d Re(U_T) / d x_0 (magnitude)
                res[f"{kind}|beta={mag}|T={T}"] = {"grad_first_input": g0, "beta_pow_T_minus_1": mag ** (T - 1)}
    return res


def _gaps(theta: float, N: int) -> tuple[float, float]:
    pts = np.sort(np.mod(np.arange(N) * theta, 2 * np.pi))
    gaps = np.diff(np.concatenate([pts, [pts[0] + 2 * np.pi]]))
    return gaps.min() * N / (2 * np.pi), gaps.max() * N / (2 * np.pi)  # 1.0 = perfectly even


def n2_golden_phases() -> dict:
    thetas = {
        "golden 2pi/phi^2": 2 * np.pi / PHI**2,
        "sqrt2 2pi*frac(sqrt2)": 2 * np.pi * (2**0.5 % 1),
        "pi-based 2pi*frac(pi)": 2 * np.pi * (np.pi % 1),
        "rational 2pi*3/8": 2 * np.pi * 3 / 8,
        "rational 2pi*1/100": 2 * np.pi / 100,
    }
    out = {}
    for name, th in thetas.items():
        rows = {}
        for N in (10, 100, 1000, 10000):
            mn, mx = _gaps(th, N)
            rows[N] = {"min_gap_x_N": round(mn, 4), "max_gap_x_N": round(mx, 4)}
        worst_min = min(_gaps(th, N)[0] for N in range(2, 2001))
        # repeats: smallest n>0 with n*theta = 0 mod 2pi (within 1e-9), up to 1e6
        n = np.arange(1, 1_000_001)
        rep = np.where(np.abs(np.mod(n * th + 1e-12, 2 * np.pi)) < 1e-9)[0]
        out[name] = {"gaps": rows, "worst_min_gap_x_N_for_N_2_2000": round(worst_min, 4),
                     "first_exact_repeat_n": int(n[rep[0]]) if len(rep) else None}
    return out


def _crosstalk(omegas: np.ndarray, max_d: int) -> np.ndarray:
    """s(D) = |mean_k exp(i D w_k)| for D = 1..max_d (similarity of codes D positions apart)."""
    s = np.empty(max_d)
    chunk = 200_000
    for a in range(0, max_d, chunk):
        d = np.arange(a + 1, min(max_d, a + chunk) + 1)[:, None]
        s[a:a + d.shape[0]] = np.abs(np.exp(1j * d * omegas[None, :]).mean(1))
    return s


def n3_position_codes(K: int = 32, max_d: int = 1_000_000) -> dict:
    d_model = 2 * K
    k = np.arange(K)
    rng = np.random.default_rng(0)
    sets = {
        "RoPE base 1e4": 10000.0 ** (-2 * k / d_model),
        "RoPE base 1e6": 1e6 ** (-2 * k / d_model),
        "phi-geometric (ratio phi = base phi^(d/2))": PHI ** (-k.astype(float)),
        "golden-angle spaced w_k = pi*frac(k/phi) (k>=1)": np.pi * np.mod((k + 1) / PHI, 1.0),
        "random uniform (0, pi)": rng.uniform(0, np.pi, K),
        "rational grid w_k = pi*k/K": np.pi * (k + 1) / K,
    }
    out = {}
    for name, w in sets.items():
        s = _crosstalk(w, max_d)
        bands = {}
        for lo, hi in ((1, 100), (100, 10_000), (10_000, 1_000_000)):
            seg = s[lo - 1:hi]
            bands[f"D in [{lo},{hi}]"] = {"max": round(float(seg.max()), 4), "mean": round(float(seg.mean()), 4)}
        out[name] = {"lowest_freq": float(w.min()), "highest_freq": float(w.max()),
                     "longest_period_steps": float(2 * np.pi / w.min()), "crosstalk": bands}
    out["_reference"] = {"random_phase_expected_mean |mean of K unit phasors|": round(math.sqrt(math.pi / (4 * K)), 4), "K": K}
    return out


def n4_hurwitz(qmax: int = 100_000) -> dict:
    alphas = {"1/phi (golden)": 1 / PHI, "sqrt2-1": 2**0.5 - 1, "pi-3": math.pi - 3, "e-2": math.e - 2, "sqrt3-1": 3**0.5 - 1}
    q = np.arange(1, qmax + 1, dtype=np.float64)
    out = {}
    for name, a in alphas.items():
        dist = np.abs(q * a - np.round(q * a))
        v = q * dist      # Hurwitz: liminf q*||q a|| <= 1/sqrt5 for all irrationals, equality only for phi-equivalents
        tail = v[1000:]
        out[name] = {"min q*||q a|| for q in [1000,1e5]": round(float(tail.min()), 5)}
    out["1/sqrt(5)"] = round(1 / 5**0.5, 5)
    return out


def n5_identities() -> dict:
    t = np.array([1, 2, 3, 4, 5, 10])
    binet = (PHI**t - (PHI ** (-t.astype(float))) * np.exp(1j * np.pi * t)) / 5**0.5
    half = 2.5
    return {
        "cos(pi/5)": math.cos(math.pi / 5), "phi/2": PHI / 2,
        "2cos(pi/5) == phi": abs(2 * math.cos(math.pi / 5) - PHI) < 1e-12,
        "golden angle deg (360/phi^2)": 360 / PHI**2,
        "Binet at integers (real part)": [round(float(z.real), 9) for z in binet],
        "Binet imag at integers": [round(float(z.imag), 12) for z in binet],
        "Binet at t=2.5 (complex)": [float(((PHI**half - PHI**-half * np.exp(1j * np.pi * half)) / 5**0.5).real),
                                     float(((PHI**half - PHI**-half * np.exp(1j * np.pi * half)) / 5**0.5).imag)],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"N1_gradient_window": n1_gradient_window(), "N2_golden_phases": n2_golden_phases(),
           "N3_position_codes": n3_position_codes(), "N4_hurwitz": n4_hurwitz(), "N5_identities": n5_identities()}
    (OUT / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
