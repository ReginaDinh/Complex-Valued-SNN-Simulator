"""N1-S1b: continuous values in a knowledge memory: phasor fractional powers vs a real level code.

    cd research && .venv/bin/python -m analysis.n1_s1b          # full (about 1-2 min)
    .venv/bin/python -m analysis.n1_s1b --quick

Facts with a continuous value, e.g. "temperature of A = 0.37", are stored as a superposition
r = sum_i role_i (x) value(v_i) of M facts, v_i ~ U(0, 1). Value codes, at equal real numbers:
  fpe   : fractional power encoding with N phasor neurons (2N reals): value(v) = exp(j theta v), theta_n ~ U(-pi b, pi b),
          so value(v) * value(w) = value(v + w) and value(v) * conj(value(w)) = value(v - w): arithmetic on the code itself.
          Similarity Re<value(v), value(w)> / N = sinc(b (v - w)); the bandwidth b is chosen on separate trials.
          Roles are random phasors; bind = elementwise product, unbind = product with the conjugate.
  level : bipolar level code with 2N neurons (2N reals), in pieces (independent random end points, linear moves
          between them; the number of pieces sets the resolution, chosen on separate trials like the fpe bandwidth).
          Roles are random +-1; bind = unbind = elementwise product.
Tasks, all decoded by the best match over a grid of candidate values:
  1. recall     : the value of a stored fact (RMSE)
  2. difference : v_A - v_B for two stored facts. fpe: in the code, value(v_A) * conj(value(v_B)) decoded once
                  ("algebraic"), or decode both and subtract; level: decode both and subtract (no in-code operation)
  3. shift all  : add delta to every stored value at once. fpe: one product r * value(delta); level: no operation
                  on the record exists (each fact would have to be decoded, shifted and re-encoded)
Prediction written before running: recall about equal; difference about equal or better for fpe; only fpe can shift
all values at once, and with no added error (a product with a unit phasor is exact).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "n1_s1b"
GRID = torch.linspace(-1.0, 2.0, 3001, dtype=torch.float64)   # candidate values (covers differences and shifts)


class FPE:
    def __init__(self, n: int, b: float, g: torch.Generator):
        self.n = n
        self.theta = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * math.pi * b

    def value(self, v: torch.Tensor) -> torch.Tensor:
        return torch.polar(torch.ones(len(v), self.n, dtype=torch.float64), v.view(-1, 1) * self.theta.view(1, -1))

    def roles(self, m: int, g: torch.Generator) -> torch.Tensor:
        return torch.polar(torch.ones(m, self.n, dtype=torch.float64), torch.rand(m, self.n, generator=g, dtype=torch.float64) * 2 * math.pi)

    @staticmethod
    def bind(a, b):
        return a * b

    @staticmethod
    def unbind(r, a):
        return r * torch.conj(a)

    def decode(self, x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        grid = GRID[(GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)]
        sim = (x @ torch.conj(self.value(grid)).T).real
        return grid[sim.argmax(-1)]


class Level:
    """Bipolar level code with `segments` pieces (the real-valued counterpart of the fpe bandwidth): independent
    random end points E_0 .. E_s; inside piece k a value moves linearly from E_k to E_{k+1} by copying a growing
    prefix (fixed random order) of E_{k+1}. segments = 1 is the standard level code. Values outside [0, 1] saturate."""

    def __init__(self, n: int, g: torch.Generator, segments: int = 1):
        self.n, self.segments = n, segments
        self.ends = torch.randint(0, 2, (segments + 1, n), generator=g).double() * 2 - 1
        order = torch.randperm(n, generator=g)
        self.rank = torch.empty(n, dtype=torch.long)
        self.rank[order] = torch.arange(n)

    def value(self, v: torch.Tensor) -> torch.Tensor:
        x = v.clamp(0, 1) * self.segments
        k = torch.clamp(torch.floor(x).long(), max=self.segments - 1)
        t = x - k
        copy = (self.rank.view(1, -1) < torch.round(t * self.n).long().view(-1, 1))
        return torch.where(copy, self.ends[k + 1], self.ends[k])

    def roles(self, m: int, g: torch.Generator) -> torch.Tensor:
        return torch.randint(0, 2, (m, self.n), generator=g).double() * 2 - 1

    @staticmethod
    def bind(a, b):
        return a * b

    @staticmethod
    def unbind(r, a):
        return r * a

    def decode(self, x: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        grid = GRID[(GRID >= max(lo, 0.0) - 1e-9) & (GRID <= min(hi, 1.0) + 1e-9)]
        sim = x @ self.value(grid).T
        return grid[sim.argmax(-1)]


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().sqrt().item()


def trial(code_name: str, n: int, m: int, b: float, seed: int, pairs: int = 20, delta: float = 0.3) -> dict:
    g = torch.Generator().manual_seed(seed)
    code = FPE(n, b, g) if code_name == "fpe" else Level(2 * n, g, int(b))
    v = torch.rand(m, generator=g, dtype=torch.float64)
    roles = code.roles(m, g)
    record = code.bind(roles, code.value(v)).sum(0, keepdim=True)
    unb = code.unbind(record, roles)
    v_hat = code.decode(unb)
    res = {"recall_rmse": rmse(v_hat, v)}
    if m >= 2:
        ia = torch.randint(0, m, (pairs,), generator=g)
        ib = (ia + torch.randint(1, m, (pairs,), generator=g)) % m
        true_d = v[ia] - v[ib]
        res["diff_decode_rmse"] = rmse(v_hat[ia] - v_hat[ib], true_d)
        if code_name == "fpe":
            prod = unb[ia] * torch.conj(unb[ib])               # value(v_A - v_B) plus cross-talk, in the code
            res["diff_algebraic_rmse"] = rmse(code.decode(prod, -1.0, 1.0), true_d)
    if code_name == "fpe":
        shifted = record * code.value(torch.tensor([delta], dtype=torch.float64))   # every stored value + delta
        # decoded over the shifted range [delta, 1 + delta] (the shift is known), like the recall over [0, 1]
        res["shift_all_rmse"] = rmse(code.decode(code.unbind(shifted, roles), delta, 1.0 + delta), v + delta)
    return res


def run(code_name: str, n: int, m: int, b: float, trials: int, seed0: int) -> dict:
    rs = [trial(code_name, n, m, b, seed0 + 101 * t) for t in range(trials)]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 2 if a.quick else 10
    N = 256 if a.quick else 1024
    Ms = [1, 4, 16] if a.quick else [1, 4, 16, 64, 128]
    bands = [5.0, 10.0, 20.0, 40.0, 80.0]
    OUT.mkdir(parents=True, exist_ok=True)
    # resolution chosen on separate trials (seeds 5000+), per M, by recall RMSE: fpe bandwidth, level-code pieces
    chosen = {"fpe": {}, "level": {}}
    grids = {"fpe": bands, "level": [1, 2, 5, 10, 20, 40, 80]}
    for code_name in ("fpe", "level"):
        for m in Ms:
            scores = {b: run(code_name, N, m, b, trials, 5000 + m)["recall_rmse"] for b in grids[code_name]}
            chosen[code_name][m] = min(scores, key=scores.get)
            print("selection", code_name, "M", m, {b: round(x, 4) for b, x in scores.items()}, "->", chosen[code_name][m], flush=True)
    rows = []
    for m in Ms:
        f = run("fpe", N, m, chosen["fpe"][m], trials, 100 + m)
        lv = run("level", N, m, chosen["level"][m], trials, 100 + m)
        rows.append({"M": m, "fpe_band": chosen["fpe"][m], "level_pieces": chosen["level"][m], "fpe": f, "level": lv})
        print("M", m, "fpe", {k: round(x, 4) for k, x in f.items()}, "| level", {k: round(x, 4) for k, x in lv.items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps({"N_complex": N, "reals": 2 * N, "trials": trials, "rows": rows}, indent=2))
    L = [f"# N1-S1b: continuous values, phasor fractional powers (N = {N} phasors) vs bipolar level code ({2 * N} neurons), {trials} trials", "",
         "Values in [0, 1]. RMSE (lower is better). fpe bandwidth and level-code pieces chosen on separate trials.", "",
         "| M facts | fpe band / level pieces | recall fpe | recall level | difference fpe (in code) | difference fpe (decode, subtract) | difference level (decode, subtract) | shift all, fpe | shift all, level |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, lv = r["fpe"], r["level"]
        def g(d, k):
            return f"{d[k]:.4f}" if k in d else "–"
        L.append(f"| {r['M']} | {r['fpe_band']:g} / {r['level_pieces']} | {g(f, 'recall_rmse')} | {g(lv, 'recall_rmse')} | {g(f, 'diff_algebraic_rmse')} | "
                 f"{g(f, 'diff_decode_rmse')} | {g(lv, 'diff_decode_rmse')} | {g(f, 'shift_all_rmse')} | not possible in the code |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
