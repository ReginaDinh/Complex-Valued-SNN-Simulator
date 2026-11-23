"""N3_n1b: continuous values in a knowledge memory: phase fractional powers vs a real level code.

    cd research && .venv/bin/python -m analysis.n3_n1b          # full (about 1-2 min)
    cd research && .venv/bin/python -m analysis.n3_n1b --quick

Facts with a continuous value, e.g. "temperature of A = 0.37", are stored as a superposition
    S = sum_i code(k_i) * value(v_i)   (with * = elementwise product)
in D = N = 32 real (or N = 16 complex = 32 real) numbers:
  fpe   : fractional power encoding with N phasor neurons (2N reals): value(v) = exp(i theta v), theta_n ~ U(-pi b, pi b),
          so value(v1) * value(v2) = value(v1 + v2) and value(v1) * conj(value(v2)) = value(v1 - v2): arithmetic on the code itself.
          One neuron with theta_n = 0; value(0) = 1; value(1) = e^{i theta}; the bandwidth b is chosen on separate trials.
          Keys are random phases; bind = elementwise product, unbind = product with the conjugate.
  level : bipolar level code with 2N neurons (2N reals), in pieces (independent random end points, linear moves
          between them): the number of pieces sets the resolution, chosen on separate trials (like the fpe bandwidth).
          Keys are random +-1; bind = unbind = elementwise product.
Tasks, all decoded by the best match over a grid of candidate values:
  1. recall     : the value of a stored fact (RMSE)
  2. difference : v_A - v_B for two stored facts. Fpe: in the code, value(v_A) * conj(value(v_B)) decoded once
                  ("algebraic"), or decode both and subtract; level: decode both and subtract (no in-code operation)
  3. shift all  : add delta to every stored value at once. Fpe: one product S * value(delta); level: no operation
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

OUT = Path(__file__).resolve().parents[1] / "results" / "n3_n1b"
GRID = torch.linspace(-1.0, 2.0, 3001, dtype=torch.float64)   # candidate values (covers differences and shifts)


class FPE:
    def __init__(self, n: int, b: float, g: torch.Generator):
        self.n = n
        self.theta = (torch.rand(n, generator=g, dtype=torch.float64) * 2 - 1) * math.pi * b

    def value(self, v: torch.Tensor) -> torch.Tensor:
        return torch.polar(torch.ones_like(v, dtype=torch.float64), v.view(-1, 1) * self.theta.view(1, -1))

    def roles(self, m: int, g: torch.Generator) -> torch.Tensor:
        return torch.polar(torch.ones(m, self.n, dtype=torch.float64), torch.rand(m, self.n, generator=g, dtype=torch.float64) * 2 * math.pi)

    @staticmethod
    def bind(a, b):
        return a * b

    @staticmethod
    def unbind(s, a):
        return s * torch.conj(a)

    def decode(self, z: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        grid = GRID[(GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)]
        sim = (z @ self.value(grid).T).real
        return grid[sim.argmax(-1)]


class Level:
    """Bipolar level code with `segments` pieces (the real-valued counterpart of the fpe bandwidth): independent
    random end points E_0..E_K; inside piece k value v moves linearly from E_k to E_{k+1} by copying a growing
    prefix (fixed random order) of E_{k+1}. segments = 1 is the standard level code. Values outside [0, 1] saturate."""

    def __init__(self, n: int, g: torch.Generator, segments: int = 1):
        self.n, self.segments = n, segments
        self.ends = torch.randint(0, 2, (segments + 1, n), generator=g).double() * 2 - 1
        order = torch.rand(segments, n, generator=g).argsort(-1)
        self.rank = torch.empty_like(order)
        self.rank.scatter_(1, order, torch.arange(n).expand_as(order))

    def value(self, v: torch.Tensor) -> torch.Tensor:
        v = v.clamp(0, 1) * self.segments
        k = v.long().clamp(max=self.segments - 1)
        t = v - k
        copy = (self.rank[k] < torch.round(t * self.n).long().view(-1, 1))
        return torch.where(copy, self.ends[k + 1], self.ends[k])

    def roles(self, m: int, g: torch.Generator) -> torch.Tensor:
        return torch.randint(0, 2, (m, self.n), generator=g).double() * 2 - 1

    @staticmethod
    def bind(a, b):
        return a * b

    @staticmethod
    def unbind(s, a):
        return s * a

    def decode(self, z: torch.Tensor, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
        grid = GRID[(GRID >= lo - 1e-9) & (GRID <= hi + 1e-9)]
        sim = z @ self.value(grid).T
        return grid[sim.argmax(-1)]


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().sqrt().item()


def trial(code_name: str, n: int, int_p: int | float, g: torch.Generator, pairs: int = 10, delta: float = 0.2) -> dict:
    code = FPE(n, int_p, g) if code_name == "fpe" else Level(n, g, int(int_p))
    v = torch.rand(pairs, generator=g, dtype=torch.float64)
    K = code.roles(pairs, g)
    record = code.bind(K, code.value(v)).sum(0, keepdim=True)
    v_hat = code.decode(code.unbind(record, K))
    res = {"recall_rmse": rmse(v_hat, v)}
    # 2. difference: v_A - v_B
    idx = torch.randint(0, pairs, (2,), generator=g)
    a, b = int(idx[0]), int(idx[1])
    res["diff_decode_rmse"] = rmse(v_hat[a:a + 1] - v_hat[b:b + 1], v[a:a + 1] - v[b:b + 1])
    if code_name == "fpe":
        prod = code.unbind(record, K[a:a + 1]) * torch.conj(code.unbind(record, K[b:b + 1]))   # value(v_A - v_B) plus cross-talk, in the code
        res["diff_algebraic_rmse"] = rmse(code.decode(prod, -1.0, 1.0), v[a:a + 1] - v[b:b + 1])
    # 3. shift all stored values by delta in one operation
    shifted = record * code.value(torch.tensor([delta], dtype=torch.float64))   # every stored value + delta
    in_range = (v + delta) <= 1.0                                               # measured over the unshifted range (the shift is known), like the recall over [0, 1]
    res["shift_all_rmse"] = rmse(code.decode(code.unbind(shifted, K[in_range]), 0.0, 1.0), v[in_range] + delta)
    return res


def run(code_name: str, n: int, int_p: int | float, trials: int, seed: int) -> dict:
    rs = [trial(code_name, n, int_p, torch.Generator().manual_seed(seed + k)) for k in range(trials)]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 2 if a.quick else 10
    trials_val = 2 if a.quick else 30
    Ns = [16, 32] if a.quick else [4, 8, 16, 32, 64, 128]
    bands = [0.5, 1.0, 2.0, 4.0, 10.0, 20.0, 40.0]
    OUT.mkdir(parents=True, exist_ok=True)
    # resolution chosen on separate trials (seeds 9000+), per N, by recall RMSE: fpe bandwidth, level-code pieces
    chosen = {"fpe": {}, "level": {}}
    grids = {"fpe": bands, "level": [1, 2, 5, 10, 20, 40, 80]}
    for code_name in ("fpe", "level"):
        for n in Ns:
            nn = n if code_name == "fpe" else 2 * n
            scores = {b: run(code_name, nn, b, trials_val, 9000 + n)["recall_rmse"] for b in grids[code_name]}
            chosen[code_name][n] = min(scores, key=scores.get)
            print(f"selection {code_name} N={n}: " + ", ".join(f"{b}:{v:.3f}" for b, v in scores.items()) + f" -> {chosen[code_name][n]}", flush=True)
    rows = []
    for n in Ns:
        f = run("fpe", n, chosen["fpe"][n], trials, 100 + n)
        lv = run("level", 2 * n, chosen["level"][n], trials, 100 + n)
        rows.append({"N": n, "fpe_band": chosen["fpe"][n], "level_pieces": chosen["level"][n], "fpe": f, "level": lv})
        print(f"N={n:3d}  fpe(b={chosen['fpe'][n]}): recall={f['recall_rmse']:.3f} diff_alg={f['diff_algebraic_rmse']:.3f} shift={f['shift_all_rmse']:.3f} | "
              f"level(seg={chosen['level'][n]}): recall={lv['recall_rmse']:.3f} diff={lv['diff_decode_rmse']:.3f}", flush=True)
    L = ["# N3_n1b: continuous values, phase fractional powers (2N reals) vs bipolar level code (2N reals)", "",
         f"10 stored facts, random keys, decoded by nearest candidate (step 0.001, {trials} independent trials).",
         "N facts; fpe band / level pieces chosen on separate trials.", "",
         "| N facts | fpe band / level pieces | recall fpe | recall level | difference fpe (in code) | difference fpe (decode, subtract) | difference level (decode, subtract) | shift all, fpe | shift all, level |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, lv = r["fpe"], r["level"]
        if_4 = f" (2N = {2 * r['N']})" if r["N"] == 16 else ""
        L.append(f"| {r['N']}{if_4} | {r['fpe_band']} / {r['level_pieces']} | {f['recall_rmse']:.3f} | {lv['recall_rmse']:.3f} | "
                 f"{f['diff_algebraic_rmse']:.3f} | {f['diff_decode_rmse']:.3f} | {lv['diff_decode_rmse']:.3f} | {f['shift_all_rmse']:.3f} | not possible in the code |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    (OUT / "rows.json").write_text(json.dumps({"chosen": chosen, "rows": rows}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
