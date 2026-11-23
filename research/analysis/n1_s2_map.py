"""N1-S2: a spatial map for a moving agent: phasor fractional powers vs a real level code.

    cd research && .venv/bin/python -m analysis.n1_s2_map            # full (a few minutes)
    .venv/bin/python -m analysis.n1_s2_map --quick

M objects sit at fixed positions in [0, 1]^2. The agent starts at (0.5, 0.5) and takes T random steps (each
component U(-0.03, 0.03)). It does not know its steps exactly: its estimate of each step has Gaussian error
(std sigma). Questions about the map are egocentric (relative to the agent's TRUE position):
  (a) where is object k relative to me?   (RMSE over objects)
  (b) which object is at relative position q?   (inverse query at each object's true relative position; accuracy)
Codes at equal real numbers: N phasors (2N reals) vs 2N bipolar neurons.
  fpe 2D   : value(p) = exp(j theta . p), theta_n ~ U(-pi b, pi b)^2; value(p) value(q) = value(p + q)
  level 2D : value(x, y) = level_x(x) * level_y(y) (two independent piecewise level codes, elementwise product)
Ways to keep the map (record r = sum_k object_k (x) value(position_k)):
  1 fpe_ego   : egocentric map; after every step r <- r * value(-estimated step): one product, no decoding
  2 fpe_allo  : allocentric map + the agent's estimated position kept as two numbers; answer = decode - position
  3 level_allo : as 2 with the level code (the strong real-valued baseline)
  4 level_ego : egocentric map with the level code: every step decodes all M objects, shifts them, re-encodes
The path-integration error (the estimated position minus the true one) is the same for all four.
Prediction written before running: 1 = 2 = 3 in accuracy (limited by recall and the path error); 4 gets worse with T
(decoding errors compound) and costs M decodes per step. The phasor advantage would then be only: an egocentric map
kept up to date at a fixed cost, without compounding errors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "n1_s2_map"
STEP = 0.03


def grid2d(lo: float, hi: float, h: float) -> torch.Tensor:
    ax = torch.arange(lo, hi + 1e-9, h, dtype=torch.float64)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    return torch.stack([X.reshape(-1), Y.reshape(-1)], -1)              # (G, 2)


class FPE2:
    def __init__(self, n: int, b: float, g: torch.Generator):
        self.n = n
        self.theta = (torch.rand(n, 2, generator=g, dtype=torch.float64) * 2 - 1) * math.pi * b

    def value(self, p: torch.Tensor) -> torch.Tensor:                   # p (K, 2) -> (K, N)
        return torch.polar(torch.ones(p.shape[0], self.n, dtype=torch.float64), p @ self.theta.T)

    def symbols(self, k: int, g: torch.Generator) -> torch.Tensor:
        return torch.polar(torch.ones(k, self.n, dtype=torch.float64), torch.rand(k, self.n, generator=g, dtype=torch.float64) * 2 * math.pi)

    def unbind(self, r, a):
        return r * torch.conj(a)

    def sim(self, x, codes):
        return (x @ torch.conj(codes).T).real / self.n


class Level2:
    """2D code as the elementwise product of two independent piecewise bipolar level codes (see n1_s1b.Level)."""

    def __init__(self, n: int, pieces: int, g: torch.Generator, lo: float = -1.0, hi: float = 1.0):
        from analysis.n1_s1b import Level
        self.n, self.lo, self.hi = n, lo, hi
        self.lx, self.ly = Level(n, g, pieces), Level(n, g, pieces)

    def value(self, p: torch.Tensor) -> torch.Tensor:                   # map [lo, hi] to the code's [0, 1]
        u = (p - self.lo) / (self.hi - self.lo)
        return self.lx.value(u[:, 0]) * self.ly.value(u[:, 1])

    def symbols(self, k: int, g: torch.Generator) -> torch.Tensor:
        return torch.randint(0, 2, (k, self.n), generator=g).double() * 2 - 1

    def unbind(self, r, a):
        return r * a

    def sim(self, x, codes):
        return x @ codes.T / self.n


def decode(code, x: torch.Tensor, grid: torch.Tensor, grid_codes: torch.Tensor) -> torch.Tensor:
    return grid[code.sim(x, grid_codes).argmax(-1)]


def trial(method: str, n: int, m: int, T: int, sigma: float, res: float, seed: int, h: float) -> dict:
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(m, 2, generator=g, dtype=torch.float64)            # true object positions
    steps = (torch.rand(T, 2, generator=g, dtype=torch.float64) * 2 - 1) * STEP
    est = steps + sigma * torch.randn(T, 2, generator=g, dtype=torch.float64)  # the agent's step estimates
    agent_true = 0.5 + steps.sum(0)
    agent_est = 0.5 + est.sum(0)
    rel_true = pos - agent_true                                         # egocentric ground truth
    fpe = method.startswith("fpe")
    gc = torch.Generator().manual_seed(seed + 1)
    lo, hi = (0.0, 1.0) if method.endswith("allo") else (-1.0, 1.0)     # the level code spans the map's frame
    code = FPE2(n, res, gc) if fpe else Level2(2 * n, int(res), gc, lo, hi)
    objs = code.symbols(m, gc)
    ops = {"step": 0, "query": 0}
    if method.endswith("allo"):
        grid = grid2d(0.0, 1.0, h)
        gcodes = code.value(grid)
        record = (objs * code.value(pos)).sum(0, keepdim=True)
        p_hat = decode(code, code.unbind(record, objs), grid, gcodes)
        rel_hat = p_hat - agent_est
        ops["query"] = grid.shape[0] * code.n                           # one decode
        # inverse query at relative position q: look up allocentric q + estimated position
        q_codes = code.value(rel_true + agent_est)
        inv = code.sim(code.unbind(record, q_codes), objs).argmax(-1)
    else:
        grid = grid2d(-1.0, 1.0, h)
        gcodes = code.value(grid)
        rel0 = pos - 0.5                                                # agent starts at the centre
        record = (objs * code.value(rel0)).sum(0, keepdim=True)
        if method == "fpe_ego":
            for t in range(T):
                record = record * code.value(-est[t:t + 1])             # shift every object at once
            ops["step"] = code.n
        else:   # level_ego: decode, shift, re-encode every object at every step
            for t in range(T):
                cur = decode(code, code.unbind(record, objs), grid, gcodes) - est[t]
                record = (objs * code.value(cur.clamp(-1, 1))).sum(0, keepdim=True)
            ops["step"] = m * grid.shape[0] * code.n
        rel_hat = decode(code, code.unbind(record, objs), grid, gcodes)
        ops["query"] = grid.shape[0] * code.n
        inv = code.sim(code.unbind(record, code.value(rel_true)), objs).argmax(-1)
    err = ((rel_hat - rel_true) ** 2).sum(-1).mean().sqrt().item()
    path_err = ((agent_est - agent_true) ** 2).sum().sqrt().item()
    return {"rmse": err, "path_err": path_err, "inverse_acc": (inv == torch.arange(m)).double().mean().item(),
            "ops_per_step": ops["step"], "ops_per_query": ops["query"]}


def run(method: str, n: int, m: int, T: int, sigma: float, res: float, trials: int, h: float, seed0=0):
    rs = [trial(method, n, m, T, sigma, res, seed0 + 31 * t, h) for t in range(trials)]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 2 if a.quick else 8
    N = 256 if a.quick else 1024
    h = 0.04            # decoding grid step (the same for every method)
    Ms = [4, 16] if a.quick else [4, 16, 32]
    Ts = [0, 10, 40] if a.quick else [0, 10, 40, 100]
    sigma = 0.002
    methods = ["fpe_ego", "fpe_allo", "level_allo", "level_ego"]
    OUT.mkdir(parents=True, exist_ok=True)
    # resolution of each code chosen on separate trials (seeds 9000+) at T = 10, by egocentric RMSE
    res_grid = {"fpe": [5.0, 10.0, 20.0], "level": [5, 10, 20, 40]}
    chosen = {}
    for meth in methods:
        fam = "fpe" if meth.startswith("fpe") else "level"
        for m in Ms:
            sc = {r: run(meth, N, m, 10, sigma, r, max(2, trials // 2), h, seed0=9000)["rmse"] for r in res_grid[fam]}
            chosen[(meth, m)] = min(sc, key=sc.get)
            print("select", meth, m, {k: round(v, 4) for k, v in sc.items()}, "->", chosen[(meth, m)], flush=True)
    rows = []
    for m in Ms:
        for T in Ts:
            for meth in methods:
                r = run(meth, N, m, T, sigma, chosen[(meth, m)], trials, h)
                rows.append({"M": m, "T": T, "method": meth, "res": chosen[(meth, m)], **r})
                print(m, T, meth, {k: round(v, 4) for k, v in r.items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps({"N_complex": N, "sigma": sigma, "grid_step": h, "rows": rows}, indent=2))
    L = [f"# N1-S2: egocentric spatial map for a moving agent (N = {N} phasors / {2 * N} bipolar, step noise {sigma}, grid {h}, {trials} trials)", ""]
    for m in Ms:
        L += [f"## M = {m} objects", "", "| method | resolution | " + " | ".join(f"RMSE T={T}" for T in Ts) + " | "
              + " | ".join(f"inverse acc T={T}" for T in Ts) + " | ops/step | ops/query |", "|---" * (2 * len(Ts) + 4) + "|"]
        for meth in methods:
            rs = [r for r in rows if r["M"] == m and r["method"] == meth]
            L.append(f"| {meth} | {rs[0]['res']:g} | " + " | ".join(f"{r['rmse']:.4f}" for r in rs) + " | "
                     + " | ".join(f"{r['inverse_acc']:.3f}" for r in rs) + f" | {rs[-1]['ops_per_step']:,.0f} | {rs[-1]['ops_per_query']:,.0f} |")
        pe = [r for r in rows if r["M"] == m and r["method"] == "fpe_ego"]
        L += ["", "Path-integration error alone (same for all): " + ", ".join(f"T={r['T']}: {r['path_err']:.4f}" for r in pe), ""]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
