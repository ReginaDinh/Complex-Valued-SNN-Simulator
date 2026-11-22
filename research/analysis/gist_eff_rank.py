"""User's hypothesis (2026-11-13): CV-SNN forgets because it stores too much detail instead of the gist.
First check without new training: across the continual-learning cells of I26b (H6 factorial) and I26d (sparsity),
at the lr chosen on validation, rank-correlate the dimensionality of the first hidden layer's code after task 1
(participation ratio of spike counts, `eff_rank` in experiments.ecl) with forgetting.

    cd research && .venv/bin/python -m analysis.gist_eff_rank
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rank(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0] * len(x)
    for k, i in enumerate(order):
        r[i] = k
    return r


def spearman(a, b) -> float:
    ra, rb = _rank(a), _rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def main() -> None:
    load = lambda p: [json.loads(l) for l in (ROOT / p).read_text().splitlines()]
    cells = []
    sel, rows = json.loads((ROOT / "results/ei26b/full/selected.json").read_text()), load("results/ei26b/full/runs.jsonl")
    for k, v in sel.items():
        cells.append((f"I26b:{k}", [r for r in rows if r.get("variant") == k and r.get("lr") == v["lr"]]))
    sel, rows = json.loads((ROOT / "results/ei26d/full/selected.json").read_text()), load("results/ei26d/full/runs.jsonl")
    for k, v in sel.items():
        if k == "lif|none":   # the same runs as I26b:lif
            continue
        m, sp = k.split("|")
        cells.append((f"I26d:{k}", [r for r in rows if r["model_name"] == m and r["sparsity"] == sp and r["lr"] == v["lr"]]))
    pts = []
    for name, rs in cells:
        er, f = st.mean(r["history"][0]["eff_rank"] for r in rs), st.mean(r["forgetting"] for r in rs)
        pts.append((name, er, f))
        print(f"{name:34s} code dimensionality {er:5.2f}   forgetting {f:.3f}")
    print(f"n = {len(pts)} conditions, Spearman rho(dimensionality, forgetting) = {spearman([p[1] for p in pts], [p[2] for p in pts]):+.2f}")


if __name__ == "__main__":
    main()
