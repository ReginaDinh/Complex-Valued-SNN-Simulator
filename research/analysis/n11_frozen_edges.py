"""N11: the I61c causal 2 x 2 (frozen / learned neuron parameters x polar cLIF / free adLIF) after the grid-edge extension
of the two frozen cells.

    cd research && .venv/bin/python -m analysis.n11_frozen_edges [out_dir]

Selection on seeds 0-4 (validation score = mean of recall and E-CL average) over every setting tried for each frozen cell,
in results/ei61c/run1 (the original I61c grid plus the N11 extensions); the learned cells keep their I54i / I61 selection.
Reports 10-seed means with 95% CI and per-seed comparisons (sign test, two-sided, ties dropped).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results/ei61c/run1/runs.jsonl"
LEARNED = {"cLIF learned": ("cand_long_fw003", "0.01/0.0003"), "adLIF learned": ("adlif_long_slowfw", "0.01/0.0003")}
FROZEN = {"cLIF frozen": "cand_long_frz", "adLIF frozen": "adlif_long_frz"}
METRICS = ("val_score", "recall_final", "ecl_final_avg", "ecl_forgetting")


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main(out: Path) -> None:
    from cvsnn.utils import mean_ci95
    rows = [json.loads(l) for l in RUNS.read_text().splitlines()]
    two = [r for r in rows if "|2heads|phaselr" in r["variant"] and r["variant"].count("|") == 2]
    cells, sel = {}, {}
    for name, (v, lr) in LEARNED.items():
        cells[name] = {r["seed"]: r for r in two if r["variant"].split("|")[0] == v and r["lr"] == lr}
        sel[name] = f"{v} {lr} (I54i / I61)"
    for name, fam in FROZEN.items():
        g = defaultdict(list)
        for r in two:
            v = r["variant"].split("|")[0]
            if v.startswith(fam + "_") and r["seed"] < 5:
                g[(v, r["lr"])].append(r["val_score"])
        k = max((k for k in g if len(g[k]) == 5), key=lambda k: sum(g[k]) / 5)
        cells[name] = {r["seed"]: r for r in two if r["variant"].split("|")[0] == k[0] and r["lr"] == k[1]}
        sel[name] = f"{k[0]} {k[1]} (seed 0-4 val {sum(g[k]) / 5:.3f}; {len(g)} settings tried)"
    L = ["# N11: I61c 2 x 2 after the grid-edge extension of the frozen cells", "", "Selected settings:", ""]
    L += [f"- {n}: {s}" for n, s in sel.items()]
    L += ["", "| cell | seeds | validation score | recall at end | recall fails (< 0.8) | E-CL final | E-CL forgetting |", "|---|---|---|---|---|---|---|"]
    res = {}
    for name, c in cells.items():
        rs = [c[s] for s in sorted(c)]
        st = {m: mean_ci95([r[m] for r in rs]) for m in METRICS}
        res[name] = {"seeds": len(rs), **{m: st[m] for m in METRICS}, "selected": sel[name]}
        L.append(f"| {name} | {len(rs)} | {st['val_score'][0]:.3f} ± {st['val_score'][2]:.3f} | {st['recall_final'][0]:.3f} ± {st['recall_final'][2]:.3f} | "
                 f"{sum(r['recall_final'] < 0.8 for r in rs)}/{len(rs)} | {st['ecl_final_avg'][0]:.3f} ± {st['ecl_final_avg'][2]:.3f} | "
                 f"{st['ecl_forgetting'][0]:.3f} ± {st['ecl_forgetting'][2]:.3f} |")
    L += ["", "Per-seed comparisons (A higher than B; sign test, ties dropped):", "",
          "| A vs B | validation score | recall at end | paired diff in validation score (mean ± CI) |", "|---|---|---|---|"]
    for a, b in (("cLIF learned", "cLIF frozen"), ("cLIF learned", "adLIF frozen"), ("cLIF frozen", "adLIF frozen"), ("adLIF frozen", "adLIF learned")):
        seeds = sorted(set(cells[a]) & set(cells[b]))
        out_ = []
        for m in ("val_score", "recall_final"):
            w = sum(cells[a][s][m] > cells[b][s][m] for s in seeds)
            l_ = sum(cells[a][s][m] < cells[b][s][m] for s in seeds)
            out_.append(f"{w}/{len(seeds)} (p = {sign_p(w, l_):.3f})")
        d = mean_ci95([cells[a][s]["val_score"] - cells[b][s]["val_score"] for s in seeds])
        L.append(f"| {a} vs {b} | {out_[0]} | {out_[1]} | {d[0]:+.3f} ± {d[2]:.3f} |")
        res[f"{a} vs {b}"] = {"val": out_[0], "recall": out_[1], "diff": d}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/n11")
