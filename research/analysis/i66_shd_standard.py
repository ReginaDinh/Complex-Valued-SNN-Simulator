"""I66 (standard 20-class SHD): per model, the lr with the best mean final validation accuracy; test accuracy at that lr.

    cd research && .venv/bin/python -m analysis.i66_shd_standard results/ei66/run1 [--out <report dir>]

Prints, per model: parameters, the selected lr (edge flagged), test and validation accuracy (mean +- 95% CI over seeds),
the learning curve (test accuracy at each recorded epoch, mean over seeds) and paired per-seed test comparisons of the
first model with the others (two-sided sign test, ties dropped).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path

from cvsnn.utils import mean_ci95

FIRST = "cand_long_fw003"
NAMES = {"cand_long_fw003": "rotating, long memory", "cand_longnorot_fw001": "no rotation, long memory",
         "lif_fw2_d11_fw001": "LIF, long memory", "lif_short_fw003": "LIF, short memory",
         "se_adlif_fw001": "SE-adLIF", "brf_fw003": "BRF (bug, invalid)", "brfv2_fw003": "BRF (fixed)", "cand_longnorm_slowfw": "rotating + LRU norm"}


def sign_p(w: int, l: int) -> float:
    n = w + l
    return min(1.0, 2 * sum(comb(n, i) for i in range(max(w, l), n + 1)) / 2 ** n) if n else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.run) / "runs.jsonl").read_text().splitlines()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["variant"], r["lr"])].append(r)
    models = sorted({v for v, _ in cells}, key=lambda m: (m != FIRST, m))
    ci = lambda xs: "{:.3f} ± {:.3f}".format(mean_ci95(xs)[0], mean_ci95(xs)[2])
    lines = [f"# Standard SHD (20 classes) summary: `{a.run}`", "",
             "lr chosen per model on the mean final validation accuracy (validation = fixed 10% of the training set); test = "
             "official test set. Chance 0.05.", "", "## Per lr (mean over seeds)", "", "| model | lr | seeds | validation | test |", "|---|---|---|---|---|"]
    sel = {}
    for m in models:
        lrs = sorted(lr for v, lr in cells if v == m)
        best = max(lrs, key=lambda lr: sum(r["val"] for r in cells[(m, lr)]) / len(cells[(m, lr)]))
        sel[m] = best
        for lr in lrs:
            rs = cells[(m, lr)]
            lines.append(f"| {NAMES.get(m, m)} | {lr:g} | {len(rs)} | {sum(r['val'] for r in rs) / len(rs):.3f} | {sum(r['test'] for r in rs) / len(rs):.3f} |")
    epochs = [c["epoch"] for c in cells[(models[0], sel[models[0]])][0]["curve"]]
    lines += ["", "## Selected lr", "", "| model | params | lr | validation | **test** | test by epoch " + " / ".join(map(str, epochs)) + " |",
              "|---|---|---|---|---|---|"]
    for m in models:
        rs = sorted(cells[(m, sel[m])], key=lambda r: r["seed"])
        lrs = sorted(lr for v, lr in cells if v == m)
        edge = " ⚠ edge" if sel[m] in (lrs[0], lrs[-1]) else ""
        curve = " / ".join(f"{sum(r['curve'][i]['test'] for r in rs) / len(rs):.3f}" for i in range(len(epochs)))
        lines.append(f"| {NAMES.get(m, m)} | {rs[0]['params']} | {sel[m]:g}{edge} | {ci([r['val'] for r in rs])} | {ci([r['test'] for r in rs])} | {curve} |")
    lines += ["", f"## Paired per seed: {NAMES[FIRST]} vs each model (test accuracy)", "", "| vs | higher | mean difference (95% CI) |", "|---|---|---|"]
    A = {r["seed"]: r for r in cells[(FIRST, sel[FIRST])]}
    for m in models[1:]:
        B = {r["seed"]: r for r in cells[(m, sel[m])]}
        seeds = sorted(set(A) & set(B))
        w = sum(A[s]["test"] > B[s]["test"] for s in seeds)
        lo = sum(A[s]["test"] < B[s]["test"] for s in seeds)
        lines.append(f"| {NAMES.get(m, m)} | {w}/{len(seeds)} (lost {lo}), p = {sign_p(w, lo):.3f} | {ci([A[s]['test'] - B[s]['test'] for s in seeds])} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.md").write_text(text)
        (o / "selected.json").write_text(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
