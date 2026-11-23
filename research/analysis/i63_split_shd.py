"""I63 / I63b (Split-SHD) and other ei54 task-sequence runs with tagged variants: select per model on seeds 0-4, report 10 seeds.

    cd research && .venv/bin/python -m analysis.i63_split_shd results/ei63/run1 [--out <report dir>] [--task MNIST]

Per model (the variant name before the first "|"): the setting (lr pair, and SI strength c if present) with the best mean
validation score on seeds 0-4; then all seeds at that setting (mean +- 95% CI), the held-out seeds 5-9 alone, per-task
accuracy right after each task and at the end, and paired per-seed comparisons of the first model with every other one
(two-sided sign test, ties dropped). A choice at the edge of its grid is flagged.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from math import comb
from pathlib import Path

from cvsnn.utils import mean_ci95

FIRST = "cand_long_fw003"  # the rotating long-memory network, compared with every other model
NAMES = {"cand_long_fw003": "rotating, long memory", "cand_longnorot_fw001": "no rotation, long memory",
         "lif_fw2_d11_fw001": "LIF, long memory", "lif_short_fw003": "LIF, short memory",
         "se_adlif_fw001": "SE-adLIF", "brf_fw003": "BRF (bug, invalid)", "brfv2_fw003": "BRF (fixed)", "cand_longnorm_slowfw": "rotating + LRU norm"}


def sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def setting(r: dict) -> tuple:
    m = re.search(r"\|si=([0-9.e+-]+)", r["variant"])
    return (r["lr"], float(m.group(1)) if m else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default="SHD", help="name of the later tasks in the tables (e.g. MNIST for the I59b runs)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in (Path(a.run) / "runs.jsonl").read_text().splitlines()]
    cells = defaultdict(list)
    for r in rows:
        cells[(r["variant"].split("|")[0], setting(r))].append(r)
    models = sorted({m for m, _ in cells}, key=lambda m: (m != FIRST, m))
    grid = {m: sorted({s for mm, s in cells if mm == m}, key=lambda s: (float(s[0].split("/")[1]), s[1] or 0)) for m in models}
    lines = [f"# Split-SHD summary: `{a.run}`", "",
             "Setting chosen per model on the mean validation score (recall + task average) of seeds 0-4; all numbers below "
             "at that setting. Recall chance 0.125, SHD chance 0.25 (four words per task). Forgetting averaged over tasks 0-3.", "",
             "## Validation score per setting (seeds 0-4)", "", "| model | setting | val score | SHD final (all seeds) |", "|---|---|---|---|"]
    sel = {}
    for m in models:
        best = None
        for s in grid[m]:
            rs = cells[(m, s)]
            v = [r["val_score"] for r in rs if r["seed"] < 5]
            if len(v) < 5:
                continue
            sc = sum(v) / len(v)
            lines.append(f"| {NAMES.get(m, m)} | lr {s[0]}" + (f", c {s[1]:g}" if s[1] is not None else "") +
                         f" | {sc:.3f} | {sum(r['ecl_final_avg'] for r in rs) / len(rs):.3f} |")
            if best is None or sc > best[0]:
                best = (sc, s)
        sel[m] = best[1]
    lines += ["", "## Selected settings, 10 seeds (mean +- 95% CI; held-out seeds 5-9 in brackets)", "",
              "| model | setting | recall at end | recall fails (< 0.8) | SHD learned | **SHD final** | SHD forgetting |", "|---|---|---|---|---|---|---|"]
    out = {}
    for m in models:
        rs = sorted(cells[(m, sel[m])], key=lambda r: r["seed"])
        out[m] = rs
        lre = sorted({float(s[0].split("/")[1]) for s in grid[m]})
        cs = sorted({s[1] for s in grid[m] if s[1] is not None})
        e = float(sel[m][0].split("/")[1])
        edge = (" ⚠ lr edge" if len(lre) > 1 and e in (lre[0], lre[-1]) else "") + \
               (" ⚠ c edge" if len(cs) > 1 and sel[m][1] in (cs[0], cs[-1]) else "")
        f = lambda k, rr=rs: "{:.3f} ± {:.3f}".format(*mean_ci95([r[k] for r in rr])[:1], mean_ci95([r[k] for r in rr])[2]) + \
            " [{:.3f}]".format(sum(r[k] for r in rr if r["seed"] >= 5) / max(1, sum(r["seed"] >= 5 for r in rr)))
        fails = sum(r["recall_final"] < 0.8 for r in rs)
        lines.append(f"| {NAMES.get(m, m)} | lr {sel[m][0]}" + (f", c {sel[m][1]:g}" if sel[m][1] is not None else "") + edge +
                     f" | {f('recall_final')} | {fails}/{len(rs)} | {f('ecl_learned_avg')} | {f('ecl_final_avg')} | {f('ecl_forgetting')} |")
    lines += ["", "## Per task (mean over seeds): right after its own task / at the end", "", "| model | task 0 | task 1 | task 2 | task 3 | task 4 |",
              "|---|---|---|---|---|---|"]
    for m in models:
        rs = out[m]
        cols = []
        for t in range(5):
            after = sum(r["history"][t + 1]["ecl"][t] for r in rs) / len(rs)
            end = sum(r["history"][-1]["ecl"][t] for r in rs) / len(rs)
            cols.append(f"{after:.2f} / {end:.2f}")
        lines.append(f"| {NAMES.get(m, m)} | " + " | ".join(cols) + " |")
    lines += ["", f"## Paired per seed: {NAMES[FIRST]} vs each model (wins/losses, two-sided sign test; held-out seeds 5-9 in brackets)", "",
              "| vs | SHD final higher | SHD final: mean difference (95% CI) | SHD learned higher | forgetting lower | recall at end higher |",
              "|---|---|---|---|---|---|"]
    for m in models[1:]:
        A = {r["seed"]: r for r in out[FIRST]}
        B = {r["seed"]: r for r in out[m]}
        seeds = sorted(set(A) & set(B))

        def cmp(k, lower=False):
            w = sum((A[s][k] < B[s][k]) if lower else (A[s][k] > B[s][k]) for s in seeds)
            lo = sum((A[s][k] > B[s][k]) if lower else (A[s][k] < B[s][k]) for s in seeds)
            wh = sum((A[s][k] < B[s][k]) if lower else (A[s][k] > B[s][k]) for s in seeds if s >= 5)
            nh = sum(s >= 5 for s in seeds)
            return f"{w}/{len(seeds)} (lost {lo}), p = {sign_p(w, lo):.3f} [{wh}/{nh}]"
        dm, _, dh = mean_ci95([A[s]["ecl_final_avg"] - B[s]["ecl_final_avg"] for s in seeds])
        lines.append(f"| {NAMES.get(m, m)} | {cmp('ecl_final_avg')} | {dm:+.3f} ± {dh:.3f} | {cmp('ecl_learned_avg')} | {cmp('ecl_forgetting', True)} | {cmp('recall_final')} |")
    if a.task != "SHD":
        lines = [l.replace("SHD chance 0.25 (four words per task)", f"{a.task} chance 0.1").replace("Split-SHD summary", f"{a.task} sequence summary")
                 .replace("SHD", a.task) for l in lines]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        o = Path(a.out)
        o.mkdir(parents=True, exist_ok=True)
        (o / "summary.md").write_text(text)
        (o / "selected.json").write_text(json.dumps({m: {"lr": s[0], "si_c": s[1]} for m, s in sel.items()}, indent=1))


if __name__ == "__main__":
    main()
