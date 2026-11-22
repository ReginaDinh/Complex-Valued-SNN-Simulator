"""How often does training land in a poor solution? (after the seed-1 check, docs/results/speedup_chunk_2026-11)

    cd research && .venv/bin/python -m analysis.failure_rate

P1b, P = 8, the I21 set-ups with the selected ring [0.9, 0.999], step-by-step loop (as in I21). Seeds 0-9, each run
as in I21 (no nudge, imported from results/ep1b/i21) and twice more with the initial weights nudged by a relative 1e-6
(init_jitter 1, 2): 30 runs per network. Failure is defined before looking: test accuracy below 0.80.
Networks: cv_fw2 (rotating CV-SNN + complex plastic synapse), lif_fw2_d11 (LIF + real plastic synapse, equal memory).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("cv_fw2", "lif_fw2_d11")
FAIL = 0.80


def main() -> None:
    from cvsnn.utils import mean_ci95
    from experiments.ep1b import _run, jobs

    base = [j for j in jobs(10, 3000) if j["pairs"] == 8 and j["variant"] in VARIANTS and j["hp"]["leak_mag_range"] == [0.9, 0.999]]
    out = ROOT / "results" / "ep1b" / "failure_rate.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [{**j, "key": j["key"] + f"|jitter={i}", "init_jitter": i} for j in base for i in (1, 2)]
    todo = [j for j in todo if j["key"] not in done]
    with ProcessPoolExecutor(10) as pool, open(out, "a") as f:
        for j, r in zip(todo, pool.map(_run, todo)):
            f.write(json.dumps({**r, "key": j["key"], "jitter": j["init_jitter"]}) + "\n")
            f.flush()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    i21 = [json.loads(l) for l in (ROOT / "results" / "ep1b" / "i21" / "runs.jsonl").read_text().splitlines()]
    rows += [{**r, "jitter": 0} for r in i21 if r["variant"] in VARIANTS and r["hp"]["leak_mag_range"] == [0.9, 0.999]]
    L = ["# How often training lands in a poor solution (P1b, P = 8, I21 set-ups, loop)", "",
         f"30 runs per network: seeds 0-9 x (no nudge, two 1e-6 nudges of the initial weights). Failure: test accuracy < {FAIL}.", "",
         "| network | runs | mean ± 95% CI | median | failures | failure rate | lowest five |", "|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        a = sorted(r["test_acc"] for r in rows if r["variant"] == v)
        m, _, ci = mean_ci95(a)
        n_fail = sum(x < FAIL for x in a)
        L.append(f"| {v} | {len(a)} | {m:.3f} ± {ci:.3f} | {a[len(a) // 2]:.3f} | {n_fail} | {n_fail / len(a):.0%} | " + " / ".join(f"{x:.2f}" for x in a[:5]) + " |")
        by_seed = {}
        for r in sorted((r for r in rows if r["variant"] == v), key=lambda r: r["jitter"]):
            by_seed.setdefault(r["seed"], []).append(round(r["test_acc"], 2))
        L.append("")
        L.append(f"{v} by seed (no nudge, nudge 1, nudge 2): " + "; ".join(f"s{s}: {by_seed[s]}" for s in sorted(by_seed)))
        L.append("")
    md = "\n".join(L) + "\n"
    (ROOT / "results" / "ep1b" / "failure_rate.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
