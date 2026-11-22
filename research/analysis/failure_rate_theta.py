"""I32: how often do the theta-tagged networks land in a poor solution? (after analysis.failure_rate on P1b)

    cd research && .venv/bin/python -m analysis.failure_rate_theta

Two theta tasks, the tagged CV-SNN and LIF with the ring chosen on validation ([0.9, 0.999]), step-by-step loop as in
the original runs: I28b (8 items, one theta cycle) and I28d at 3 cycles (24 items). Seeds 0-9, each without a nudge
(seeds 0-4 imported from the original runs) and twice with the initial weights nudged by a relative 1e-6: 30 runs per
network and task. Failure is defined before looking: test accuracy below 0.80.
Prediction written before running: I28b: no failures for either network (both ~100%); 24 items: CV-SNN rarely fails
(about 10%), LIF often (about 50%).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = 0.80
NETS = ("cv_fw2_tag", "lif_fw2_d11_tag")
RING = [0.9, 0.999]


def _job_list():
    from experiments.ei28b import jobs as jobs_b
    from experiments.ei28d import jobs as jobs_d

    out = []
    for task, js in (("i28b_8items", jobs_b(10, 3000)), ("i28d_24items", [j for j in jobs_d(10, 3000) if j["cycles"] == 3])):
        for j in js:
            if j["variant"] in NETS and j["hp"]["leak_mag_range"] == RING:
                out.append({**j, "task_name": task})
    return out


def _run(job: dict) -> dict:
    from experiments.ei28b import _run as run_b
    return {**run_b(job), "task_name": job["task_name"], "jitter": job.get("init_jitter", 0)}


def main() -> None:
    from cvsnn.utils import mean_ci95

    base = _job_list()
    out = ROOT / "results" / "failure_rate_theta.jsonl"
    done = {json.loads(l)["key"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [j for j in base if j["seed"] >= 5]                                   # seeds 5-9 without a nudge
    todo += [{**j, "key": j["key"] + f"|jitter={i}", "init_jitter": i} for j in base for i in (1, 2)]
    todo = [{**j, "key": j["task_name"] + "|" + j["key"]} for j in todo]
    todo = sorted((j for j in todo if j["key"] not in done), key=lambda j: j["task_name"] != "i28d_24items")   # long ones first
    with ProcessPoolExecutor(10) as pool, open(out, "a") as f:
        for j, r in zip(todo, pool.map(_run, todo)):
            f.write(json.dumps({**r, "key": j["key"]}) + "\n")
            f.flush()
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    for task, src in (("i28b_8items", ROOT / "results" / "ei28b" / "full" / "runs.jsonl"), ("i28d_24items", ROOT / "results" / "ei28d" / "full" / "runs.jsonl")):
        for l in src.read_text().splitlines():
            r = json.loads(l)
            if r["variant"] in NETS and r["hp"]["leak_mag_range"] == RING and r["seed"] < 5 and r.get("cycles", 3) == 3:
                rows.append({**r, "task_name": task, "jitter": 0})
    L = ["# I32: how often the theta-tagged networks land in a poor solution (loop, ring [0.9, 0.999])", "",
         f"30 runs per network and task: seeds 0-9 x (no nudge, two 1e-6 nudges of the initial weights). Failure: test accuracy < {FAIL}.", "",
         "| task | network | runs | mean ± 95% CI | median | failures | failure rate | lowest five |", "|---|---|---|---|---|---|---|---|"]
    for task in ("i28b_8items", "i28d_24items"):
        for v in NETS:
            a = sorted(r["test_acc"] for r in rows if r["task_name"] == task and r["variant"] == v)
            if not a:
                continue
            m, _, ci = mean_ci95(a)
            nf = sum(x < FAIL for x in a)
            L.append(f"| {task} | {v} | {len(a)} | {m:.3f} ± {ci:.3f} | {a[len(a) // 2]:.3f} | {nf} | {nf / len(a):.0%} | " + " / ".join(f"{x:.2f}" for x in a[:5]) + " |")
    md = "\n".join(L) + "\n"
    (ROOT / "results" / "failure_rate_theta.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
