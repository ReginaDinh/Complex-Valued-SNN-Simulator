"""N15: standard SHD test accuracy split by speaker (speakers seen in training vs the two new test speakers).

    cd research && .venv/bin/python -m analysis.n15_speakers [out_dir] [seeds] [workers]

Retrains three models of I66 / I68 at their selected lr, 80 epochs (rotating long-memory cLIF 1e-3, SE-adLIF 1e-3, LIF short
3e-3; same code path, so the overall test accuracy must equal the original runs of the same key), records per-speaker test
accuracy, and saves the trained models (for N16).
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLS = {"cand_long_fw003": 1e-3, "se_adlif_fw001": 1e-3, "lif_short_fw003": 3e-3}
ORIGINALS = ("results/ei68/brfv2_cmp/runs.jsonl", "results/ei66/long80_all/runs.jsonl", "results/ei68/long80_all/runs.jsonl")


def main(out: Path, seeds: int = 4, workers: int = 12) -> None:
    from cvsnn.utils import mean_ci95
    from experiments.ei66 import _run, jobs
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    rp = out / "runs.jsonl"
    done = {json.loads(l)["key"] for l in rp.read_text().splitlines()} if rp.exists() else set()
    js = [{**j, "by_speaker": True, "save_model": str(out / "models" / (j["key"].replace("|", "_").replace("=", "") + ".pt"))}
          for v, lr in CELLS.items() for j in jobs(seeds, 80, (lr,), (v,))]
    todo = [j for j in js if j["key"] not in done]
    print(f"{len(todo)} runs to do ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(workers) as pool, open(rp, "a") as f:
        for i, fu in enumerate(as_completed([pool.submit(_run, j) for j in todo]), 1):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            print(f"  {i}/{len(todo)}", flush=True)
    rows = [json.loads(l) for l in rp.read_text().splitlines()]
    ref = {}
    for f_ in ORIGINALS:
        for l in (ROOT / f_).read_text().splitlines():
            o = json.loads(l)
            ref.setdefault(o["key"], o)
    L = ["# N15: standard SHD test accuracy by speaker (80 epochs, selected lr)", "",
         "| model | seeds | validation | test (all) | **test, known speakers** | **test, new speakers (4, 5)** | gap known - new (paired) | new < known in |",
         "|---|---|---|---|---|---|---|---|"]
    res = {}
    for v in CELLS:
        rs = sorted((r for r in rows if r["variant"] == v), key=lambda r: r["seed"])
        if not rs:
            continue
        m = {k: mean_ci95([r[k] for r in rs]) for k in ("val", "test", "test_known", "test_new")}
        gap = mean_ci95([r["test_known"] - r["test_new"] for r in rs])
        res[v] = {k: m[k] for k in m} | {"gap": gap, "per_speaker": {s: sum(r["test_by_speaker"][s] for r in rs) / len(rs) for s in rs[0]["test_by_speaker"]}}
        L.append(f"| {v} | {len(rs)} | {m['val'][0]:.3f} | {m['test'][0]:.3f} ± {m['test'][2]:.3f} | **{m['test_known'][0]:.3f} ± {m['test_known'][2]:.3f}** | "
                 f"**{m['test_new'][0]:.3f} ± {m['test_new'][2]:.3f}** | {gap[0]:+.3f} ± {gap[2]:.3f} | {sum(r['test_new'] < r['test_known'] for r in rs)}/{len(rs)} |")
    L += ["", f"Samples: {rows[0]['n_known']} from known speakers, {rows[0]['n_new']} from the two new speakers.", "", "Per speaker (mean over seeds):", ""]
    for v, x in res.items():
        L.append(f"- {v}: " + ", ".join(f"speaker {s}: {a:.3f}" for s, a in sorted(x["per_speaker"].items(), key=lambda kv: int(kv[0]))))
    same = [abs(r["test"] - ref[r["key"]]["test"]) < 1e-12 for r in rows if r["key"] in ref]
    L += ["", f"Reproduction: overall test accuracy identical to the original run of the same key in {sum(same)}/{len(same)} runs "
              f"({len(rows) - len(same)} without an original)."]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/n15",
         int(sys.argv[2]) if len(sys.argv) > 2 else 4, int(sys.argv[3]) if len(sys.argv) > 3 else 12)
