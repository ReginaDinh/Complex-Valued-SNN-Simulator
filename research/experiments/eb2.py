"""B2: the memory system of B1v2 on long lists (24 items = 3 theta cycles), where the tagged CV-SNN had the edge on a
single task (I28d: 94% vs LIF 70%). Step 2 of plan B (PROJECT-STATUS.md).

    cd research && .venv/bin/python -m experiments.eb2        # full (parallel, resumable with --out)
    .venv/bin/python -m experiments.eb2 --quick

Task: ordered recall of 24 items, 8 gamma slots per theta cycle (slots n and n + 8 share a theta phase; at the query
the replayed phase comes with a cue for the cycle), distributed +-1 symbol codes over 16 channels, 5 task versions that
permute the 16 channels, as in B1v2. Training: 3000 steps on the first version, 1000 on each later one, batch 64;
lr {1e-3, 3e-3} chosen on validation. Same networks as B1v2 (chunk 16).
Conditions: replay (64 stored lists per version, a quarter of every batch) and joint (all versions mixed; the
capacity ceiling of each network on this task).
Question (one per run): does the CV-SNN's advantage on long lists hold inside the continual-learning system with
replay?
Prediction written before running: yes: CV-SNN above LIF both with replay and with joint training (as in I28d), and
replay close to joint for both.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from experiments.ei28b import FW2, HET_CV

TASK = dict(items=24, n_symbols=8, slot=4, delay=40, tail=5, cycle_slots=8, cycle_cue=True)

ROOT = Path(__file__).resolve().parents[1]
N_TASKS = 5
N_CH = 16
MODES = ("replay", "joint")
STEPS = {"first": 3000, "later": 1000}
REPLAY = {"per_task": 64, "frac": 0.25}
LRS = (1e-3, 3e-3, 1e-2)  # 1e-2 added after the first pass chose 3e-3, the grid edge, in every cell
NETS = {
    "cv": {"kind": "cvsnn", "hidden": [18], "extra": {"reset": "none", **HET_CV, "fast_weights": {**FW2, "theta_tag": True, "chunk": 16}}},
    "lif": {"kind": "lif", "hidden": [30], "extra": {"reset": "none", "fast_weights": {**FW2, "d": 11, "theta_tag": True, "chunk": 16}}},
}
HP = {"threshold": 0.5, "leak_mag_range": [0.9, 0.999]}  # chosen on validation in I28b for both networks


def jobs(seeds: int, quick: bool) -> list[dict]:
    out = []
    for (name, v), mode, lr, seed in itertools.product(NETS.items(), MODES, LRS, range(seeds)):
        model = {"kind": v["kind"], "hidden": v["hidden"], "readout_last": TASK["tail"], **v["extra"], **HP}
        out.append({"key": f"{name}|{mode}|lr={lr}|s={seed}", "net": name, "mode": mode, "lr": lr, "seed": seed, "model": model,
                    "steps": {"first": 10, "later": 5} if quick else STEPS})
    return out


def _run(job: dict) -> dict:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    from cvsnn.data.synthetic import serial_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from cvsnn.utils import seed_all
    from experiments.ei28b import network_input

    seed_all(job["seed"])
    cplx = job["model"]["kind"] == "cvsnn"
    ns = TASK["n_symbols"]
    codes = torch.randint(0, 2, (ns, N_CH), generator=torch.Generator().manual_seed(900 + job["seed"])).float() * 2 - 1  # +-1 symbol codes
    perms = [torch.arange(N_CH)] + [torch.randperm(N_CH, generator=torch.Generator().manual_seed(700 + 31 * job["seed"] + i)) for i in range(1, N_TASKS)]

    def sample(task, n, g):
        """task: an int, or None for a random task per list (joint training)."""
        x, theta, y, _ = serial_recall_batch(n, generator=g, **TASK)
        dist = x[..., :ns] @ codes                                        # (T, B, 16): the code of the symbol shown, 0 elsewhere
        tk = torch.randint(0, N_TASKS, (n,), generator=g) if task is None else torch.full((n,), task)
        idx = torch.stack([perms[int(t)] for t in tk], 0)                  # (B, 16)
        dist = torch.gather(dist, 2, idx.unsqueeze(0).expand(dist.shape[0], -1, -1))
        return network_input(torch.cat([dist, x[..., ns:]], -1), theta, cplx), theta, y

    model = SequenceClassifier(sample(0, 1, torch.Generator().manual_seed(0))[0].shape[-1], ns, ModelConfig(**job["model"]))
    opt = torch.optim.AdamW(model.parameters(), lr=job["lr"])
    tests = [sample(t, 512, torch.Generator().manual_seed(20_000 + 7 * job["seed"] + t)) for t in range(N_TASKS)]
    vals = [sample(t, 256, torch.Generator().manual_seed(40_000 + 7 * job["seed"] + t)) for t in range(N_TASKS)]
    g_tr, g_buf = torch.Generator().manual_seed(10_000 + job["seed"]), torch.Generator().manual_seed(30_000 + job["seed"])
    store = []

    def evaluate(sets):
        model.eval()
        with torch.no_grad():
            accs = [(model(x, theta=th)[0].argmax(-1) == y).float().mean().item() for x, th, y in sets]
        model.train()
        return accs

    history = []
    for task in range(N_TASKS):
        for _ in range(job["steps"]["first" if task == 0 else "later"]):
            B = 64
            if job["mode"] == "joint":
                x, th, y = sample(None, B, g_tr)
            elif job["mode"] == "replay" and store:
                n_old = round(REPLAY["frac"] * B)
                x, th, y = sample(task, B - n_old, g_tr)
                bx, bth, by = (torch.cat([s[i] for s in store], 1 if i < 2 else 0) for i in range(3))
                idx = torch.randint(0, len(by), (n_old,), generator=g_buf)
                x, th, y = torch.cat([x, bx[:, idx]], 1), torch.cat([th, bth[:, idx]], 1), torch.cat([y, by[idx]])
            else:
                x, th, y = sample(task, B, g_tr)
            loss = F.cross_entropy(model(x, theta=th)[0], y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if job["mode"] == "replay":
            store.append(sample(task, REPLAY["per_task"], g_buf))
        history.append({"after_task": task, "accs": evaluate(tests)})
    final = history[-1]["accs"]
    best = [max(h["accs"][t] for h in history[t:]) for t in range(N_TASKS)]
    return {"key": job["key"], "net": job["net"], "mode": job["mode"], "lr": job["lr"], "seed": job["seed"], "params": model.num_params(),
            "final_avg_acc": sum(final) / N_TASKS, "forgetting": sum(best[t] - final[t] for t in range(N_TASKS - 1)) / (N_TASKS - 1),
            "learned": sum(h["accs"][h["after_task"]] for h in history) / N_TASKS, "final_avg_val_acc": sum(evaluate(vals)) / N_TASKS,
            "history": history}


def analyse(rows: list[dict], out: Path) -> str:
    from cvsnn.utils import mean_ci95

    mean = lambda rs, f: sum(f(r) for r in rs) / len(rs)
    L = ["# B2: the B1v2 memory system on 24-item lists (3 theta cycles), 5 task versions (chance 0.125)", "",
         "lr chosen per cell on validation. learned = accuracy on each task right after it.", "",
         "| network | mode | lr | params | final avg acc | forgetting (↓) | learned | per-seed forgetting |", "|---|---|---|---|---|---|---|---|"]
    sel = {}
    for name in NETS:
        for rp in MODES:
            groups = {}
            for r in rows:
                if r["net"] == name and r["mode"] == rp:
                    groups.setdefault(r["lr"], []).append(r)
            if not groups:
                continue
            lr = max(groups, key=lambda k: mean(groups[k], lambda r: r["final_avg_val_acc"]))
            rs = sorted(groups[lr], key=lambda r: r["seed"])
            a, f, le = mean_ci95([r["final_avg_acc"] for r in rs]), mean_ci95([r["forgetting"] for r in rs]), mean_ci95([r["learned"] for r in rs])
            sel[f"{name}|{rp}"] = {"lr": lr, "final_avg_acc": a, "forgetting": f, "learned": le,
                                   "acc_matrix": [[mean(rs, lambda r: r["history"][i]["accs"][t]) for t in range(N_TASKS)] for i in range(N_TASKS)]}
            L.append(f"| {name} | {rp} | {lr:g} | {rs[0]['params']} | {a[0]:.3f} ± {a[2]:.3f} | {f[0]:.3f} ± {f[2]:.3f} | {le[0]:.3f} | "
                     + ", ".join(f"{r['forgetting']:.2f}" for r in rs) + " |")
    (out / "selected.json").write_text(json.dumps(sel, indent=2))
    md = "
".join(L) + "
"
    (out / "summary.md").write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = 1 if a.quick else 5
    out = Path(a.out) if a.out else ROOT / "results" / ("eb2_quick" if a.quick else "eb2") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "runs.jsonl"
    done = {json.loads(l)["key"] for l in rp.read_text().splitlines()} if rp.exists() else set()
    todo = [j for j in jobs(seeds, a.quick) if j["key"] not in done]
    import shutil
    from cvsnn.utils import environment, git_state, write_json
    for d in ("cvsnn", "experiments"):
        shutil.copytree(ROOT / d, out / "code_snapshot" / f"{datetime.now():%Y%m%d-%H%M%S}" / d, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(out / "meta.json", {"git": git_state(ROOT.parent), "env": environment(), "nets": NETS, "hp": HP, "steps": STEPS, "replay": REPLAY,
                                   "lrs": LRS, "task": TASK, "n_tasks": N_TASKS, "n_channels": N_CH, "modes": MODES})
    print(f"{len(todo)} runs to do ({len(done)} done), {a.workers} workers -> {out}", flush=True)
    with ProcessPoolExecutor(a.workers) as pool, open(rp, "a") as f:
        futs = [pool.submit(_run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fu.result()) + "
")
            f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  {datetime.now():%H:%M:%S}", flush=True)
    print(analyse([json.loads(l) for l in rp.read_text().splitlines()], out))


if __name__ == "__main__":
    main()
