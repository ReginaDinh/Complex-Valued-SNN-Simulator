"""B3b: strong replay plus synaptic consolidation (SI) for long lists in the memory system (plan B, PROJECT-STATUS.md).

    cd research && .venv/bin/python -m experiments.eb3b       # full (parallel, resumable with --out)

B3: strong replay (256 lists per version, half of every batch) brought the CV-SNN to 82.8% (joint training 90.3%) and
LIF to 64.2% (joint 66.4%). Here Synaptic Intelligence (Re/Im, cvsnn.consolidation) is added on top, strength c in
{1, 10}, lr 3e-3 (chosen in B2 and B3 for both networks), 5 seeds; c chosen on validation.
Question (one per run): does SI on top of strong replay bring the system closer to joint training?
Prediction written before running: a modest gain for the CV-SNN (about 85-88%), little for LIF (already at 97% of its
ceiling).
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
MODES = ("replay",)
STEPS = {"first": 3000, "later": 1000}
REPLAY = {"per_task": 256, "frac": 0.5}  # B2: 64 and 0.25
LRS = (3e-3,)
SI_C = (0.1, 1.0, 10.0)  # 0.1 added after the first pass chose 1, the lower edge
NETS = {
    "cv": {"kind": "cvsnn", "hidden": [18], "extra": {"reset": "none", **HET_CV, "fast_weights": {**FW2, "theta_tag": True, "chunk": 16}}},
    "lif": {"kind": "lif", "hidden": [30], "extra": {"reset": "none", "fast_weights": {**FW2, "d": 11, "theta_tag": True, "chunk": 16}}},
}
HP = {"threshold": 0.5, "leak_mag_range": [0.9, 0.999]}  # chosen on validation in I28b for both networks


def jobs(seeds: int, quick: bool) -> list[dict]:
    out = []
    for (name, v), mode, lr, c, seed in itertools.product(NETS.items(), MODES, LRS, SI_C, range(seeds)):
        model = {"kind": v["kind"], "hidden": v["hidden"], "readout_last": TASK["tail"], **v["extra"], **HP}
        out.append({"key": f"{name}|{mode}|lr={lr}|c={c}|s={seed}", "net": name, "mode": mode, "lr": lr, "si_c": c, "seed": seed, "model": model,
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
    from cvsnn.consolidation import SynapticIntelligence
    si = SynapticIntelligence(model, "cart", job["si_c"])
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
            loss = F.cross_entropy(model(x, theta=th)[0], y) + si.penalty()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            si.pre_step()
            opt.step()
            si.post_step()
        si.end_task()
        if job["mode"] == "replay":
            store.append(sample(task, REPLAY["per_task"], g_buf))
        history.append({"after_task": task, "accs": evaluate(tests)})
    final = history[-1]["accs"]
    best = [max(h["accs"][t] for h in history[t:]) for t in range(N_TASKS)]
    return {"key": job["key"], "net": job["net"], "mode": job["mode"], "lr": job["lr"], "si_c": job["si_c"], "seed": job["seed"], "params": model.num_params(),
            "final_avg_acc": sum(final) / N_TASKS, "forgetting": sum(best[t] - final[t] for t in range(N_TASKS - 1)) / (N_TASKS - 1),
            "learned": sum(h["accs"][h["after_task"]] for h in history) / N_TASKS, "final_avg_val_acc": sum(evaluate(vals)) / N_TASKS,
            "history": history}


def analyse(rows: list[dict], out: Path) -> str:
    from cvsnn.utils import mean_ci95

    mean = lambda rs, f: sum(f(r) for r in rs) / len(rs)
    L = ["# B3b: strong replay + SI on 24-item lists, 5 task versions (chance 0.125); SI strength c chosen on validation", "",
         "lr chosen per cell on validation. learned = accuracy on each task right after it.", "",
         "| network | condition (lr 3e-3) | params | final avg acc | forgetting (↓) | learned | per-seed forgetting |", "|---|---|---|---|---|---|---|---|"]
    sel = {}
    for name in NETS:
        for rp in MODES:
            groups = {}
            for r in rows:
                if r["net"] == name and r["mode"] == rp:
                    groups.setdefault(r["si_c"], []).append(r)
            if not groups:
                continue
            lr = max(groups, key=lambda k: mean(groups[k], lambda r: r["final_avg_val_acc"]))
            rs = sorted(groups[lr], key=lambda r: r["seed"])
            a, f, le = mean_ci95([r["final_avg_acc"] for r in rs]), mean_ci95([r["forgetting"] for r in rs]), mean_ci95([r["learned"] for r in rs])
            sel[f"{name}|{rp}"] = {"lr": lr, "final_avg_acc": a, "forgetting": f, "learned": le,
                                   "acc_matrix": [[mean(rs, lambda r: r["history"][i]["accs"][t]) for t in range(N_TASKS)] for i in range(N_TASKS)]}
            L.append(f"| {name} | {rp} + SI c={lr:g} | {rs[0]['params']} | {a[0]:.3f} ± {a[2]:.3f} | {f[0]:.3f} ± {f[2]:.3f} | {le[0]:.3f} | "
                     + ", ".join(f"{r['forgetting']:.2f}" for r in rs) + " |")
    (out / "selected.json").write_text(json.dumps(sel, indent=2))
    md = "\n".join(L) + "\n"
    (out / "summary.md").write_text(md)
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = 1 if a.quick else 5
    out = Path(a.out) if a.out else ROOT / "results" / ("eb3b_quick" if a.quick else "eb3b") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rp = out / "runs.jsonl"
    done = {json.loads(l)["key"] for l in rp.read_text().splitlines()} if rp.exists() else set()
    todo = [j for j in jobs(seeds, a.quick) if j["key"] not in done]
    import shutil
    from cvsnn.utils import environment, git_state, write_json
    for d in ("cvsnn", "experiments"):
        shutil.copytree(ROOT / d, out / "code_snapshot" / f"{datetime.now():%Y%m%d-%H%M%S}" / d, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(out / "meta.json", {"git": git_state(ROOT.parent), "env": environment(), "nets": NETS, "hp": HP, "steps": STEPS, "replay": REPLAY,
                                   "lrs": LRS, "si_c": SI_C, "task": TASK, "n_tasks": N_TASKS, "n_channels": N_CH, "modes": MODES})
    print(f"{len(todo)} runs to do ({len(done)} done), {a.workers} workers -> {out}", flush=True)
    with ProcessPoolExecutor(a.workers) as pool, open(rp, "a") as f:
        futs = [pool.submit(_run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(futs):
                print(f"  {i}/{len(futs)}  {datetime.now():%H:%M:%S}", flush=True)
    print(analyse([json.loads(l) for l in rp.read_text().splitlines()], out))
    print(f"results: {out}")


if __name__ == "__main__":
    main()
