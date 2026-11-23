"""N10 (theory check for Proposition 1): after training, do the rotations of the long-memory cLIF stay away from 0? N12 (control): do the no-rotation long-memory cLIF and LIF long also shorten their memories during training? cd research && .venv/bin/python -m analysis.n10_omega [out_dir] [seeds] [workers] # N10 cd research && .venv/bin/python -m analysis.n10_omega results/n12 10 10 --cells norot,lif --also results/n10 # N12 cd research && .venv/bin/python -m analysis.n10_omega results/n13 10 10 --cells rot_lr1e-1,norot_lr1e-2,lif_lr1e-2 --also results/n10,results/n12 # N13 cd research && .venv/bin/python -m analysis.n10_omega results/n17 10 10 --cells norot_lr1e-2_rs3,lif_lr1e-2_rs3 --also results/n10,results/n13 # N17 The pile-up bound mu / sin(omega) only helps while omega is clearly non-zero. This reruns the selected rotating long-memory cell (cand_long, plastic synapse x0.03, recall lr 1e-2) at its selected settings on the three continual-learning tasks of the paper, 10 seeds, and records (r, omega) of every hidden neuron at initialisation, after the recall phase and at the end: - synthetic sequence (I54i): two heads, E-CL lr 3e-4; - permuted MNIST + replay 100 (I64b): 120 hidden, MNIST-phase lr 1e-3; - Split-SHD + replay 100 (I64a): 120 hidden, SHD-phase lr 1e-3. Reports |omega| quantiles, the share of neurons with |omega| < 0.05, and the pile-up gain 1/|1 - lambda| next to the gain 1/(1 - r) the same neuron would have without rotation. Reproduction check: the final metrics must equal the original runs (same keys, same code path). """

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {  # name: (job extras, lr_ecl, original runs.jsonl)
    "synthetic (I54i)": ({}, 3e-4, ROOT / "results/ei54i/run1/runs.jsonl"),
    "permuted MNIST + replay (I64b)": ({"mnist": True, "hidden": 120, "replay": 100}, 1e-3, ROOT / "results/ei64/mnist/runs.jsonl"),
    "Split-SHD + replay (I64a)": ({"shd": True, "hidden": 120, "replay": 100}, 1e-3, ROOT / "results/ei64/shd/runs.jsonl"),
}
TAGS = {"synthetic (I54i)": "", "permuted MNIST + replay (I64b)": "|mnist|h120|rp100", "Split-SHD + replay (I64a)": "|shd|h120|rp100"}
# cell: (name, variant, recall lr, later-phase lr per setting), each at the setting selected before (I54i / I54g, I64a, I64b)
CELLS = {"rot": ("rotating long", "cand_long_fw003", 1e-2, (3e-4, 1e-3, 1e-3)),
         "norot": ("no-rotation long", "cand_longnorot_fw001", 1e-1, (3e-4, 3e-3, 1e-3)),
         "lif": ("LIF long", "lif_fw2_d11_fw001", 1e-1, (1e-3, 3e-3, 1e-3)),
         # N13: the same cells with the recall lr swapped (synthetic sequence only; None = setting not run)
         "rot_lr1e-1": ("rotating long @ recall lr 1e-1", "cand_long_fw003", 1e-1, (3e-4, None, None)),
         "norot_lr1e-2": ("no-rotation long @ recall lr 1e-2", "cand_longnorot_fw001", 1e-2, (3e-4, None, None)),
         "lif_lr1e-2": ("LIF long @ recall lr 1e-2", "lif_fw2_d11_fw001", 1e-2, (1e-3, None, None)),
         # N17: the two controls at recall lr 1e-2 with a 3x recall budget (13 500 steps; keys tagged |rs13500)
         "norot_lr1e-2_rs3": ("no-rotation long @ recall lr 1e-2, 13 500 steps", "cand_longnorot_fw001", 1e-2, (3e-4, None, None), 13500),
         "lif_lr1e-2_rs3": ("LIF long @ recall lr 1e-2, 13 500 steps", "lif_fw2_d11_fw001", 1e-2, (1e-3, None, None), 13500)}


def jobs(out: Path, seeds: int, cells=("rot",)) -> list[dict]:
    from experiments.ei54 import ECL_STEPS, RECALL_STEPS, phase_jobs
    js = []
    for c in cells:
        cname, variant, lr_r, lr_es, *rest = CELLS[c]
        rs = rest[0] if rest else RECALL_STEPS
        rtag = f"|rs{rs}" if rs != RECALL_STEPS else ""   # as ei54 --recall-steps: longer runs never mix with 4500-step ones
        for (name, (extra, _, _)), lr_e in zip(SETTINGS.items(), lr_es):
            if lr_e is None:
                continue
            steps = ECL_STEPS if not extra else 600
            for j in phase_jobs(seeds, rs, steps, lrs=[(lr_r, lr_e)], variants=(variant,)):
                if j["two_heads"]:
                    key = j["key"] + TAGS[name] + rtag
                    js.append({**j, **extra, "key": key, "variant": j["variant"] + TAGS[name] + rtag, "setting": name, "cell": cname,
                               "cache_dir": str(out / "cache"), "save_model": str(out / "models" / (key.replace("|", "_").replace("=", "") + ".pt"))})
    return js


def _one(job: dict) -> dict:
    from experiments.ei54 import _run
    r = _run(job)
    r.pop("history", None)
    return {**r, "setting": job["setting"], "cell": job["cell"], "save_model": job["save_model"]}


def wrap(om):
    """|omega| as an angle in [0, pi]: a learned omega near 2 pi is (almost) no rotation per step."""
    import torch
    return torch.atan2(torch.sin(om), torch.cos(om)).abs()


def stats(params: list[dict]) -> dict:
    import torch
    r = torch.cat([p["r"] for p in params]).double()
    om = wrap(torch.cat([p["omega"] for p in params]).double())
    lam = torch.polar(r, om)
    gain, gain0 = 1 / (1 - lam).abs(), 1 / (1 - r)
    q = lambda x, k: float(torch.quantile(x, k))
    return {"omega_q": [q(om, k) for k in (0.05, 0.5, 0.95)], "omega_lt005": float((om < 0.05).double().mean()),
            "omega_min": float(om.min()), "gain_q": [q(gain, k) for k in (0.5, 0.95)], "gain_max": float(gain.max()),
            "gain0_q": [q(gain0, k) for k in (0.5, 0.95)], "r_q": [q(r, k) for k in (0.05, 0.5, 0.95)], "n": len(r)}


def main(out: Path, seeds: int = 10, workers: int = 10, cells=("rot",), also=()) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import torch
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    rp = out / "runs.jsonl"
    done = {json.loads(l)["key"] for l in rp.read_text().splitlines()} if rp.exists() else set()
    todo = [j for j in jobs(out, seeds, cells) if j["key"] not in done]
    print(f"{len(todo)} runs to do ({len(done)} done)", flush=True)
    with ProcessPoolExecutor(workers) as pool, open(rp, "a") as f:
        for i, fu in enumerate(as_completed([pool.submit(_one, j) for j in todo]), 1):
            f.write(json.dumps(fu.result()) + "\n")
            f.flush()
            print(f" {i}/{len(todo)}", flush=True)
    rows = []
    for d in (out, *map(Path, also)):   # N12: the N10 rows (rotating long) join the table
        rows += [{"cell": "rotating long", **json.loads(l)} for l in (d / "runs.jsonl").read_text().splitlines()]
    order = [CELLS[c][0] for c in CELLS]
    groups = [(c, name) for c in order for name in SETTINGS if any(r["cell"] == c and r["setting"] == name for r in rows)]
    L = ["# N10 / N12: memory (r) and rotation (omega) of the hidden neurons after training (selected settings, pooled over seeds)", "",
         "|omega| quantiles 5 / 50 / 95%; pile-up gain 1/|1 - lambda| (median / 95% / max) vs 1/(1 - r) without rotation.", "",
         "| model | setting | seeds | stage | \\|omega\\| q05 / q50 / q95 | share \\|omega\\| < 0.05 | gain 1/\\|1-λ\\| q50 / q95 / max | 1/(1-r) q50 / q95 | r q05 / q50 / q95 |",
         "|---|---|---|---|---|---|---|---|---|"]
    summary, repro, finals = {}, {}, {}
    for c, name in groups:
        rs = sorted((r for r in rows if r["cell"] == c and r["setting"] == name), key=lambda r: r["seed"])
        saved = [torch.load(r["save_model"], weights_only=False) for r in rs]
        finals[(c, name)] = [p for sv in saved for p in sv["final"]]
        for stage in ("init", "phase1", "final"):
            st = stats([p for sv in saved for p in sv[stage]])
            summary[f"{c}|{name}|{stage}"] = st
            L.append(f"| {c} | {name} | {len(rs)} | {stage} | {st['omega_q'][0]:.3f} / {st['omega_q'][1]:.3f} / {st['omega_q'][2]:.3f} | "
                     f"{st['omega_lt005']:.1%} | {st['gain_q'][0]:.1f} / {st['gain_q'][1]:.1f} / {st['gain_max']:.1f} | "
                     f"{st['gain0_q'][0]:.0f} / {st['gain0_q'][1]:.0f} | {st['r_q'][0]:.3f} / {st['r_q'][1]:.3f} / {st['r_q'][2]:.3f} |")
        per = []
        for r, sv in zip(rs, saved):
            rr = torch.cat([p["r"] for p in sv["final"]])
            om = wrap(torch.cat([p["omega"] for p in sv["final"]]))
            per.append(f"s{r['seed']}: r {float(rr.median()):.2f}, |ω| {float(om.median()):.2f}")
        L.append(f"| | | | per seed (final medians) | {'; '.join(per)} | | | | |")
        ref = {}
        for f in [SETTINGS[name][2], *sorted((ROOT / "results/ei54g").rglob("runs.jsonl"))]:   # LIF seeds 5-9: I54g-B
            for l in f.read_text().splitlines():
                o = json.loads(l)
                ref.setdefault(o["key"], o)
        same = [all(abs(r[k] - ref[r["key"]][k]) < 1e-12 for k in ("recall_final", "ecl_final_avg", "ecl_forgetting", "val_score"))
                for r in rs if r["key"] in ref]
        repro[f"{c} | {name}"] = f"{sum(same)}/{len(same)} identical (of {len(rs)} runs; {len(rs) - len(same)} without an original)"
    # joint view: rotation and pile-up gain per memory band (r) at the end, pooled over seeds
    L += ["", "By memory band at the end (pooled over seeds): share of neurons, median |omega|, share |omega| < 0.05, pile-up gain", "",
          "| model | setting | r band | share of neurons | median \\|omega\\| | share \\|omega\\| < 0.05 | gain 1/\\|1-λ\\| median / max | 1/(1-r) median |",
          "|---|---|---|---|---|---|---|---|"]
    for (c, name), fin in finals.items():
        rr = torch.cat([p["r"] for p in fin]).double()
        om = wrap(torch.cat([p["omega"] for p in fin]).double())
        gain, gain0 = 1 / (1 - torch.polar(rr, om)).abs(), 1 / (1 - rr)
        for lo, hi in ((0.0, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0)):
            m = (rr >= lo) & (rr < hi)
            if m.any():
                L.append(f"| {c} | {name} | [{lo:g}, {hi:g}) | {float(m.double().mean()):.1%} | {float(om[m].median()):.3f} | {float((om[m] < 0.05).double().mean()):.1%} | "
                         f"{float(gain[m].median()):.1f} / {float(gain[m].max()):.1f} | {float(gain0[m].median()):.1f} |")
    # N12: per seed, does the rotating network keep longer memories than each control?
    def per_seed(c, name):
        out_ = {}
        for r in rows:
            if r["cell"] == c and r["setting"] == name:
                rr = torch.cat([p["r"] for p in torch.load(r["save_model"], weights_only=False)["final"]])
                out_[r["seed"]] = (float(rr.median()), float((rr >= 0.8).float().mean()))
        return out_
    lr_of = {}
    for r in rows:
        lr_of[(r["cell"], r["setting"])] = r["lr_rec"]
    rots = [c for c in order if c.startswith("rotating") and any(r["cell"] == c for r in rows)]
    ctrls = [c for c in order if not c.startswith("rotating") and any(r["cell"] == c for r in rows)]
    same_lr = [(a, b, name) for name in SETTINGS for a in rots for b in ctrls
               if (a, name) in lr_of and (b, name) in lr_of and lr_of[(a, name)] == lr_of[(b, name)]]
    if same_lr:
        L += ["", "Per seed, same recall lr: rotating vs control (median r higher; share r >= 0.8 higher), and the recall / validation score:", "",
              "| setting | recall lr | rotating | control | median r higher | share r >= 0.8 higher | share r >= 0.8: rot / ctrl | recall at end: rot / ctrl | val score: rot / ctrl |",
              "|---|---|---|---|---|---|---|---|---|"]
        for a, b, name in same_lr:
            ra, rb = per_seed(a, name), per_seed(b, name)
            seeds = sorted(set(ra) & set(rb))
            m = lambda c, k: sum(r[k] for r in rows if r["cell"] == c and r["setting"] == name) / len([r for r in rows if r["cell"] == c and r["setting"] == name])
            L.append(f"| {name} | {lr_of[(a, name)]:g} | {a} | {b} | {sum(ra[s][0] > rb[s][0] for s in seeds)}/{len(seeds)} | {sum(ra[s][1] > rb[s][1] for s in seeds)}/{len(seeds)} | "
                     f"{sum(ra[s][1] for s in seeds) / len(seeds):.1%} / {sum(rb[s][1] for s in seeds) / len(seeds):.1%} | "
                     f"{m(a, 'recall_final'):.3f} / {m(b, 'recall_final'):.3f} | {m(a, 'val_score'):.3f} / {m(b, 'val_score'):.3f} |")
    others = [c for c in order if c != "rotating long" and not c.startswith("rotating") and "@" not in c and any(r["cell"] == c for r in rows)]
    if others and any(r["cell"] == "rotating long" for r in rows):
        L += ["", "Per seed at the end: rotating long vs each control (median r higher; share of neurons with r >= 0.8 higher):", "",
              "| setting | control | median r higher | share r >= 0.8 higher | share r >= 0.8: rotating / control |", "|---|---|---|---|---|"]
        for name in SETTINGS:
            rot = per_seed("rotating long", name)
            for c in others:
                o_ = per_seed(c, name)
                seeds = sorted(set(rot) & set(o_))
                if seeds:
                    L.append(f"| {name} | {c} | {sum(rot[s][0] > o_[s][0] for s in seeds)}/{len(seeds)} | {sum(rot[s][1] > o_[s][1] for s in seeds)}/{len(seeds)} | "
                             f"{sum(rot[s][1] for s in seeds) / len(seeds):.1%} / {sum(o_[s][1] for s in seeds) / len(seeds):.1%} |")
    L += ["", "Reproduction (final recall, E-CL average, forgetting, validation score equal to the original run of the same key):", ""]
    L += [f"- {k}: {v}" for k, v in repro.items()]
    (out / "summary.json").write_text(json.dumps({"stats": summary, "repro": repro}, indent=1))
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "results/n10"))
    ap.add_argument("seeds", nargs="?", type=int, default=10)
    ap.add_argument("workers", nargs="?", type=int, default=10)
    ap.add_argument("--cells", default="rot", help="comma-separated keys of CELLS (rot, norot, lif)")
    ap.add_argument("--also", default="", help="comma-separated result directories whose runs join the summary (N12: results/n10)")
    a = ap.parse_args()
    main(Path(a.out), a.seeds, a.workers, tuple(a.cells.split(",")), tuple(x for x in a.also.split(",") if x))
