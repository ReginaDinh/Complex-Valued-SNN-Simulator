"""N17 follow-up: are the long-memory neurons that the controls keep actually in use, or saturated / silent?
N18: the same probe on the models trained on permuted MNIST + replay and Split-SHD + replay (saved by N10 / N12).

    cd research && .venv/bin/python -m analysis.n17_long_neuron_activity [out_dir]            # N17 (synthetic)
    cd research && .venv/bin/python -m analysis.n17_long_neuron_activity results/n18 --benchmarks   # N18

Loads the final trained models of the synthetic-sequence runs saved by analysis/n10_omega.py (N10 rotating long at recall
lr 1e-2; N13 controls at recall lr 1e-2, 4 500 steps; N17 controls at 1e-2, 13 500 steps; N12 controls at their selected
1e-1), runs them on a fixed recall probe batch (256 sequences, generator seed 123, as T1-B) and reports, separately for
neurons with r >= 0.8 and r < 0.8: the share that is saturated (firing rate > 0.9), silent (< 0.01) or active, and mean
|U| / threshold. Recall accuracy on the probe is a sanity check against the recorded run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = {"cand_long_fw003": "cand_long", "cand_longnorot_fw001": "cand_long_norot", "lif_fw2_d11_fw001": "lif_fw2_d11"}
SOURCES = ("results/n10", "results/n12", "results/n13", "results/n17")


def probe_model(row: dict, probe, n_out: int = 16, hidden: int | None = None, head=slice(0, 8)) -> dict:
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei54 import EXTRA_MODELS, MODELS
    base = BASE[row["variant"].split("|")[0]]
    cfg = {**(MODELS[base] if base in MODELS else EXTRA_MODELS[base]), "threshold": 0.5}
    if hidden:
        cfg["hidden"] = [hidden]
    x, y = probe
    model = SequenceClassifier(x.shape[-1], n_out, ModelConfig(**cfg))
    sv = torch.load(row["save_model"], weights_only=False)
    model.load_state_dict(sv["state_dict"])
    r = sv["final"][0]["r"]
    caught = {}
    model.neurons[0].register_forward_hook(lambda m, i, o: caught.update(pot=o[1].detach(), fire=o[2].detach()))
    model.eval()
    with torch.no_grad():
        acc = (model(x)[0][:, head].argmax(-1) == y).float().mean().item()
    rate = caught["fire"].float().mean(dim=(0, 1))
    mag = caught["pot"].abs().mean(dim=(0, 1)) / model.neurons[0].threshold
    out = {"recall_probe": acc, "recall_run": row["recall_final"]}
    for name, m in (("long", r >= 0.8), ("short", r < 0.8)):
        n = int(m.sum())
        out[name] = {"n": n, "saturated": int((rate[m] > 0.9).sum()), "silent": int((rate[m] < 0.01).sum()),
                     "mag": float(mag[m].mean()) if n else float("nan")}
    return out


def main(out: Path) -> None:
    from cvsnn.data.synthetic import assoc_recall_batch
    from experiments.ei54 import C_ECL, T
    from experiments.ep1b import TASK as REC_TASK
    c_rec = REC_TASK["n_keys"] + REC_TASK["n_vals"] + 1
    xr, y = assoc_recall_batch(256, pairs=8, generator=torch.Generator().manual_seed(123), **REC_TASK)
    x = torch.zeros(T, 256, C_ECL + c_rec)
    x[:, :, C_ECL:] = xr
    rows = []
    for d in SOURCES:
        for l in (ROOT / d / "runs.jsonl").read_text().splitlines():
            r = {"cell": "rotating long", **json.loads(l)}
            if r["setting"] == "synthetic (I54i)":
                rows.append(r)
    cells = {}
    for r in rows:
        cells.setdefault(r["cell"], []).append(probe_model(r, (x, y)))
    L = ["# N17 follow-up: activity of long-memory (r >= 0.8) and short-memory hidden neurons on a recall probe (synthetic sequence)", "",
         "Pooled over seeds. Saturated = firing rate > 0.9, silent = < 0.01, active = the rest; |U| / threshold averaged over time and batch.", "",
         "| model | runs | recall (probe / run) | long neurons: n | saturated | silent | **active** | mean \\|U\\|/thr | short neurons: n | saturated | silent | active | mean \\|U\\|/thr |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for c, rs in cells.items():
        row = [c, str(len(rs)), f"{sum(r['recall_probe'] for r in rs) / len(rs):.3f} / {sum(r['recall_run'] for r in rs) / len(rs):.3f}"]
        res[c] = {}
        for g in ("long", "short"):
            n = sum(r[g]["n"] for r in rs)
            sat, sil = sum(r[g]["saturated"] for r in rs), sum(r[g]["silent"] for r in rs)
            mags = [r[g]["mag"] for r in rs if r[g]["n"]]
            act = n - sat - sil
            res[c][g] = {"n": n, "saturated": sat, "silent": sil, "active": act, "mag": sum(mags) / len(mags) if mags else None}
            row += [str(n), f"{sat / n:.0%}" if n else "–", f"{sil / n:.0%}" if n else "–", f"**{act / n:.0%}**" if n else "–",
                    f"{sum(mags) / len(mags):.2f}" if mags else "–"]
        L.append("| " + " | ".join(row) + " |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "activity.md").write_text("\n".join(L) + "\n")
    (out / "activity.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


def task_probe(setting: str, seed: int, c_rec: int):
    """Test data of the last later-phase task (task 4), built as in experiments/ei54.py."""
    import numpy as np
    from experiments.ei54 import C_ECL, T
    if setting.startswith("permuted MNIST"):
        d = np.load(ROOT / "data/mnist.npz")
        X = torch.tensor(d["x_test"], dtype=torch.float32).reshape(-1, 784) / 255.0
        Y = torch.tensor(d["y_test"], dtype=torch.long)
        perm = torch.randperm(784, generator=torch.Generator().manual_seed(900 + 31 * seed + 4))
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(123))[:512]
        full = torch.zeros(T, len(idx), C_ECL + c_rec)
        full[T - 28:, :, :28] = X[idx][:, perm].reshape(-1, 28, 28).permute(1, 0, 2)
        return full, Y[idx]
    from cvsnn.data.shd import encode
    d = np.load(ROOT / "data/shd.npz")
    X, Y = torch.tensor(d["x_test"]), torch.tensor(d["y_test"])
    m = (Y // 4) == 4
    full = torch.zeros(T, int(m.sum()), C_ECL + c_rec)
    full[:, :, :C_ECL] = encode(X[m]).permute(1, 0, 2)
    return full, (Y[m] - 16).long()


def main_benchmarks(out: Path) -> None:
    from cvsnn.data.synthetic import assoc_recall_batch
    from experiments.ei54 import C_ECL, T
    from experiments.ep1b import TASK as REC_TASK
    c_rec = REC_TASK["n_keys"] + REC_TASK["n_vals"] + 1
    xr, yr = assoc_recall_batch(256, pairs=8, generator=torch.Generator().manual_seed(123), **REC_TASK)
    xrec = torch.zeros(T, 256, C_ECL + c_rec)
    xrec[:, :, C_ECL:] = xr
    rows = []
    for d in ("results/n10", "results/n12"):
        for l in (ROOT / d / "runs.jsonl").read_text().splitlines():
            r = {"cell": "rotating long", **json.loads(l)}
            if r["setting"] != "synthetic (I54i)":
                rows.append(r)
    L = ["# N18: activity of long-memory (r >= 0.8) and short-memory hidden neurons, models trained on the two benchmarks", "",
         "Selected settings (N10 / N12), 10 seeds, pooled. Probes: the recall probe (256 sequences, as T1-B) and the test data of the last",
         "later-phase task (MNIST: 512 test images with task 4's permutation; Split-SHD: every test sample of words 16-19).", "",
         "| setting | probe | model | accuracy (probe) | long neurons: n | saturated | silent | active | **mean \\|U\\|/thr** | short neurons: n | mean \\|U\\|/thr |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for setting in ("permuted MNIST + replay (I64b)", "Split-SHD + replay (I64a)"):
        n_out = 18 if setting.startswith("permuted") else 12
        for pname in ("recall", "task 4"):
            for cell in ("rotating long", "no-rotation long", "LIF long"):
                rs = [r for r in rows if r["setting"] == setting and r["cell"] == cell]
                outs = []
                for r in rs:
                    probe = (xrec, yr) if pname == "recall" else task_probe(setting, r["seed"], c_rec)
                    outs.append(probe_model(r, probe, n_out, 120, slice(0, 8) if pname == "recall" else slice(8, n_out)))
                lg = {k: sum(o["long"][k] for o in outs) for k in ("n", "saturated", "silent")}
                lm = [o["long"]["mag"] for o in outs if o["long"]["n"]]
                sm = [o["short"]["mag"] for o in outs if o["short"]["n"]]
                n = lg["n"]
                act = n - lg["saturated"] - lg["silent"]
                key = f"{setting}|{pname}|{cell}"
                res[key] = {**lg, "active": act, "long_mag": sum(lm) / len(lm) if lm else None, "short_mag": sum(sm) / len(sm),
                            "acc": sum(o["recall_probe"] for o in outs) / len(outs)}
                L.append(f"| {setting} | {pname} | {cell} | {res[key]['acc']:.3f} | {n} | {lg['saturated'] / n:.0%} | {lg['silent'] / n:.0%} | {act / n:.0%} | "
                         f"**{res[key]['long_mag']:.2f}** | {sum(o['short']['n'] for o in outs)} | {res[key]['short_mag']:.2f} |" if n else
                         f"| {setting} | {pname} | {cell} | {res[key]['acc']:.3f} | 0 | – | – | – | – | {sum(o['short']['n'] for o in outs)} | {res[key]['short_mag']:.2f} |")
    out.mkdir(parents=True, exist_ok=True)
    (out / "activity.md").write_text("\n".join(L) + "\n")
    (out / "activity.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    if "--benchmarks" in sys.argv:
        main_benchmarks(Path(sys.argv[1]) if len(sys.argv) > 2 else ROOT / "results/n18")
    else:
        main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/n17")
