"""T1 (theory check, Proposition 1): rotation bounds the history pile-up but not the memory.

    cd research && .venv/bin/python -m analysis.t1_pileup [out_dir]

A. One neuron U_t = lambda U_{t-1} + I_t, lambda = r e^{i omega}, real input I_t = mu + xi_t (xi i.i.d., var sigma^2):
   stationary |E U| = mu / |1 - lambda| (no rotation: mu / (1 - r); rotation: <= mu / sin(omega) for every r) and
   Var(Re U) = (sigma^2 / 2) [1 / (1 - r^2) + Re 1 / (1 - lambda^2)] (memory set by r). Simulation vs formula.
B. The networks of the paper (no reset, |beta| ring [0.9, 0.999]): rotating (cand_long) vs the same without rotation
   (cand_long_norot) vs LIF long, at initialisation and after the recall phase (1 500 steps, each at its chosen lr and
   plastic-synapse multiplier): mean |U| / threshold, share of saturated (rate > 0.9) and silent (< 0.01) hidden neurons,
   and the overlap of the plastic-synapse keys across time (mean |cos| between keys at least 5 steps apart).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cvsnn.data.synthetic import assoc_recall_batch
from cvsnn.models import ModelConfig, SequenceClassifier
from cvsnn.utils import seed_all
from experiments.ei54 import C_ECL, EXTRA_MODELS, MODELS, T
from experiments.ep1b import TASK as REC_TASK


def part_a(lines: list[str]) -> None:
    mu, sigma, steps, burn = 0.3, 1.0, 40_000, 8_000
    g = torch.Generator().manual_seed(0)
    lines += ["## A. One neuron: simulation vs formula", "",
              "| r | omega | E abs(U) sim | formula mu/abs(1-lambda) | bound mu/sin(omega) | std Re U sim | formula |", "|---|---|---|---|---|---|---|"]
    for om in (0.0, 0.15):
        for r in (0.9, 0.99, 0.999):
            lam = complex(r * math.cos(om), r * math.sin(om))
            x = mu + sigma * torch.randn(steps, generator=g, dtype=torch.float64)
            u, us = 0j, []
            for t in range(steps):
                u = lam * u + float(x[t])
                if t >= burn:
                    us.append(u)
            us = torch.tensor(us)
            mean_sim = abs(complex(us.mean()))
            mean_th = mu / abs(1 - lam)
            var_th = sigma ** 2 / 2 * (1 / (1 - r * r) + (1 / (1 - lam * lam)).real)
            bound = f"{mu / math.sin(om):.2f}" if om else "none"
            lines.append(f"| {r} | {om} | {mean_sim:.2f} | {mean_th:.2f} | {bound} | {us.real.std():.2f} | {math.sqrt(var_th):.2f} |")


def train_and_probe(name: str, base: str, fw_mult: float, lr: float, steps: int, seed: int = 0) -> dict:
    torch.set_num_threads(1)
    seed_all(seed)
    c_rec = REC_TASK["n_keys"] + REC_TASK["n_vals"] + 1
    cfg = {**(MODELS[base] if base in MODELS else EXTRA_MODELS[base]), "threshold": 0.5}
    model = SequenceClassifier(C_ECL + c_rec, 16, ModelConfig(**cfg))
    fwp = {id(p) for p in model.fw.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in fwp], "lr": lr},
                             {"params": [p for p in model.parameters() if id(p) in fwp], "lr": lr * fw_mult}], lr=lr)

    def batch(n, g):
        x, y = assoc_recall_batch(n, pairs=8, generator=g, **REC_TASK)
        full = torch.zeros(T, n, C_ECL + c_rec)
        full[:, :, C_ECL:] = x
        return full, y

    probe = batch(256, torch.Generator().manual_seed(123))
    caught = {}
    model.neurons[0].register_forward_hook(lambda m, i, o: caught.__setitem__("pot", o[1].detach()) or caught.__setitem__("fire", o[2].detach()))
    model.fw.k.register_forward_hook(lambda m, i, o: caught.__setitem__("k", o.detach()))

    def measure():
        model.eval()
        with torch.no_grad():
            x, y = probe
            acc = (model(x)[0][:, :8].argmax(-1) == y).float().mean().item()
        model.train()
        pot = caught["pot"]
        mag = (pot.abs() if torch.is_complex(pot) else pot.abs()) / model.neurons[0].threshold
        rate = caught["fire"].float().mean(dim=(0, 1))
        k = F.normalize(caught["k"], dim=-1)                    # (T, B, heads*d)
        Tn = k.shape[0]
        cos = torch.einsum("tbd,sbd->bts", k, k).abs()
        far = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), diagonal=5)
        return {"recall": acc, "mean_abs_U_over_thr": mag.mean().item(), "saturated": (rate > 0.9).float().mean().item(),
                "silent": (rate < 0.01).float().mean().item(), "key_overlap": cos[:, far].mean().item()}

    out = {"name": name, "init": measure()}
    g = torch.Generator().manual_seed(10_000 + seed)
    for _ in range(steps):
        x, y = batch(64, g)
        loss = F.cross_entropy(model(x)[0][:, :8], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    out["trained"] = measure()
    return out


def main(out: Path, seeds: int = 10) -> None:
    from cvsnn.utils import mean_ci95
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# T1: history pile-up with and without rotation (Proposition 1)", ""]
    part_a(lines)
    cells = [("rotating, long memory", "cand_long", 0.03, 1e-2), ("no rotation, long memory", "cand_long_norot", 0.01, 1e-1),
             ("LIF, long memory", "lif_fw2_d11", 0.01, 1e-1)]
    from concurrent.futures import ProcessPoolExecutor
    args = [(n, b, m, lr, 1500, s) for n, b, m, lr in cells for s in range(seeds)]
    with ProcessPoolExecutor(10) as pool:
        res = list(pool.map(train_and_probe, *zip(*args)))
    for r, a in zip(res, args):
        r["seed"] = a[-1]
    keys = ("recall", "mean_abs_U_over_thr", "saturated", "silent", "key_overlap")
    ci = lambda xs: "{:.3f} ± {:.3f}".format(mean_ci95(xs)[0], mean_ci95(xs)[2])
    lines += ["", f"## B. Networks (hidden layer, recall probe batch; {seeds} seeds, mean ± CI95; recall phase 1 500 steps at each model's chosen lr)", "",
              "| model | when | recall | mean abs(U) / threshold | saturated neurons (rate > 0.9) | silent (< 0.01) | key overlap across time |",
              "|---|---|---|---|---|---|---|"]
    for name, *_ in cells:
        rs = [r for r in res if r["name"] == name]
        for w in ("init", "trained"):
            lines.append(f"| {name} | {w} | " + " | ".join(ci([r[w][k] for r in rs]) for k in keys) + " |")
    rot = {r["seed"]: r for r in res if r["name"] == cells[0][0]}
    lines += ["", "Per seed: key overlap of the rotating network lower than the other model (init / trained):"]
    for name, *_ in cells[1:]:
        oth = {r["seed"]: r for r in res if r["name"] == name}
        a = sum(rot[s]["init"]["key_overlap"] < oth[s]["init"]["key_overlap"] for s in rot)
        b = sum(rot[s]["trained"]["key_overlap"] < oth[s]["trained"]["key_overlap"] for s in rot)
        lines.append(f"- vs {name}: {a}/{len(rot)} / {b}/{len(rot)}")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/t1"), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
