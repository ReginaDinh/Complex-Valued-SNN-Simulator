"""T2 (theory check, Proposition 2): the polar form keeps every eigenvalue inside the unit disk during learning; the free
(adLIF) form can leave it.

    cd research && .venv/bin/python -m analysis.t2_stability [out_dir]

Both networks start from identical dynamics (I61: alpha = beta_w = r cos(omega), a = r^2 sin^2(omega), |beta| ring
[0.9, 0.999], omega 0.15). The adLIF hidden layer's linear part is A = [[alpha, -1], [a, beta_w]], complex eigenvalues
with |lambda|^2 = det A = alpha beta_w + a; stable iff det A < 1 and |tr A| < 1 + det A (alpha, beta_w are sigmoids, a is
free). The cLIF has |lambda| = sigmoid(raw) < 1 by construction. Recall phase (8 pairs), lr 1e-2 and 1e-1 (the I61 /
I54i range), plastic synapse x0.03, 1 500 steps, seed 0; every 50 steps: largest |lambda| / det A, share of unstable
neurons, recall on a fixed probe batch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cvsnn.data.synthetic import assoc_recall_batch
from cvsnn.models import ModelConfig, SequenceClassifier
from cvsnn.utils import seed_all
from experiments.ei54 import C_ECL, EXTRA_MODELS, T
from experiments.ep1b import TASK as REC_TASK


def run(base: str, lr: float, steps: int = 1500, seed: int = 0) -> dict:
    torch.set_num_threads(1)
    seed_all(seed)
    c_rec = REC_TASK["n_keys"] + REC_TASK["n_vals"] + 1
    model = SequenceClassifier(C_ECL + c_rec, 16, ModelConfig(**{**EXTRA_MODELS[base], "threshold": 0.5}))
    fwp = {id(p) for p in model.fw.parameters()}
    opt = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in fwp], "lr": lr},
                             {"params": [p for p in model.parameters() if id(p) in fwp], "lr": lr * 0.03}], lr=lr)
    n = model.neurons[0]

    def eig():
        with torch.no_grad():
            if base == "adlif_long":
                al, bw, a = torch.sigmoid(n.leak_raw), torch.sigmoid(n.wleak_raw), n.a
                det, tr = al * bw + a, al + bw
                unstable = (det >= 1) | (tr.abs() >= 1 + det)
                return det.clamp(min=0).sqrt().max().item(), unstable.float().mean().item()
            r = torch.sigmoid(n.leak_mag_raw)
            return r.max().item(), (r >= 1).float().mean().item()

    def batch(k, g):
        x, y = assoc_recall_batch(k, pairs=8, generator=g, **REC_TASK)
        full = torch.zeros(T, k, C_ECL + c_rec)
        full[:, :, C_ECL:] = x
        return full, y

    probe = batch(256, torch.Generator().manual_seed(123))

    def recall():
        model.eval()
        with torch.no_grad():
            a = (model(probe[0])[0][:, :8].argmax(-1) == probe[1]).float().mean().item()
        model.train()
        return a

    log = []
    g = torch.Generator().manual_seed(10_000 + seed)
    for s in range(steps + 1):
        if s % 50 == 0:
            m, u = eig()
            log.append({"step": s, "max_abs_lambda": m, "unstable_share": u, "recall": recall()})
        if s == steps:
            break
        x, y = batch(64, g)
        loss = F.cross_entropy(model(x)[0][:, :8], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return {"base": base, "lr": lr, "log": log}


def main(out: Path, seeds: int = 10) -> None:
    from cvsnn.utils import mean_ci95
    out.mkdir(parents=True, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    cells = [("cand_long", 1e-2), ("cand_long", 1e-1), ("adlif_long", 1e-2), ("adlif_long", 1e-1)]
    args = [(b, lr, 1500, s) for b, lr in cells for s in range(seeds)]
    with ProcessPoolExecutor(10) as pool:
        res = list(pool.map(run, *zip(*args)))
    for r, a in zip(res, args):
        r["seed"] = a[-1]
    ci = lambda xs: "{:.3f} ± {:.3f}".format(mean_ci95(xs)[0], mean_ci95(xs)[2])
    lines = ["# T2: eigenvalues during learning, polar (cLIF) vs free (adLIF) form (Proposition 2)", "",
             f"{seeds} seeds per row, mean ± CI95.", "",
             "| model | lr | peak max abs(lambda) | seeds with an unstable neuron | peak share of unstable neurons | first unstable step (median) | recall at step 1500 |",
             "|---|---|---|---|---|---|---|"]
    for b, lr in cells:
        rs = [r for r in res if r["base"] == b and r["lr"] == lr]
        peaks = [max(e["max_abs_lambda"] for e in r["log"]) for r in rs]
        shares = [max(e["unstable_share"] for e in r["log"]) for r in rs]
        firsts = sorted(next((e["step"] for e in r["log"] if e["unstable_share"] > 0), 10**9) for r in rs)
        med = firsts[len(firsts) // 2]
        rec = [r["log"][-1]["recall"] for r in rs]
        lines.append(f"| {'cLIF (polar)' if b == 'cand_long' else 'adLIF (free)'} | {lr:g} | {ci(peaks)} | {sum(s > 0 for s in shares)}/{len(rs)} | "
                     f"{ci(shares)} | {med if med < 10**9 else 'never'} | {ci(rec)} |")
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(text)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/t2"), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
