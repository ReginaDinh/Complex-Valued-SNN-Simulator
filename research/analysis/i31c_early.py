"""I31c: what does rotation do in the early phase of learning? (I31/I31b: rotation must stay on until the solution is
found; switched off afterwards it costs nothing; never on, training lands in poor solutions)

    cd research && .venv/bin/python -m analysis.i31c_early

P1b, P = 8, ring [0.9, 0.999]; cv_fw2 (rotating) and cv_fw2_w0 (omega = 0), 10 seeds, the first 1500 training steps
of the I21 recipe (same seeds and data). Every 50 steps, on a fixed probe batch of 256 sequences:
  probe_acc       accuracy
  key_overlap     mean |<k_i, k_j>| of the keys at the true write times (experiments.ep1b.key_stats)
  query_contrast  query match to the queried key minus match to the other keys
  grad_k / grad_in gradient norms of the key projection and of the input layer on the training batch
Prediction written before running: already at initialisation the rotating network's keys overlap less (traces of
earlier pairs dephase), its query contrast and accuracy rise earlier, and its key-projection gradients are larger.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("cv_fw2", "cv_fw2_w0")
STEPS, EVERY = 1500, 50


def run(job: dict) -> dict:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    from cvsnn.data.synthetic import assoc_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from cvsnn.utils import seed_all
    from experiments.ep1b import TASK, key_stats

    seed_all(job["seed"])

    def sample(n, g):
        x, y = assoc_recall_batch(n, pairs=8, generator=g, **TASK)
        return torch.complex(x, torch.zeros_like(x)), y

    model = SequenceClassifier(17, 8, ModelConfig(**job["model"]))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g_tr = torch.Generator().manual_seed(10_000 + job["seed"])
    px, py = sample(256, torch.Generator().manual_seed(77_000 + job["seed"]))
    log = []
    for step in range(STEPS + 1):
        x, y = sample(64, g_tr)
        loss = F.cross_entropy(model(x)[0], y)
        opt.zero_grad()
        loss.backward()
        if step % EVERY == 0:
            gk = torch.sqrt(model.fw.k.weight_re.grad.pow(2).sum() + model.fw.k.weight_im.grad.pow(2).sum()).item()
            l0 = model.linears[0]
            gi = torch.sqrt(l0.weight_re.grad.pow(2).sum() + l0.weight_im.grad.pow(2).sum()).item()
            model.eval()
            with torch.no_grad():
                acc = (model(px)[0].argmax(-1) == py).float().mean().item()
                ks = key_stats(px.real, model.fw.last, 8)
            model.train()
            log.append({"step": step, "loss": loss.item(), "probe_acc": acc, "key_overlap": ks["key_overlap"],
                        "query_contrast": ks["query_match"] - ks["query_cross"], "grad_k": gk, "grad_in": gi})
        if step < STEPS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return {"variant": job["variant"], "seed": job["seed"], "log": log}


def main() -> None:
    from experiments.ep1b import jobs
    base = [j for j in jobs(10, STEPS) if j["pairs"] == 8 and j["variant"] in VARIANTS and j["hp"]["leak_mag_range"] == [0.9, 0.999]]
    with ProcessPoolExecutor(10) as pool:
        rows = list(pool.map(run, base))
    out = ROOT / "results" / "i31c"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows))
    keys = ("probe_acc", "key_overlap", "query_contrast", "grad_k", "grad_in", "loss")
    L = ["# I31c: early learning with and without rotation (P1b, P = 8, mean over 10 seeds)", "",
         "| step | " + " | ".join(f"{k} rot / w0" for k in keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for i, step in enumerate(range(0, STEPS + 1, EVERY)):
        if step in (0, 50, 100, 200, 300, 500, 750, 1000, 1500):
            cells = []
            for k in keys:
                m = {v: sum(r["log"][i][k] for r in rows if r["variant"] == v) / 10 for v in VARIANTS}
                cells.append(f"{m['cv_fw2']:.3f} / {m['cv_fw2_w0']:.3f}")
            L.append(f"| {step} | " + " | ".join(cells) + " |")
    md = "\n".join(L) + "\n"
    (out / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
