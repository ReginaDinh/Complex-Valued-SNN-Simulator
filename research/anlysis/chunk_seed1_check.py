"""Is the 0.75 seed of the chunked plastic synapse a numerical problem or ordinary training chaos?

    cd research && .venv/bin/python -m analysis.chunk_seed1_check

(1) Retrain cv_fw2_chunk, seed 1 (I21 set-up, P = 8), save it, and evaluate the same trained weights with the
    step-by-step loop: if the loop gives the same accuracy, the chunked computation itself is not at fault.
(2) Nudge the initial weights by a relative 1e-6 (the size of float32 rounding) five times, for the loop and for the
    chunked scan: if the loop's own outcomes for seed 1 spread as far as 0.75, the low value is training chaos.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import torch
    from cvsnn.data.synthetic import assoc_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ep1b import TASK, _run, jobs

    base = [j for j in jobs(10, 3000) if j["pairs"] == 8 and j["seed"] == 1 and j["hp"]["leak_mag_range"] == [0.9, 0.999]]
    loop = next(j for j in base if j["variant"] == "cv_fw2")
    chunk = next(j for j in base if j["variant"] == "cv_fw2_chunk")
    save = ROOT / "results" / "ep1b" / "chunk_seed1"
    todo = [{**chunk, "save_dir": str(save)}]
    todo += [{**loop, "key": loop["key"] + f"|jitter={i}", "init_jitter": i} for i in range(1, 6)]
    todo += [{**chunk, "key": chunk["key"] + f"|jitter={i}", "init_jitter": i} for i in range(1, 6)]
    with ProcessPoolExecutor(10) as pool:
        res = list(pool.map(_run, todo))
    rows = [{**r, "jitter": j.get("init_jitter")} for j, r in zip(todo, res)]
    # (1) the saved chunk-trained model, evaluated both ways on the I21 test batch
    ck = torch.load(save / "P8_cv_fw2_chunk_s1.pt", weights_only=False)
    x, y = assoc_recall_batch(512, pairs=8, generator=torch.Generator().manual_seed(20_000 + 1), **TASK)
    x = torch.complex(x, torch.zeros_like(x))
    accs = {}
    for name, ch in (("chunk", 16), ("loop", None)):
        cfg = {**ck["model"], "fast_weights": {**ck["model"]["fast_weights"], "chunk": ch}}
        m = SequenceClassifier(17, 8, ModelConfig(**cfg))
        m.load_state_dict(ck["state"])
        m.eval()
        with torch.no_grad():
            accs[name] = (m(x)[0].argmax(-1) == y).float().mean().item()
    out = ROOT / "results" / "ep1b" / "chunk_seed1_check.json"
    out.write_text(json.dumps({"runs": rows, "same_weights_eval": accs}, indent=2))
    print("same chunk-trained weights evaluated with:", accs)
    for r in rows:
        print(r["variant"], "jitter", r["jitter"], "test %.3f" % r["test_acc"])


if __name__ == "__main__":
    main()
