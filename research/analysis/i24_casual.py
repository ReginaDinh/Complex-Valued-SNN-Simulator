"""I24, causal check: in the trained rotating network (cv_fw2, P1b P = 8, the I24 models), stop the rotation of the
hidden neurons at test time, or shuffle their frequencies, without retraining. Does the history then crowd the keys,
as in the networks that never rotated?

    cd research && .venv/bin/python -m analysis.i24_causal

Measures from analysis.i24_key_history on the I21 test batch: accuracy, overlap of actual keys, overlap of history-free
keys, history-added overlap (their difference), history alignment (+1 in phase, < 0 cancelling) and history size.
Interventions on the hidden layer only (the keys are made from it): omega = 0 for every hidden neuron; omega shuffled
across the hidden neurons (same set of frequencies, wrong neurons).
Prediction written before running: with omega = 0 the history-added overlap rises towards the non-rotating networks'
(+0.20 to +0.23) and the alignment turns positive; accuracy drops. Shuffling: in between.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from analysis.i24_key_history import analyse_model, load
    from cvsnn.utils import mean_ci95

    rows = []
    for s in range(10):
        path = ROOT / "results" / "ep1b" / "i24_models" / f"P8_cv_fw2_s{s}.pt"
        for name in ("trained", "omega0", "shuffled"):
            m = load(path)
            with torch.no_grad():
                ph = m.neurons[0].leak_phase
                if name == "omega0":
                    ph.zero_()
                elif name == "shuffled":
                    ph.copy_(ph[torch.randperm(len(ph), generator=torch.Generator().manual_seed(100 + s))])
            r = analyse_model(m, s)
            r["added"] = r["overlap_actual"] - r["overlap_free"]
            rows.append({"intervention": name, **r})
    keys = ("test_acc", "overlap_actual", "overlap_free", "added", "history_alignment", "history_size", "sim_actual_free")
    L = ["# I24 causal: stop or shuffle the hidden rotation of the trained cv_fw2 at test time (P1b, P = 8, 10 seeds)", "",
         "| intervention | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name in ("trained", "omega0", "shuffled"):
        rs = [r for r in rows if r["intervention"] == name]
        L.append(f"| {name} | " + " | ".join("{:.3f} ± {:.3f}".format(*[mean_ci95([r[k] for r in rs])[i] for i in (0, 2)]) for k in keys) + " |")
    md = "\n".join(L) + "\n"
    out = ROOT / "results" / "i24_causal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
