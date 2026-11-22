"""I28c, step 2 of the follow-up: where does the cycle cue pick between the two items that share a theta phase?

    cd research && .venv/bin/python -m analysis.i28c_cue_mechanism [--models results/ei28c/models]

The read-out layer sees [hidden spikes h, plastic-synapse output f]. With the cue swapped at the query the network
answers the other item (analysis.i28c_key_overlap). Here the read-out input is spliced between a run with the normal
cue (N) and a run with the swapped cue (S):
  (h_N, f_S): if the answer follows f, the choice is made when the plastic synapse is read (by the query);
  (h_S, f_N): if the answer follows h, the read-out layer makes it from the cue carried by the hidden spikes.
Counted on wrapped samples whose two same-phase items differ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def analyse(path: Path, seed: int) -> dict:
    from cvsnn.data.synthetic import serial_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei28b import network_input

    ck = torch.load(path, weights_only=False)
    task, cplx = ck["task"], ck["model"]["kind"] == "cvsnn"
    x, theta, y, pos = serial_recall_batch(1024, generator=torch.Generator().manual_seed(20_000 + seed), **task)
    n_items, cyc, slot, tail, ns = task["items"], task["cycle_slots"], task["slot"], task["tail"], task["n_symbols"]
    xs = x.clone()
    xs[-tail:, :, ns + 1:] = xs[-tail:, :, ns + 1:].flip(-1)                    # swapped cycle cue
    model = SequenceClassifier(network_input(x, theta, cplx).shape[-1], ns, ModelConfig(**ck["model"]))
    model.load_state_dict(ck["state"])
    model.eval()
    n_h = ck["model"]["hidden"][-1]
    seen, sub = {}, {}
    hook = model.linears[-1].register_forward_pre_hook(lambda m, a: (sub["x"],) if "x" in sub else seen.__setitem__("x", a[0]))
    with torch.no_grad():
        model(network_input(x, theta, cplx), theta=theta)
        X_N = seen.pop("x")
        model(network_input(xs, theta, cplx), theta=theta)
        X_S = seen.pop("x")
        answers = {}
        for name, X in (("normal", X_N), ("swapped", X_S), ("h_N+f_S", torch.cat([X_N[..., :n_h], X_S[..., n_h:]], -1)),
                        ("h_S+f_N", torch.cat([X_S[..., :n_h], X_N[..., n_h:]], -1))):
            sub["x"] = X
            answers[name] = model(network_input(x, theta, cplx), theta=theta)[0].argmax(-1)
            sub.clear()
    hook.remove()
    b = torch.arange(x.shape[1])
    items = torch.stack([x[n * slot, :, :ns].argmax(-1) for n in range(n_items)], 1)
    partner = torch.where(pos < cyc, pos + cyc, pos - cyc).clamp(max=n_items - 1)
    diff = ((pos < n_items - cyc) | (pos >= cyc)) & (items[b, pos] != items[b, partner])
    out = {"seed": seed}
    for name, a in answers.items():
        out[f"{name}_queried"] = (a == items[b, pos])[diff].float().mean().item()
        out[f"{name}_partner"] = (a == items[b, partner])[diff].float().mean().item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "results" / "ei28c" / "models"))
    ap.add_argument("--out", default=str(ROOT / "results" / "i28c_cue"))
    a = ap.parse_args()
    rows = []
    for v in ("cv_fw2_tag", "lif_fw2_d11_tag"):
        for path in sorted(Path(a.models).glob(f"{v}_fast_s*.pt")):
            rows.append({"variant": v, **analyse(path, int(path.stem.split("_s")[-1]))})
    L = ["# I28c follow-up: where the cycle cue picks the item (read-out input spliced between normal-cue and swapped-cue runs)", "",
         "Fraction of wrapped samples (two same-phase items differ) answered with the queried item / with the partner item; mean over 5 seeds.", "",
         "| variant | normal | swapped cue | h normal + f swapped | h swapped + f normal |", "|---|---|---|---|---|"]
    for v in ("cv_fw2_tag", "lif_fw2_d11_tag"):
        rs = [r for r in rows if r["variant"] == v]
        m = lambda k: sum(r[k] for r in rs) / len(rs)
        L.append(f"| {v} | " + " | ".join(f"{m(n + '_queried'):.3f} / {m(n + '_partner'):.3f}" for n in ("normal", "swapped", "h_N+f_S", "h_S+f_N")) + " |")
    md = "\n".join(L) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
