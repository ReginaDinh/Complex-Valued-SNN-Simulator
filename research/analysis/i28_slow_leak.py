"""Why does the theta-tagged plastic synapse keep working with slow leaks (|beta| in [0.99, 0.9999]) in CV-SNN but not
in LIF? (I28b/c/d: 0.99 vs 0.32, 0.93 vs 0.26, 0.84 vs 0.23)

    cd research && .venv/bin/python -m analysis.i28_slow_leak [--models results/ei28b/models]

On the I28b test batch (8 items, one theta cycle), for the tagged networks trained with the fast and the slow ring:
  stuck_on / _off  fraction of hidden neurons that fire on > 95 % / < 5 % of all (time, sample) points (a CV-SNN
                   neuron that fires on every step still sends its phase; a LIF neuron that does sends nothing)
  value_item_sep   1 - mean |cos(v_n, v_m)| over slot pairs holding different items, divided by the same for pairs
                   holding the same item: > 0 means the stored values tell items apart (0 = not at all)
  content_match    |<k_n, q>| before the theta tag, for the queried slot n: the theta tag can only pick slot n
                   cleanly if the content parts of the write key and of the query agree
  retrieval        |<k_n, q>| after the tag for the queried slot, and the mean over the other 7 slots
Candidate explanations: (a) slow LIF neurons get stuck (no reset, |beta| near 1), so the values carry no item;
(b) the content parts of keys and queries drift apart (history), so the tag cannot separate slots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def inner(a, c):
    return (torch.conj(a) * c).sum(-1).abs() if torch.is_complex(a) else (a * c).sum(-1).abs()


def unit(z):
    return z / (torch.sqrt((z.abs() ** 2).sum(-1, keepdim=True)) + 1e-9)


def analyse(path: Path, seed: int) -> dict:
    from cvsnn.data.synthetic import serial_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei28b import network_input

    ck = torch.load(path, weights_only=False)
    task, cplx = ck["task"], ck["model"]["kind"] == "cvsnn"
    x, theta, y, pos = serial_recall_batch(1024, generator=torch.Generator().manual_seed(20_000 + seed), **task)
    xin = network_input(x, theta, cplx)
    model = SequenceClassifier(xin.shape[-1], task["n_symbols"], ModelConfig(**ck["model"]))
    model.load_state_dict(ck["state"])
    model.eval()
    cap = {}
    h = model.neurons[0].register_forward_hook(lambda m, i, o: cap.__setitem__("fire", o[2]))
    with torch.no_grad():
        lg, _ = model(xin, theta=theta)
    h.remove()
    L = model.fw.last
    n_items, slot, tail, ns = task["items"], task["slot"], task["tail"], task["n_symbols"]
    frac = cap["fire"].float().mean((0, 1))                                     # per hidden neuron
    stuck = ((frac > 0.95) | (frac < 0.05)).float().mean().item()
    stuck_on, stuck_off = (frac > 0.95).float().mean().item(), (frac < 0.05).float().mean().item()
    tw = [n * slot for n in range(n_items)]
    b = torch.arange(x.shape[1])
    items = torch.stack([x[t, :, :ns].argmax(-1) for t in tw], 1)              # (B, items)
    V = unit(torch.stack([L["v"][t] for t in tw], 1))                            # (B, items, H, d)
    C = inner(V.unsqueeze(2), V.unsqueeze(1)).mean(-1)                           # (B, items, items)
    off = ~torch.eye(n_items, dtype=torch.bool)
    same = (items.unsqueeze(2) == items.unsqueeze(1)) & off
    diff = (items.unsqueeze(2) != items.unsqueeze(1)) & off
    sep = 1 - C[diff].mean().item() / C[same].mean().item()
    Kr = torch.stack([L["k_raw"][t] for t in tw], 1)                             # (B, items, H, d)
    K = torch.stack([L["k"][t] for t in tw], 1)
    qr, q = L["q_raw"][-1], L["q"][-1]                                           # (B, H, d), the query (same on every read-out step)
    content = inner(Kr[b, pos], qr).mean().item()
    M = inner(K, q.unsqueeze(1))                                                 # (B, items, H)
    tgt = M[b, pos].mean().item()
    mask = torch.ones(x.shape[1], n_items, dtype=torch.bool)
    mask[b, pos] = False
    others = M[mask].mean().item()
    return {"seed": seed, "test_acc": (lg.argmax(-1) == y).float().mean().item(), "spike_rate_hidden": frac.mean().item(),
            "stuck": stuck, "stuck_on": stuck_on, "stuck_off": stuck_off, "value_item_sep": sep, "content_match": content, "retrieval_target": tgt, "retrieval_others": others}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "results" / "ei28b" / "models"))
    ap.add_argument("--out", default=str(ROOT / "results" / "i28_slow_leak"))
    a = ap.parse_args()
    from cvsnn.utils import mean_ci95

    rows = []
    for v in ("cv_fw2_tag", "cv_fw2_tag_bin", "cv_fw2_tag_reset", "lif_fw2_d11_tag", "lif_fw2_d11_tag_reset"):
        for ring in ("fast", "slow"):
            for path in sorted(Path(a.models).glob(f"{v}_{ring}_s*.pt")):
                rows.append({"variant": v, "ring": ring, **analyse(path, int(path.stem.split("_s")[-1]))})
    keys = ("test_acc", "spike_rate_hidden", "stuck_on", "stuck_off", "value_item_sep", "content_match", "retrieval_target", "retrieval_others")
    L = ["# Why the theta tag survives slow leaks in CV-SNN but not in LIF (I28b task, tagged networks, 5 seeds)", "",
         "Mean ± 95% CI over seeds. fast = |beta| ring [0.9, 0.999], slow = [0.99, 0.9999].", "",
         "| variant | ring | " + " | ".join(keys) + " |", "|---" * (len(keys) + 2) + "|"]
    for v in ("cv_fw2_tag", "cv_fw2_tag_bin", "cv_fw2_tag_reset", "lif_fw2_d11_tag", "lif_fw2_d11_tag_reset"):
        for ring in ("fast", "slow"):
            rs = [r for r in rows if r["variant"] == v and r["ring"] == ring]
            if rs:
                L.append(f"| {v} | {ring} | " + " | ".join("{:.3f} ± {:.3f}".format(*[mean_ci95([r[k] for r in rs])[i] for i in (0, 2)]) for k in keys) + " |")
    md = "\n".join(L) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
