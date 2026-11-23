"""I28c follow-up: in the theta-tagged networks, are the keys of slots n and n + 8 (same theta phase, different theta
cycle) kept apart, and does the query pick the right one?

    cd research && .venv/bin/python -m analysis.i28c_key_overlap [--models results/ei28c/models]

Slots n and n + 8 get the same theta tag, so their tagged keys overlap exactly as much as their untagged content parts
do. If the network told the two cycles apart through the content part (traces of earlier items while writing, the
cycle cue while asking), these keys should overlap no more than keys of unrelated slots. Measures on the test batch
of I28c (same seeds), per sample and head, keys taken at the write times t_n = 4 n:
  overlap_alias      mean |<k_n, k_{n+8}>|, n = 0..3 (same theta phase, different cycle)
  overlap_same_cycle mean |<k_n, k_m>|, n != m both in cycle 1 (different phases)
  overlap_cross      mean |<k_n, k_m>|, n in cycle 1, m in cycle 2, different phases
  query_target / query_alias  mean |<k, q>| at the read-out for the queried slot and for the slot sharing its phase
  write_gate_alias   mean write gate at the aliased slots (0-3, 8-11), to check both were written
  cue_swap / cue_none  at the query, swap or remove the cycle cue (theta phase unchanged): how often the network then
                   answers the queried slot's item and how often the item of the slot sharing its phase
Chance overlap of random unit vectors: complex d = 8 0.318, real d = 11 0.246.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def inner(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (torch.conj(a) * c).sum(-1).abs() if torch.is_complex(a) else (a * c).sum(-1).abs()


def analyse(path: Path, seed: int) -> dict:
    from cvsnn.data.synthetic import serial_recall_batch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei28b import network_input

    ck = torch.load(path, weights_only=False)
    task, cplx = ck["task"], ck["model"]["kind"] == "cvsnn"
    x, theta, y, pos = serial_recall_batch(1024, generator=torch.Generator().manual_seed(20_000 + seed), **task)  # the I28c test batch
    xin = network_input(x, theta, cplx)
    model = SequenceClassifier(xin.shape[-1], task["n_symbols"], ModelConfig(**ck["model"]))
    model.load_state_dict(ck["state"])
    model.eval()
    with torch.no_grad():
        lg, _ = model(xin, theta=theta)
    k, q, beta = model.fw.last["k"], model.fw.last["q"], model.fw.last["beta"]  # (T, B, H, d), (T, B, H)
    n_items, cyc, slot, tail = task["items"], task["cycle_slots"], task["slot"], task["tail"]
    K = torch.stack([k[n * slot] for n in range(n_items)], 1)                  # (B, items, H, d)
    G = inner(K.unsqueeze(2), K.unsqueeze(1))                                  # (B, items, items, H)
    alias = [(n, n + cyc) for n in range(n_items - cyc)]
    same = [(n, m) for n in range(cyc) for m in range(cyc) if n != m]
    cross = [(n, m) for n in range(cyc) for m in range(cyc, n_items) if m - cyc != n]
    mean_pairs = lambda pairs: torch.stack([G[:, i, j] for i, j in pairs]).mean().item()
    qr = q[-tail:]                                                             # (tail, B, H, d), same input on every read-out step
    b = torch.arange(x.shape[1])
    wrapped = (pos < n_items - cyc) | (pos >= cyc)
    partner = torch.where(pos < cyc, pos + cyc, pos - cyc)
    tgt = inner(K[b, pos].unsqueeze(0), qr)[:, wrapped].mean().item()
    ali = inner(K[b, partner.clamp(max=n_items - 1)].unsqueeze(0), qr)[:, wrapped].mean().item()
    gate_idx = [n * slot for n in range(n_items) if n < n_items - cyc or n >= cyc]
    # cue probe: at the query, keep the theta phase but swap (or remove) the cycle cue; on samples whose two items of
    # the same phase differ, which of them does the network answer?
    items = torch.stack([x[n * slot, :, :task["n_symbols"]].argmax(-1) for n in range(n_items)], 1)  # (B, items)
    pi = partner.clamp(max=n_items - 1)
    diff = wrapped & (items[b, pos] != items[b, pi])
    probe = {}
    cue = slice(task["n_symbols"] + 1, None)
    for name in ("swap", "none"):
        xx = x.clone()
        xx[-tail:, :, cue] = xx[-tail:, :, cue].flip(-1) if name == "swap" else 0.0
        with torch.no_grad():
            pr = model(network_input(xx, theta, cplx), theta=theta)[0].argmax(-1)
        probe[f"cue_{name}_answers_queried"] = (pr == items[b, pos])[diff].float().mean().item()
        probe[f"cue_{name}_answers_partner"] = (pr == items[b, pi])[diff].float().mean().item()
    return {"seed": seed, "test_acc": (lg.argmax(-1) == y).float().mean().item(),
            "overlap_alias": mean_pairs(alias), "overlap_same_cycle": mean_pairs(same), "overlap_cross": mean_pairs(cross),
            "query_target": tgt, "query_alias": ali, "write_gate_alias": beta[gate_idx].mean().item(),
            "write_gate_between": beta[[t for t in range(n_items * slot) if t % slot]].mean().item(), **probe}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "results" / "ei28c" / "models"))
    ap.add_argument("--out", default=str(ROOT / "results" / "i28c_keys"))
    a = ap.parse_args()
    from cvsnn.utils import mean_ci95

    rows = []
    for v in ("cv_fw2_tag", "lif_fw2_d11_tag"):
        for path in sorted(Path(a.models).glob(f"{v}_fast_s*.pt")):
            r = {"variant": v, **analyse(path, int(path.stem.split("_s")[-1]))}
            rows.append(r)
            print({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
    keys = ("test_acc", "overlap_alias", "overlap_same_cycle", "overlap_cross", "query_target", "query_alias", "write_gate_alias", "write_gate_between",
            "cue_swap_answers_queried", "cue_swap_answers_partner", "cue_none_answers_queried", "cue_none_answers_partner")
    L = ["# I28c follow-up: keys of slots n and n + 8 in the theta-tagged networks (test batch, 5 seeds)", "",
         "Mean ± 95% CI over seeds. Chance overlap: complex d = 8 0.318, real d = 11 0.246.", "",
         "| variant | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for v in ("cv_fw2_tag", "lif_fw2_d11_tag"):
        rs = [r for r in rows if r["variant"] == v]
        if rs:
            L.append(f"| {v} | " + " | ".join("{:.3f} ± {:.3f}".format(*[mean_ci95([r[k] for r in rs])[i] for i in (0, 2)]) for k in keys) + " |")
    md = "\n".join(L) + "\n"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
