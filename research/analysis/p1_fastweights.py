"""P1: knowledge stored in synapses (Hebbian fast weights) vs in neuron state, complex vs real. No training.

    cd research && .venv/bin/python -m analysis.p1_fastweights            # full (about 1-2 min)
    .venv/bin/python -m analysis.p1_fastweights --quick

M key-value facts, keys and values random codes, values from a codebook of F = 1000 symbols.
  state   : neuron-state memory (as N1-S1): one vector r = sum_i key_i (x) value_i (elementwise binding)
  synapse : Hebbian fast-weight matrix A = sum_i value_i key_i^H (outer products; complex keys conjugated),
            read out as A key_q: the "plastic synapse" memory, mathematically linear attention / fast weights
Codes: complex unit phasors vs real bipolar +-1. Resources compared:
  equal vector reals : N phasors (2N reals per vector, matrix 2N^2 reals) vs 2N bipolar (matrix 4N^2 reals)
  equal matrix reals : N phasors (2N^2 reals) vs round(sqrt(2) N) bipolar (about 2N^2 reals)
Metrics: retrieval accuracy (best match in the codebook) and "don't know" AUROC (stored vs never-stored keys).
Prediction written before running: the synaptic matrix holds about N times more facts than the state vector;
complex and real are about equal at equal reals (as in N1-S1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "p1_fastweights"
F = 1000
UNSTORED = 200


def codes(kind: str, n: int, count: int, g: torch.Generator) -> torch.Tensor:
    if kind == "complex":
        return torch.polar(torch.ones(count, n, dtype=torch.float64), torch.rand(count, n, generator=g, dtype=torch.float64) * 2 * math.pi)
    return torch.randint(0, 2, (count, n), generator=g).double() * 2 - 1


def sim(v: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
    return (v @ torch.conj(book).T).real if torch.is_complex(book) else v @ book.T


def auroc(pos, neg):
    p, q = pos.view(-1, 1), neg.view(1, -1)
    return ((p > q).double().mean() + 0.5 * (p == q).double().mean()).item()


def trial(memory: str, kind: str, n: int, m: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    book = codes(kind, n, F, g)
    keys = codes(kind, n, m + UNSTORED, g)
    stored, unstored = keys[:m], keys[m:]
    fill = torch.randint(0, F, (m,), generator=g)
    vals = book[fill]
    if memory == "state":
        r = (stored * vals).sum(0, keepdim=True)
        read = lambda k: r * (torch.conj(k) if torch.is_complex(k) else k)
    else:
        kt = torch.conj(stored) if kind == "complex" else stored
        A = vals.T @ kt                                     # (n, n): sum_i value_i key_i^H
        read = lambda k: (A @ k.T).T
    s_in, s_out = sim(read(stored), book), sim(read(unstored), book)
    return {"acc": (s_in.argmax(-1) == fill).double().mean().item(),
            "auroc": auroc(s_in.max(-1).values, s_out.max(-1).values)}


def run(memory, kind, n, m, trials):
    rs = [trial(memory, kind, n, m, 1000 * t + 7 * m + n) for t in range(trials)]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 2 if a.quick else 8
    N = 128 if a.quick else 256
    Ms = [16, 64, 256, 1024] if a.quick else [16, 64, 256, 1024, 2048, 4096, 8192]
    configs = [  # label, memory, code, neurons, vector reals, memory reals
        ("state, complex (N)", "state", "complex", N, 2 * N, 2 * N),
        ("state, bipolar (2N)", "state", "real", 2 * N, 2 * N, 2 * N),
        ("synapse, complex (N)", "synapse", "complex", N, 2 * N, 2 * N * N),
        ("synapse, bipolar (2N) = equal vector reals", "synapse", "real", 2 * N, 2 * N, 4 * N * N),
        ("synapse, bipolar (sqrt2 N) = equal matrix reals", "synapse", "real", round(math.sqrt(2) * N), round(2 ** 0.5 * N) * 2 // 2, round(math.sqrt(2) * N) ** 2),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, mem, kind, n, vr, mr in configs:
        for m in Ms:
            r = run(mem, kind, n, m, trials)
            rows.append({"code": label, "memory": mem, "kind": kind, "neurons": n, "memory_reals": mr, "M": m, **r})
            print(label, m, {k: round(v, 3) for k, v in r.items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps({"N": N, "F": F, "trials": trials, "rows": rows}, indent=2))
    L = [f"# P1: knowledge in synapses (Hebbian fast weights) vs in neuron state (N = {N}, F = {F}, {trials} trials)", "",
         "Retrieval accuracy (M = stored facts):", "", "| memory / code | memory reals | " + " | ".join(f"M={m}" for m in Ms) + " |",
         "|---" * (len(Ms) + 2) + "|"]
    for label, *_ in configs:
        rs = [r for r in rows if r["code"] == label]
        L.append(f"| {label} | {rs[0]['memory_reals']:,} | " + " | ".join(f"{r['acc']:.3f}" for r in rs) + " |")
    L += ["", "\"Don't know\" AUROC:", "", "| memory / code | " + " | ".join(f"M={m}" for m in Ms) + " |", "|---" * (len(Ms) + 1) + "|"]
    for label, *_ in configs:
        rs = [r for r in rows if r["code"] == label]
        L.append(f"| {label} | " + " | ".join(f"{r['auroc']:.3f}" for r in rs) + " |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
