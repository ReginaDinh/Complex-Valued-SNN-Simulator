"""N1-S1: knowledge memory with phasor (complex) codes vs real codes, without training.

    cd research && .venv/bin/python -m analysis.n1_vsa            # full (about 1-2 min)
    .venv/bin/python -m analysis.n1_vsa --quick

A record stores M facts as a superposition of bound pairs, r = sum_i role_i (x) filler_i (e.g. colour (x) red +
shape (x) round + ...). A query unbinds one role and returns the most similar symbol of a codebook of F fillers.
Codes of N "neurons":
  fhrr     : unit phasors e^{j theta}, theta ~ U(0, 2 pi). bind = elementwise product (phases add), unbind = product
             with the conjugate (exact inverse), similarity Re<a, b> / N  (Plate's FHRR)
  phase_K  : phasors with K phase levels: the K-slot timing code of E-HW. K = 2 is the bipolar +-1 code
             (bind = elementwise product, its own inverse)
  hrr      : real Gaussian N(0, 1/N). bind = circular convolution, unbind = circular correlation (approximate inverse)
Comparisons at equal resources:
  (a) equal neurons N;
  (b) equal real numbers: a complex neuron holds 2 reals, so fhrr with N neurons vs bipolar / hrr with 2N;
  (c) equal bits: phase_K with N neurons (N log2 K bits) vs bipolar with N log2 K neurons.
Metrics: retrieval accuracy; "knowing what it does not know": AUROC for telling stored role queries from queries of
roles that were never stored, using the top similarity as confidence.

Prediction written before running: at equal real numbers / equal bits the codes are about equivalent (the unbinding
cross-talk has the same variance per real number; phase_4 is two bipolar codes), so a complex advantage, if any, is
not capacity per bit. Exact unbinding should make fhrr better than real hrr at equal real numbers.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "results" / "n1_vsa"
F = 1000          # fillers in the codebook
UNSTORED = 200    # role queries that were never stored (for the "don't know" test)


def make_codes(kind: str, n: int, count: int, g: torch.Generator) -> torch.Tensor:
    if kind == "hrr":
        return torch.randn(count, n, generator=g, dtype=torch.float64) / math.sqrt(n)
    if kind == "fhrr":
        return torch.polar(torch.ones(count, n, dtype=torch.float64), torch.rand(count, n, generator=g, dtype=torch.float64) * 2 * math.pi)
    K = int(kind.split("_")[1])
    if K == 2:
        return torch.randint(0, 2, (count, n), generator=g).double() * 2 - 1
    k = torch.randint(0, K, (count, n), generator=g).double()
    return torch.polar(torch.ones(count, n, dtype=torch.float64), k * 2 * math.pi / K)


def bind(kind: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if kind == "hrr":
        n = a.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(a) * torch.fft.rfft(b), n=n)
    return a * b


def unbind(kind: str, r: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    if kind == "hrr":
        n = a.shape[-1]
        return torch.fft.irfft(torch.fft.rfft(r) * torch.conj(torch.fft.rfft(a)), n=n)
    return r * (torch.conj(a) if torch.is_complex(a) else a)


def similarity(kind: str, v: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """(Q, N) x (F, N) -> (Q, F)."""
    n = codebook.shape[-1]
    if torch.is_complex(codebook):
        return (v @ torch.conj(codebook).T).real / n
    return v @ codebook.T * (1.0 if kind == "hrr" else 1.0 / n)


def auroc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score of a stored query > score of a never-stored query), ties counted half."""
    p, q = pos.view(-1, 1), neg.view(1, -1)
    return ((p > q).double().mean() + 0.5 * (p == q).double().mean()).item()


def trial(kind: str, n: int, m: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    codebook = make_codes(kind, n, F, g)
    roles = make_codes(kind, n, m + UNSTORED, g)
    stored, unstored = roles[:m], roles[m:]
    fill = torch.randint(0, F, (m,), generator=g)
    record = bind(kind, stored, codebook[fill]).sum(0, keepdim=True)
    s_in = similarity(kind, unbind(kind, record, stored), codebook)          # (m, F)
    s_out = similarity(kind, unbind(kind, record, unstored), codebook)       # (UNSTORED, F)
    acc = (s_in.argmax(-1) == fill).double().mean().item()
    top_in, top_out = s_in.max(-1).values, s_out.max(-1).values
    return {"acc": acc, "auroc": auroc(top_in, top_out), "top_in": top_in.mean().item(), "top_out": top_out.mean().item()}


def run(kind: str, n: int, m: int, trials: int) -> dict:
    rs = [trial(kind, n, m, 1000 * t + 17 * m + n) for t in range(trials)]
    return {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    trials = 2 if a.quick else 10
    Ms = [1, 4, 16, 64, 256] if a.quick else [1, 4, 16, 32, 64, 128, 256, 512]
    base_n = [256] if a.quick else [256, 1024]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for N in base_n:
        # (label, code kind, neurons, real numbers, bits): bits for float codes are counted as None
        configs = [
            ("fhrr (N phasors)", "fhrr", N, 2 * N, None),
            ("phase_16 (N)", "phase_16", N, 2 * N, 4 * N),
            ("phase_8 (N)", "phase_8", N, 2 * N, 3 * N),
            ("phase_4 (N)", "phase_4", N, 2 * N, 2 * N),
            ("bipolar (N)", "phase_2", N, N, N),
            ("bipolar (2N) = equal reals / equal bits to phase_4", "phase_2", 2 * N, 2 * N, 2 * N),
            ("bipolar (3N) = equal bits to phase_8", "phase_2", 3 * N, 3 * N, 3 * N),
            ("bipolar (4N) = equal bits to phase_16", "phase_2", 4 * N, 4 * N, 4 * N),
            ("hrr real (N)", "hrr", N, N, None),
            ("hrr real (2N) = equal reals to fhrr", "hrr", 2 * N, 2 * N, None),
        ]
        for label, kind, n, reals, bits in configs:
            for m in Ms:
                r = run(kind, n, m, trials)
                results.append({"N": N, "code": label, "kind": kind, "neurons": n, "reals": reals, "bits": bits, "M": m, **r})
                print(N, label, m, {k: round(v, 3) for k, v in r.items()}, flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    L = [f"# N1-S1: phasor vs real codes for a key-value knowledge memory (F = {F} fillers, {trials} trials)", ""]
    for N in base_n:
        rows = [r for r in results if r["N"] == N]
        codes = list(dict.fromkeys(r["code"] for r in rows))
        L += [f"## N = {N}", "", "Retrieval accuracy (M = facts stored in one record):", "",
              "| code | reals | bits | " + " | ".join(f"M={m}" for m in Ms) + " |", "|---" * (len(Ms) + 3) + "|"]
        for c in codes:
            rs = [r for r in rows if r["code"] == c]
            L.append(f"| {c} | {rs[0]['reals']} | {rs[0]['bits'] or 'float'} | " + " | ".join(f"{r['acc']:.3f}" for r in rs) + " |")
        L += ["", "\"Don't know\" AUROC (stored vs never-stored role queries, by top similarity):", "",
              "| code | " + " | ".join(f"M={m}" for m in Ms) + " |", "|---" * (len(Ms) + 1) + "|"]
        for c in codes:
            rs = [r for r in rows if r["code"] == c]
            L.append(f"| {c} | " + " | ".join(f"{r['auroc']:.3f}" for r in rs) + " |")
        L.append("")
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
