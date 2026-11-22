"""I23d (theory-light, no training): can a property of the frequency set's "time basis" explain why regular frequency
lattices help the wide-window memory task? (I23, I23b, I23c, I28a; H6d table)

    cd research && .venv/bin/python -m analysis.i23d_time_basis

For each frequency set (32 neurons, |beta| drawn on the ring r^2 ~ U(0.99^2, 0.9999^2) as in the selected H6d
setting, 5 draws), an event at lag L before the read-out leaves the trace z_j(L) = |beta_j|^L e^{j w_j L} on neuron
j. Over the task's lags L in [D, D + W) = [400, 800) this gives a code matrix C (W x 32). Measures:
  eff_rank     participation ratio of the singular values of C (how many independent time directions)
  crowding     mean |cos| between the codes of lags at least 10 steps apart (0 = all distinct)
  decode_acc   a ridge read-out decoding the lag bin (16 bins of 25 steps) from z + noise (sigma 0.05 x code norm);
               the read-out sees Re and Im, as the network's read-out does
  slow         neurons with w < 0.05 rad/step
and Spearman correlations of each with the trained-network accuracy at W = 400 (frozen frequencies, H6d/I23/I23b/
I23c/I28a). Exploratory: a correlation only suggests which property matters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ACC_W400 = {  # trained cv_phasor, frozen frequencies, W = 400 (docs/results: eh6, i23, i23b, i23c, i28a)
    "zero": 0.943, "ladder2": 0.875, "phi_ladder": 0.866, "brain5": 0.859, "ladder5": 0.849, "phi_jitter": 0.802,
    "banded_strat": 0.754, "quantized12": 0.746, "quantized8": 0.739, "banded_log": 0.709, "geometric": 0.708, "log": 0.654,
    "sidon12": 0.600, "harmonic": 0.583, "independent": 0.554, "golden": 0.540, "sidon8": 0.525, "uniform": 0.506}
W, D, N, LO, HI = 400, 400, 32, 0.01, 1.5


def freqs(name: str, g: torch.Generator) -> torch.Tensor:
    from cvsnn.frequencies import frequency_set
    if name.startswith("quantized") or name.startswith("sidon"):
        kind, bits = name.rstrip("0123456789"), int(name[len(name.rstrip("0123456789")):])
        return frequency_set(kind, N, LO, HI, bits=bits, generator=g)
    return frequency_set(name, N, LO, HI, generator=g)


def measures(w: torch.Tensor, g: torch.Generator) -> dict:
    r = torch.sqrt(torch.rand(N, generator=g, dtype=torch.float64) * (0.9999 ** 2 - 0.99 ** 2) + 0.99 ** 2)
    L = torch.arange(D, D + W, dtype=torch.float64).view(-1, 1)
    C = torch.polar(r.view(1, -1) ** L, w.view(1, -1) * L)                       # (W, N) complex
    sv = torch.linalg.svdvals(C)
    eff = (sv.sum() ** 2 / (sv ** 2).sum()).item()
    U = C / C.abs().pow(2).sum(1, keepdim=True).sqrt()
    G = (U @ U.conj().T).abs()
    far = (torch.arange(W).view(-1, 1) - torch.arange(W).view(1, -1)).abs() >= 10
    crowd = G[far].mean().item()
    X = torch.cat([C.real, C.imag], 1)                                            # read-out sees Re and Im
    bins = (torch.arange(W) // 25).long()
    Y = torch.nn.functional.one_hot(bins, 16).double()
    sig = 0.05 * X.norm(dim=1).mean()
    Xtr = X.repeat(8, 1) + sig * torch.randn(8 * W, 2 * N, generator=g, dtype=torch.float64)
    Xte = X.repeat(4, 1) + sig * torch.randn(4 * W, 2 * N, generator=g, dtype=torch.float64)
    Ytr = Y.repeat(8, 1)
    A = Xtr.T @ Xtr + 1e-3 * torch.eye(2 * N, dtype=torch.float64) * Xtr.pow(2).mean() * len(Xtr)
    Wr = torch.linalg.solve(A, Xtr.T @ Ytr)
    acc = ((Xte @ Wr).argmax(1) == bins.repeat(4)).double().mean().item()
    return {"eff_rank": eff, "crowding": crowd, "decode_acc": acc, "slow": int((w < 0.05).sum())}


def spearman(x, y):
    rank = lambda a: torch.tensor(a).argsort().argsort().double()
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return (rx @ ry / (rx.norm() * ry.norm())).item()


def main() -> None:
    rows = {}
    for name in ACC_W400:
        ms = [measures(freqs(name, torch.Generator().manual_seed(s)).double(), torch.Generator().manual_seed(100 + s)) for s in range(5)]
        rows[name] = {k: sum(m[k] for m in ms) / len(ms) for k in ms[0]}
    names = list(ACC_W400)
    acc = [ACC_W400[n] for n in names]
    L = ["# I23d: properties of each frequency set's time basis vs trained accuracy at W = 400 (exploratory, no training)", "",
         "| set | trained acc (W = 400) | eff_rank | crowding | decode_acc | slow (w < 0.05) |", "|---|---|---|---|---|---|"]
    for n in sorted(names, key=lambda n: -ACC_W400[n]):
        r = rows[n]
        L.append(f"| {n} | {ACC_W400[n]:.3f} | {r['eff_rank']:.2f} | {r['crowding']:.3f} | {r['decode_acc']:.3f} | {r['slow']:.1f} |")
    L += ["", "Spearman correlation with trained accuracy (18 sets): " + ", ".join(f"{k} {spearman([rows[n][k] for n in names], acc):+.2f}"
                                                                              for k in ("eff_rank", "crowding", "decode_acc", "slow"))]
    md = "\n".join(L) + "\n"
    out = ROOT / "results" / "i23d"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
