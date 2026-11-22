"""I24: do the plastic synapse's keys separate because they tag the write time with phase (A), or because rotation
dephases the trace of earlier pulses so that each key is set by its own pulse (B)?

    cd research && .venv/bin/python -m analysis.i24_key_history [--models results/ep1b/i24_models]

When pair i arrives at time t_i, the hidden state = its own pulse (independent of t_i) + the traces of the pairs
shown before it (rotated by beta^dt in a cLIF network; with omega = 0 or LIF they decay without rotating, so they add up
in phase). For every pair we compare the network's actual key with its history-free key: the key the same network
produces at t_i when the input holds only that pair's pulse. Measures, on the test batch of I21 (same seeds):
  overlap_actual    mean |<k_i, k_j>|, i != j, of the actual keys at the true write times (= key_overlap of I21)
  overlap_free      the same for the history-free keys
  sim_actual_free   mean |<k_i actual, k_i history-free>| over pairs that have history (not the first one)
  sim_by_rank       the same by the pair's order in the sequence (rank 0 has no history, so its similarity is 1)
  history_coherence mean |cos| between the history parts (actual - history-free) of two pairs: how shared the
                    history is (chance = the overlap of random unit vectors)
  carry_over_excess |cos(history of pair i, history-free key of an earlier pair)| minus the same for a later pair
  history_alignment cos of the angle between the history term of an overlap, <k_i,k_j>_actual - <k_i,k_j>_free, and
                    the free term <k_i,k_j>_free (+1: history adds in phase and pushes keys together; 0: random; < 0: cancels)
  history_size      mean |history term| / mean |free term|
  empty             fraction of keys that are (near) zero because no hidden neuron fired
(A) predicts overlap_actual < overlap_free (history is what separates the keys).
(B) predicts, for the rotating network, actual ~ history-free (sim high, overlap_actual ~ overlap_free); for omega = 0
and LIF, actual keys far from their history-free keys and overlapping more than them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("cv_fw2", "cv_fw2_w0", "lif_fw2_d11")


def load(path: Path):
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ep1b import TASK

    ck = torch.load(path, weights_only=False)
    m = SequenceClassifier(TASK["n_keys"] + TASK["n_vals"] + 1, TASK["n_vals"], ModelConfig(**ck["model"]))
    m.load_state_dict(ck["state"])
    return m.eval()


def inner(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return (torch.conj(a) * c).sum(-1).abs() if torch.is_complex(a) else (a * c).sum(-1).abs()


def overlap(kk: torch.Tensor) -> float:
    """kk (B, P, H, d): mean |<k_i, k_j>| over i != j."""
    P = kk.shape[1]
    off = ~torch.eye(P, dtype=torch.bool)
    return inner(kk.unsqueeze(2), kk.unsqueeze(1)).permute(0, 3, 1, 2)[..., off].mean().item()


def analyse_model(model, seed: int, P: int = 8, n: int = 512) -> dict:
    from cvsnn.data.synthetic import assoc_recall_batch
    from experiments.ep1b import TASK

    x, y = assoc_recall_batch(n, pairs=P, generator=torch.Generator().manual_seed(20_000 + seed), **TASK)  # the I21 test batch
    cplx = model.cfg.kind == "cvsnn"
    run = lambda inp: model(torch.complex(inp, torch.zeros_like(inp)) if cplx else inp)
    with torch.no_grad():
        lg, _ = run(x)
        k = model.fw.last["k"]                                        # (T, B, H, d)
        B, w, nk = x.shape[1], TASK["window"], TASK["n_keys"]
        b = torch.arange(B)
        shown = x[:w, :, :nk].sum(-1) > 0
        times = torch.sort(torch.argsort(shown.T.float(), dim=1, descending=True, stable=True)[:, :P], dim=1).values  # (B, P), in order
        k_act = k[times, b.unsqueeze(1)]                              # (B, P, H, d)
        free = []
        for i in range(P):                                            # only pair i's pulse (write times are distinct)
            xi = torch.zeros_like(x)
            xi[times[:, i], b] = x[times[:, i], b]
            run(xi)
            free.append(model.fw.last["k"][times[:, i], b])           # (B, H, d)
        k_free = torch.stack(free, 1)
    sim = inner(k_act, k_free)                                        # (B, P, H)
    # what the history adds to each key, and whether it is shared between pairs: mean |cos| between the history parts
    # of two pairs (both with history). Shared (coherent) history pushes all keys the same way and raises their overlap.
    dk = (k_act - k_free)[:, 1:]                                      # (B, P-1, H, d)
    dn = dk / (torch.sqrt((dk.abs() ** 2).sum(-1, keepdim=True)) + 1e-9)
    # carry-over: does pair i's history part point at the history-free key of an EARLIER pair (that is in its
    # history) more than at a LATER one (control, cannot be in it)?
    ci = inner(dn.unsqueeze(2), k_free.unsqueeze(1)).permute(0, 3, 1, 2)  # (B, H, P-1, P): |cos(history of i, free key of j)|
    rank_i = torch.arange(1, P).view(-1, 1)
    earlier, later = torch.arange(P).view(1, -1) < rank_i, torch.arange(P).view(1, -1) > rank_i
    carry = ci[..., earlier].mean().item() - ci[..., later].mean().item()
    # how the history term of an overlap, <k_i, k_j>_actual - <k_i, k_j>_free, combines with the free term
    ip = lambda a, c: (torch.conj(a) * c).sum(-1) if torch.is_complex(a) else (a * c).sum(-1)
    A, F = ip(k_act.unsqueeze(2), k_act.unsqueeze(1)), ip(k_free.unsqueeze(2), k_free.unsqueeze(1))  # (B, P, P, H)
    both = ~torch.eye(P, dtype=torch.bool)
    both[0, :] = both[:, 0] = False                                   # both pairs have history
    Hs, Fs = (A - F).permute(0, 3, 1, 2)[..., both], F.permute(0, 3, 1, 2)[..., both]
    align = (((Hs * torch.conj(Fs)).real if torch.is_complex(Hs) else Hs * Fs) / (Hs.abs() * Fs.abs() + 1e-9)).mean().item()
    norm = lambda z: torch.sqrt((z.abs() ** 2).sum(-1))
    return {"seed": seed, "test_acc": (lg.argmax(-1) == y).float().mean().item(),
            "overlap_actual": overlap(k_act), "overlap_free": overlap(k_free),
            "sim_actual_free": sim[:, 1:].mean().item(), "sim_by_rank": sim.mean((0, 2)).tolist(), "history_coherence": overlap(dn),
            "carry_over_excess": carry, "history_alignment": align, "history_size": (Hs.abs().mean() / Fs.abs().mean()).item(),
            "empty_actual": (norm(k_act) < 0.5).float().mean().item(), "empty_free": (norm(k_free) < 0.5).float().mean().item()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(ROOT / "results" / "ep1b" / "i24_models"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    from cvsnn.utils import mean_ci95

    rows = []
    for v in VARIANTS:
        for path in sorted(Path(a.models).glob(f"P8_{v}_s*.pt"), key=lambda p: int(p.stem.split("_s")[-1])):
            r = analyse_model(load(path), int(path.stem.split("_s")[-1]))
            rows.append({"variant": v, **r})
        print(v, {k: (round(val, 3) if isinstance(val, float) else val) for k, val in r.items() if k != "sim_by_rank"}, flush=True)
    keys = ("test_acc", "overlap_actual", "overlap_free", "sim_actual_free", "history_coherence", "carry_over_excess",
            "history_alignment", "history_size", "empty_actual", "empty_free")
    L = ["# I24: actual keys vs history-free keys of the plastic synapse (P1b, P = 8, I21 models, 10 seeds)", "",
         "Mean ± 95% CI over seeds. Chance overlap of random unit vectors: complex d = 8 0.318, real d = 11 0.246.", "",
         "| variant | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    summary = {}
    for v in VARIANTS:
        rs = [r for r in rows if r["variant"] == v]
        if not rs:
            continue
        summary[v] = {k: mean_ci95([r[k] for r in rs]) for k in keys}
        summary[v]["sim_by_rank"] = [sum(r["sim_by_rank"][i] for r in rs) / len(rs) for i in range(len(rs[0]["sim_by_rank"]))]
        L.append(f"| {v} | " + " | ".join(f"{summary[v][k][0]:.3f} ± {summary[v][k][2]:.3f}" for k in keys) + " |")
    L += ["", "Similarity of actual and history-free key by the pair's order in the sequence (0 = first pair, no history):", "",
          "| variant | " + " | ".join(str(i) for i in range(8)) + " |", "|---" * 9 + "|"]
    for v, s in summary.items():
        L.append(f"| {v} | " + " | ".join(f"{x:.2f}" for x in s["sim_by_rank"]) + " |")
    md = "\n".join(L) + "\n"
    out = Path(a.out) if a.out else ROOT / "results" / "i24"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print(md)


if __name__ == "__main__":
    main()
