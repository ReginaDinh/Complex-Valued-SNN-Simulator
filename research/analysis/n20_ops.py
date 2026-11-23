"""N20: operation counts per inference sample on the two benchmarks: spiking networks (measured spike sparsity) vs the
parameter-matched non-spiking GRU / LSTM references (N19).

    cd research && .venv/bin/python -m analysis.n20_ops [out_dir]

Counting model (event-driven hardware; the same rules for every network):
- a binary spike entering a synapse costs one accumulate (AC); a real-valued input costs one multiply-accumulate (MAC);
- zero inputs are skipped for every network (input sparsity counted from the data, the same for all);
- spiking hidden layer, per step: input W x_t (nnz(x_t) * N MAC); neuron update (cLIF: complex multiply by lambda = 4 MAC
  + 1 AC per neuron; LIF: 1 MAC + 1 AC);
- GRU / LSTM, per step: input part nnz(x_t) * G*H MAC; recurrent part G*H*H MAC (dense, every step); gate arithmetic
  (GRU 4H, LSTM 5H MAC) (G = 3 for GRU, 4 for LSTM);
- plastic synapse (identical in every network): key / value / query / gate projections from the hidden output (spikes:
  nnz(h_t) * (3*heads*d + heads) AC; GRU / LSTM: H * (3*heads*d + heads) MAC); key and query normalisation
  (4*heads*d MAC); gated delta-rule write, decay and read (4*heads*d^2 MAC);
- read-out: from the hidden output (spikes: nnz(h_t) * n_out AC; GRU / LSTM: H * n_out MAC), from the plastic-synapse
  read (heads*d * n_out MAC), leaky read-out units (n_out MAC).
Nonlinearities (sigmoid, tanh, spike threshold) are not counted. Energy proxy: 4.6 pJ per 32-bit float MAC and 0.9 pJ per
AC, the 45 nm figures commonly quoted from Horowitz (2014); a rough guide only.
Spike counts: the trained models saved by N10 / N12 (selected settings, 10 seeds), on the test data of all five tasks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
E_MAC, E_AC = 4.6e-12, 0.9e-12
HEADS, D = 2, 11
BASE = {"cand_long_fw003": "cand_long", "cand_longnorot_fw001": "cand_long_norot", "lif_fw2_d11_fw001": "lif_fw2_d11"}
ACC = {"mnist": {"rotating cLIF": 0.656, "no-rotation cLIF": 0.614, "LIF (long)": 0.621, "GRU": 0.716, "LSTM": 0.685},
       "shd": {"rotating cLIF": 0.682, "no-rotation cLIF": 0.604, "LIF (long)": 0.614, "GRU": 0.758, "LSTM": 0.736}}


def task_data(task: str, seed: int, c_rec: int):
    """Test inputs of all five later tasks, as experiments/ei54.py builds them (MNIST: 512 images per task)."""
    from experiments.ei54 import C_ECL, T
    xs = []
    if task == "mnist":
        d = np.load(ROOT / "data/mnist.npz")
        X = torch.tensor(d["x_test"], dtype=torch.float32).reshape(-1, 784) / 255.0
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(123))[:512]
        for t in range(5):
            perm = torch.arange(784) if t == 0 else torch.randperm(784, generator=torch.Generator().manual_seed(900 + 31 * seed + t))
            full = torch.zeros(T, len(idx), C_ECL + c_rec)
            full[T - 28:, :, :28] = X[idx][:, perm].reshape(-1, 28, 28).permute(1, 0, 2)
            xs.append(full)
    else:
        from cvsnn.data.shd import encode
        d = np.load(ROOT / "data/shd.npz")
        X = torch.tensor(d["x_test"])
        full = torch.zeros(T, len(X), C_ECL + c_rec)
        full[:, :, :C_ECL] = encode(X).permute(1, 0, 2)
        xs.append(full)
    return torch.cat(xs, 1)


def spiking_counts(row: dict, x: torch.Tensor, n_out: int) -> dict:
    from cvsnn.layers import CLIF
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei54 import EXTRA_MODELS, MODELS
    base = BASE[row["variant"].split("|")[0]]
    cfg = {**(MODELS[base] if base in MODELS else EXTRA_MODELS[base]), "threshold": 0.5, "hidden": [120]}
    model = SequenceClassifier(x.shape[-1], n_out, ModelConfig(**cfg))
    model.load_state_dict(torch.load(row["save_model"], weights_only=False)["state_dict"])
    caught = {}
    model.neurons[0].register_forward_hook(lambda m, i, o: caught.update(fire=o[2].detach()))
    model.eval()
    with torch.no_grad():
        for k in range(0, x.shape[1], 512):
            model(x[:, k:k + 512])
            caught.setdefault("spikes", 0.0)
            caught["spikes"] += float(caught["fire"].sum())
    T, B, _ = x.shape
    N = 120
    nnz_x = float((x != 0).sum()) / B            # input non-zeros per sample (whole sequence)
    spk = caught["spikes"] / B                    # hidden spikes per sample (whole sequence)
    cplx = isinstance(model.neurons[0], CLIF)
    mac = nnz_x * N + T * N * (4 if cplx else 1) + T * (4 * HEADS * D + 4 * HEADS * D * D) + T * (HEADS * D * n_out + n_out)
    ac = T * N + spk * (3 * HEADS * D + HEADS) + spk * n_out
    return {"mac": mac, "ac": ac, "spike_rate": spk / (T * N), "nnz_x": nnz_x}


def rnn_counts(cell: str, H: int, x: torch.Tensor, n_out: int) -> dict:
    T, B, _ = x.shape
    G, elem = (3, 4) if cell == "gru" else (4, 5)
    nnz_x = float((x != 0).sum()) / B
    mac = nnz_x * G * H + T * (G * H * H + elem * H) + T * H * (3 * HEADS * D + HEADS) + T * (4 * HEADS * D + 4 * HEADS * D * D) \
        + T * (H * n_out + HEADS * D * n_out + n_out)
    return {"mac": mac, "ac": 0.0, "nnz_x": nnz_x}


def main(out: Path) -> None:
    from cvsnn.utils import mean_ci95
    from experiments.ep1b import TASK as REC_TASK
    c_rec = REC_TASK["n_keys"] + REC_TASK["n_vals"] + 1
    rows = [{"cell": "rotating long", **json.loads(l)} for l in (ROOT / "results/n10/runs.jsonl").read_text().splitlines()]
    rows += [json.loads(l) for l in (ROOT / "results/n12/runs.jsonl").read_text().splitlines()]
    names = {"rotating long": "rotating cLIF", "no-rotation long": "no-rotation cLIF", "LIF long": "LIF (long)"}
    L = ["# N20: operations per inference sample (whole sequence), event-driven counting", "",
         "MAC = multiply-accumulate, AC = accumulate (a binary spike into a synapse). Energy proxy: 4.6 pJ / MAC + 0.9 pJ / AC (45 nm, as commonly quoted from Horowitz 2014). "
         "Spiking networks: 10 seeds, mean ± 95% CI of the per-seed counts; GRU / LSTM: from the layer sizes (dense, activity-independent).", ""]
    res = {}
    for task, setting, n_out, H in (("mnist", "permuted MNIST + replay (I64b)", 18, {"gru": 44, "lstm": 37}), ("shd", "Split-SHD + replay (I64a)", 12, {"gru": 43, "lstm": 36})):
        L += [f"## {'Permuted MNIST' if task == 'mnist' else 'Split-SHD'} (with replay)", "",
              "| model | final accuracy | hidden spike rate | MAC per sample | AC per sample | energy proxy (nJ) | MAC vs rotating | energy vs rotating |", "|---|---|---|---|---|---|---|---|"]
        per = {}
        for cell, name in names.items():
            rs = sorted((r for r in rows if r["cell"] == cell and r["setting"] == setting), key=lambda r: r["seed"])
            cs = [spiking_counts(r, task_data(task, r["seed"], c_rec), n_out) for r in rs]
            per[name] = {k: mean_ci95([c[k] for c in cs]) for k in ("mac", "ac", "spike_rate")} | {"energy": mean_ci95([E_MAC * c["mac"] + E_AC * c["ac"] for c in cs])}
        x0 = task_data(task, 0, c_rec)
        for cell, name in (("gru", "GRU"), ("lstm", "LSTM")):
            c = rnn_counts(cell, H[cell], x0, n_out)
            per[name] = {"mac": (c["mac"], 0, 0), "ac": (0.0, 0, 0), "spike_rate": None, "energy": (E_MAC * c["mac"], 0, 0)}
        rot = per["rotating cLIF"]
        for name, v in per.items():
            sr = "–" if v["spike_rate"] is None else f"{v['spike_rate'][0]:.3f}"
            ci = lambda q, f: f"{q[0]:{f}}" + (f" ± {q[2]:{f}}" if q[2] else "")
            L.append(f"| {name} | {ACC[task][name]:.3f} | {sr} | {ci(v['mac'], ',.0f')} | {ci(v['ac'], ',.0f')} | {v['energy'][0] * 1e9:.1f} | "
                     f"{v['mac'][0] / rot['mac'][0]:.2f}× | {v['energy'][0] / rot['energy'][0]:.2f}× |")
            res[f"{task}|{name}"] = {k: (v[k][0] if v[k] is not None else None) for k in v}
        L.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/n20")
