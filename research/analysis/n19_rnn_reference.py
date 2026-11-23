"""N19: non-spiking references (GRU, LSTM; parameter-matched, same plastic synapse, read-out and protocol) vs the rotating
long-memory cLIF and the spiking baselines.

    cd research && .venv/bin/python -m analysis.n19_rnn_reference [out_dir]

Synthetic sequence (I54i protocol): GRU / LSTM selected on seeds 0-4 over lr_rec x lr_ecl x multiplier (results/n19/syn),
10 seeds at the selection; the rotating network from I54i (same seeds). Benchmarks with replay (I64a / I64b protocol): the
recall lr and multiplier inherited from the synthetic selection, the later-phase lr selected on seeds 0-4; the rotating
network and the spiking baselines from I64a / I64b / I67 at their selections. Sign test (two-sided, ties dropped) and
paired differences with 95% CI over the 10 seeds.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign_p(w: int, l: int) -> float:
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(max(w, l), n + 1)) / 2 ** n)


def load(patterns) -> dict:
    rows = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat), recursive=True)):
            for l in open(f):
                r = json.loads(l)
                rows.setdefault(r["key"], r)
    return rows


def cell(rows, variant, lr) -> dict:
    got = {r["seed"]: r for r in rows.values() if r["variant"] == variant and r["lr"] == lr}
    return {s: got[s] for s in sorted(got) if s < 10}


def select(rows, prefix_variants, lrs_filter=lambda lr: True):
    g = defaultdict(list)
    for r in rows.values():
        if r["variant"] in prefix_variants and r["seed"] < 5 and lrs_filter(r["lr"]):
            g[(r["variant"], r["lr"])].append(r["val_score"])
    full = {k: sum(x) / 5 for k, x in g.items() if len(x) == 5}
    best = max(full, key=full.get)
    return best, full


def main(out: Path) -> None:
    from cvsnn.utils import mean_ci95
    L = ["# N19: non-spiking references (GRU, LSTM) vs the rotating long-memory cLIF", ""]
    res = {}
    # --- synthetic sequence
    syn = load(["results/n19/syn/**/runs.jsonl"])
    ref = load(["results/ei54i/run1/runs.jsonl"])
    rot = cell(ref, "cand_long_fw003|2heads|phaselr", "0.01/0.0003")
    L += ["## Synthetic sequence (recall 8 pairs, then 5 later tasks; two read-outs)", "",
          "| model | params | selected (multiplier; lr_rec → lr_ecl) | seeds | recall at end | fails | later-task final | forgetting | val score |",
          "|---|---|---|---|---|---|---|---|---|"]
    cells_syn = {"rotating cLIF (I54i)": rot}
    sel_syn = {}
    for fam, names in (("GRU", ("gru_slowfw", "gru_fw003", "gru_fw001", "gru_fw0003")), ("LSTM", ("lstm_slowfw", "lstm_fw003", "lstm_fw001"))):
        (v, lr), full = select(syn, {n + "|2heads|phaselr" for n in names})
        sel_syn[fam] = (v.split("|")[0], lr)
        cells_syn[fam] = cell(syn, v, lr)
    for name, c in cells_syn.items():
        rs = list(c.values())
        m = {k: mean_ci95([r[k] for r in rs]) for k in ("recall_final", "ecl_final_avg", "ecl_forgetting", "val_score")}
        sel = "×0.03; 1e-2 → 3e-4" if name.startswith("rotating") else f"{sel_syn[name][0]} {sel_syn[name][1]}"
        L.append(f"| {name} | {rs[0]['params']} | {sel} | {len(rs)} | {m['recall_final'][0]:.3f} ± {m['recall_final'][2]:.3f} | "
                 f"{sum(r['recall_final'] < 0.8 for r in rs)}/{len(rs)} | {m['ecl_final_avg'][0]:.3f} ± {m['ecl_final_avg'][2]:.3f} | "
                 f"{m['ecl_forgetting'][0]:.1%} ± {m['ecl_forgetting'][2]:.1%} | {m['val_score'][0]:.3f} |")
        res[f"syn|{name}"] = {k: m[k] for k in m}
    L += ["", "Per seed, rotating cLIF vs each reference (rotating better = higher val score / later-task accuracy, lower forgetting):", "",
          "| vs | val score | later-task final | forgetting | recall | paired diff, later-task final |", "|---|---|---|---|---|---|"]
    for fam in ("GRU", "LSTM"):
        c = cells_syn[fam]
        seeds = sorted(set(rot) & set(c))
        out_ = []
        for k, better in (("val_score", 1), ("ecl_final_avg", 1), ("ecl_forgetting", -1), ("recall_final", 1)):
            w = sum((rot[s][k] - c[s][k]) * better > 0 for s in seeds)
            l_ = sum((rot[s][k] - c[s][k]) * better < 0 for s in seeds)
            out_.append(f"{w}/{len(seeds)} (p = {sign_p(w, l_):.3f})")
        d = mean_ci95([rot[s]["ecl_final_avg"] - c[s]["ecl_final_avg"] for s in seeds])
        L.append(f"| {fam} | " + " | ".join(out_) + f" | {d[0]:+.3f} ± {d[2]:.3f} |")
    # --- benchmarks with replay
    bench = load(["results/n19/mnist/**/runs.jsonl", "results/n19/shd/**/runs.jsonl"])
    old = load(["results/ei64/**/runs.jsonl", "results/ei67/**/runs.jsonl"])
    base = {"rotating cLIF": ("cand_long_fw003", {"mnist": "0.01/0.001", "shd": "0.01/0.001"}),
            "no-rotation cLIF": ("cand_longnorot_fw001", {"mnist": "0.1/0.003", "shd": "0.1/0.001"}),
            "LIF (long)": ("lif_fw2_d11_fw001", {"mnist": "0.1/0.003", "shd": "0.1/0.001"}),
            "LIF (short)": ("lif_short_fw003", {"mnist": "0.03/0.003", "shd": "0.03/0.001"}),
            "SE-adLIF": ("se_adlif_fw001", {"mnist": "0.1/0.001", "shd": "0.1/0.003"}),
            "BRF": ("brfv2_fw003", {"mnist": "0.01/0.003", "shd": "0.01/0.001"})}
    for task, label in (("mnist", "Permuted MNIST + replay (5 tasks)"), ("shd", "Split-SHD + replay (5 tasks)")):
        L += ["", f"## {label}", "", "| model | params | later-phase lr | seeds | final accuracy | learned | forgetting | rotating higher in | paired diff (rotating − model) |",
              "|---|---|---|---|---|---|---|---|---|"]
        cells_b = {}
        for name, (v, lrs) in base.items():
            cells_b[name] = (cell(old, f"{v}|2heads|phaselr|{task}|h120|rp100", lrs[task]), lrs[task])
        for fam in ("GRU", "LSTM"):
            v = sel_syn[fam][0]
            (vv, lr), full = select(bench, {f"{v}|2heads|phaselr|{task}|h120|rp100"})
            tested = sorted(float(k[1].split("/")[1]) for k in full)
            edge = " ⚠ edge" if float(lr.split("/")[1]) in (tested[0], tested[-1]) else ""
            cells_b[fam] = (cell(bench, vv, lr), lr + edge)
        rot_b = cells_b["rotating cLIF"][0]
        for name, (c, lr) in cells_b.items():
            rs = list(c.values())
            if not rs:
                continue
            m = {k: mean_ci95([r[k] for r in rs]) for k in ("ecl_final_avg", "ecl_learned_avg", "ecl_forgetting")}
            seeds = sorted(set(rot_b) & set(c))
            if name == "rotating cLIF":
                cmp_, d_ = "–", "–"
            else:
                w = sum(rot_b[s]["ecl_final_avg"] > c[s]["ecl_final_avg"] for s in seeds)
                l_ = sum(rot_b[s]["ecl_final_avg"] < c[s]["ecl_final_avg"] for s in seeds)
                d = mean_ci95([rot_b[s]["ecl_final_avg"] - c[s]["ecl_final_avg"] for s in seeds])
                cmp_, d_ = f"{w}/{len(seeds)} (p = {sign_p(w, l_):.3f})", f"{d[0]:+.3f} ± {d[2]:.3f}"
            L.append(f"| {name} | {rs[0]['params']} | {lr} | {len(rs)} | {m['ecl_final_avg'][0]:.3f} ± {m['ecl_final_avg'][2]:.3f} | "
                     f"{m['ecl_learned_avg'][0]:.3f} | {m['ecl_forgetting'][0]:.1%} | {cmp_} | {d_} |")
            res[f"{task}|{name}"] = {k: m[k] for k in m} | {"lr": lr}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(L) + "\n")
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/n19")
