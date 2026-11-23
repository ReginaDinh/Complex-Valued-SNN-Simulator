"""Convert the bAbI 1k parquet files (20 tasks; HuggingFace RawthiL/babi_tasks, verified identical to the official
facebook/babi_qa en-valid-qa1 for task 1) into research/data/babi1k.json for experiments/ei72.py.

    cd research && .venv/bin/python -I tools/convert_babi.py data/babi_raw/rawthil data/babi1k.json

Output: a shared word vocabulary and answer list over all 20 tasks, and per task and split a list of samples
("sents": [[word ids] per sentence], "q": [word ids], "a": answer id). Multi-word answers (task 8 lists, task 19
paths) are kept as one answer label.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from cvsnn.data.babi import tokens


def main(raw: Path, out: Path) -> None:
    rows = {}
    for t in range(1, 21):
        for sp in ("train", "validation", "test"):
            rows[(t, sp)] = pq.read_table(raw / f"qa{t}_{sp}.parquet").to_pylist()
    words = sorted({w for rs in rows.values() for r in rs for w in tokens(r["passage"] + " " + r["question"])})
    answers = sorted({r["answer"].strip().lower() for rs in rows.values() for r in rs})
    wid = {w: i for i, w in enumerate(words)}
    aid = {a: i for i, a in enumerate(answers)}
    data = {"vocab": words, "answers": answers, "tasks": {}}
    for (t, sp), rs in rows.items():
        data["tasks"].setdefault(str(t), {})[sp] = [
            {"sents": [[wid[w] for w in tokens(line)] for line in r["passage"].split("\n") if line.strip()],
             "q": [wid[w] for w in tokens(r["question"])], "a": aid[r["answer"].strip().lower()]} for r in rs]
    out.write_text(json.dumps(data))
    lens = {t: max(len(s["sents"]) for sp in d.values() for s in sp) for t, d in data["tasks"].items()}
    print(f"vocab {len(words)}, answers {len(answers)}, max sentences per task {lens}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
