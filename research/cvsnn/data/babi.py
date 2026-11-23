"""bAbI (Weston et al. 2015) for the sequence runner: tokenised stories -> per-step input currents.

A story of n sentences becomes 2n + 5 steps: each sentence as a bag of words over the shared vocabulary for one step,
then one empty step; then the question (bag of words plus a question flag channel) for two steps and three empty
steps, so that the read-out over the last 5 steps sees the question and what follows. Shorter stories in a batch
are padded with empty steps at the START (every sequence ends where the read-out is).
"""

from __future__ import annotations

import re

Q_STEPS, TAIL = 2, 3


def tokens(text:str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def steps_of(n_sequences:int) -> int:
    return 2 * n_sequences + Q_STEPS + TAIL


def encode_batch(samples: list[dict], vocab_size: int):
    """samples: dicts with "sents" (list of word-id lists) and "q" (word ids). Returns (T, B, vocab_size + 1) floats."""
    import torch
    T = max(steps_of(len(s["sents"])) for s in samples)
    x = torch.zeros(T, len(samples), vocab_size + 1)
    for b, s in enumerate(samples):
        t0 = T - steps_of(len(s["sents"]))
        for i, sent in enumerate(s["sents"]):
            for w in sent:
                x[t0 + 2 * i, b, w] = 1.0
        tq = t0 + 2 * len(s["sents"])
        for k in range(Q_STEPS):
            for w in s["q"]:
                x[tq + k, b, w] = 1.0
            x[tq + k, b, vocab_size] = 1.0   # question flag
    return x
