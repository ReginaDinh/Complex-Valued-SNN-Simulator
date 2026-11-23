"""Synthetic multi-channel complex carriers: a vectorised port of ``generateSpectralSignal``
in ``src/simulationEngine.ts`` (same class definitions, normalised time t / T)."""

from __future__ import annotations

import math

import torch


def spectral_batch(
    targets: torch.Tensor,
    time_steps: int = 30,
    features: int = 16,
    freq0: float = 1.5,
    freq1: float = 4.5,
    phase_shift: float = math.pi / 2,
    noise: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Returns complex64 (T, B, F).

    * Class 0: carrier at ``freq0`` cycles/window, no inter-channel phase gradient.
    * Class 1: carrier at ``freq1`` with inter-channel phase gradient ``phase_shift``.
    * The last quarter of the channels is uncorrelated clutter.
    Iso-frequency (phase-only) task: ``freq1 == freq0``.
    """
    B = targets.shape[0]
    g = generator
    n_sig = (features * 3) // 4
    t = (torch.arange(time_steps, dtype=torch.float32) / time_steps).view(time_steps, 1, 1)
    is1 = (targets == 1).view(1, B, 1).float()
    freq = freq0 + (freq1 - freq0) * is1
    shift = phase_shift * is1
    ch = torch.arange(n_sig, dtype=torch.float32).view(1, 1, n_sig)

    mag = 0.85 + (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 0.15
    phase = 2 * math.pi * freq * t + ch * shift
    nre = (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 2 * noise
    nim = (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 2 * noise
    sig = torch.complex(mag * torch.cos(phase) + nre, mag * torch.sin(phase) + nim)

    n_clut = features - n_sig
    cmag = torch.rand(time_steps, B, n_clut, generator=g) * (0.35 + noise * 0.5)
    cph = torch.rand(time_steps, B, n_clut, generator=g) * 2 * math.pi
    clutter = torch.polar(cmag, cph)
    return torch.cat([sig, clutter], dim=-1)


def phase_class_batch(
    targets: torch.Tensor,
    time_steps: int = 30,
    features: int = 16,
    freq: float = 1.5,
    delta: float = math.pi / 8,
    noise: float = 0.15,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Multi-class phase-only task: class c has inter-channel phase gradient c * delta, all classes share
    one carrier frequency and amplitude statistics. Small ``delta`` demands phase precision.
    The global phase is random per sample, so only *relative* channel phases carry the class."""
    B = targets.shape[0]
    g = generator
    n_sig = (features * 3) // 4
    t = (torch.arange(time_steps, dtype=torch.float32) / time_steps).view(time_steps, 1, 1)
    grad = (targets.float() * delta).view(1, B, 1)
    offset = (torch.rand(1, B, 1, generator=g) * 2 * math.pi)
    ch = torch.arange(n_sig, dtype=torch.float32).view(1, 1, n_sig)
    mag = 0.85 + (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 0.15
    phase = 2 * math.pi * freq * t + ch * grad + offset
    nre = (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 2 * noise
    nim = (torch.rand(time_steps, B, n_sig, generator=g) - 0.5) * 2 * noise
    sig = torch.complex(mag * torch.cos(phase) + nre, mag * torch.sin(phase) + nim)
    n_clut = features - n_sig
    clutter = torch.polar(torch.rand(time_steps, B, n_clut, generator=g) * (0.35 + noise * 0.5),
                          torch.rand(time_steps, B, n_clut, generator=g) * 2 * math.pi)
    return torch.cat([sig, clutter], dim=-1)


def delayed_order_batch(
    batch: int,
    symbols: int = 4,
    channels: int = 8,
    window: int = 10,
    delay: int = 50,
    tail: int = 5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal-order memory task (after Hochreiter & Schmidhuber, 1997).

    Two different symbols appear as one-step pulses at two random times inside the first ``window``
    steps; then ``delay`` silent steps; then ``tail`` read-out steps (also silent). The label is the
    ordered pair (first, second): symbols * (symbols - 1) classes. Symbol s drives channels
    [s * k, (s + 1) * k) with k = channels // symbols. The input is real (no phase): any phase a model
    uses must come from its own dynamics. Returns real x (T, B, channels) and labels (B,).
    """
    g = generator
    T = window + delay + tail
    k = channels // symbols
    a = torch.randint(0, symbols, (batch,), generator=g)
    b = (a + torch.randint(1, symbols, (batch,), generator=g)) % symbols   # b != a
    t1 = torch.randint(0, window - 1, (batch,), generator=g)
    t2 = t1 + 1 + (torch.rand(batch, generator=g) * (window - 1 - t1).float()).long()  # t1 < t2 < window
    x = torch.zeros(T, batch, channels)
    idx = torch.arange(batch)
    for s_vec, t_vec in ((a, t1), (b, t2)):
        for j in range(k):
            x[t_vec, idx, s_vec * k + j] = 1.0
    pair = a * (symbols - 1) + torch.where(b > a, b - 1, b)   # index of ordered pair (a, b)
    return x, pair


def binding_batch(
    targets: torch.Tensor,
    time_steps: int = 30,
    freq: float = 3.0,
    noise: float = 0.3,
    sep_min: float = math.pi / 4,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """I17, cross-modal binding by phase. Two objects, each with a colour (modality A: channels 0-3) and a shape
    (modality B: channels 4-7). The two feature channels of one object carry the same oscillation
    e^{j(2 pi freq t / T + phi_object)}; the two objects differ in phase by at least sep_min (random global phase).
    Which colours and shapes are present is random and says nothing about the label:
      target 1: the lower-index colour goes with the lower-index shape; target 0: it goes with the higher one.
    Only the phase relation between the modalities tells them apart. Inactive channels carry noise only.
    Returns complex64 (T, B, 8)."""
    B = targets.shape[0]
    g = generator
    cols = torch.stack([torch.randperm(4, generator=g)[:2] for _ in range(B)]).sort(-1).values      # (B, 2), ca < cb
    shps = torch.stack([torch.randperm(4, generator=g)[:2] for _ in range(B)]).sort(-1).values + 4
    phi1 = torch.rand(B, generator=g) * 2 * math.pi
    phi2 = phi1 + sep_min + torch.rand(B, generator=g) * (2 * math.pi - 2 * sep_min)
    t = (torch.arange(time_steps, dtype=torch.float32) / time_steps).view(time_steps, 1)
    x = torch.complex((torch.rand(time_steps, B, 8, generator=g) - 0.5) * 2 * noise,
                      (torch.rand(time_steps, B, 8, generator=g) - 0.5) * 2 * noise)
    idx = torch.arange(B)
    same = targets.bool()
    shape_of_ca = torch.where(same, shps[:, 0], shps[:, 1])
    shape_of_cb = torch.where(same, shps[:, 1], shps[:, 0])
    for col, shp, phi in ((cols[:, 0], shape_of_ca, phi1), (cols[:, 1], shape_of_cb, phi2)):
        mag = 0.85 + (torch.rand(time_steps, B, generator=g) - 0.5) * 0.15
        wave = torch.polar(mag, 2 * math.pi * freq * t + phi.view(1, B))
        for ch in (col, shp):
            x[:, idx, ch] = x[:, idx, ch] + wave
    return x


def assoc_recall_batch(
    batch: int,
    pairs: int = 4,
    n_keys: int = 8,
    n_vals: int = 8,
    window: int = 40,
    delay: int = 40,
    tail: int = 5,
    generator: torch.Generator | None = None,
    sustained_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """P1b, associative recall (key-value knowledge written into a network as it runs). `pairs` distinct keys,
    each with a random value, appear as one-step pulses (key channel + value channel together) at random distinct
    times in the first `window` steps; after `delay` silent steps a query arrives (one key channel + the query flag,
    channel n_keys + n_vals) at the first of `tail` read-out steps (sustained_query: on every read-out step). Label: the value stored with the queried key
    (n_vals classes). Returns real x (T, B, n_keys + n_vals + 1) and labels (B,)."""
    g = generator
    T = window + delay + tail
    x = torch.zeros(T, batch, n_keys + n_vals + 1)
    keys = torch.stack([torch.randperm(n_keys, generator=g)[:pairs] for _ in range(batch)])           # (B, P)
    vals = torch.randint(0, n_vals, (batch, pairs), generator=g)
    times = torch.stack([torch.randperm(window, generator=g)[:pairs] for _ in range(batch)])           # (B, P)
    b = torch.arange(batch)
    for i in range(pairs):
        x[times[:, i], b, keys[:, i]] = 1.0
        x[times[:, i], b, n_keys + vals[:, i]] = 1.0
    which = torch.randint(0, pairs, (batch,), generator=g)
    q_steps = range(window + delay, T) if sustained_query else [window + delay]
    for qt in q_steps:
        x[qt, b, keys[b, which]] = 1.0
        x[qt, b, n_keys + n_vals] = 1.0
    return x, vals[b, which]


def serial_recall_batch(
    batch: int,
    items: int = 8,
    n_symbols: int = 8,
    slot: int = 4,
    delay: int = 40,
    tail: int = 5,
    generator: torch.Generator | None = None,
    cycle_slots: int | None = None,
    cycle_cue: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """I28b, ordered (positional) recall with a theta clock.
    I28c: cycle_slots (default = items) gamma slots per theta cycle; a list longer than one cycle wraps around, so
    slots n and n + cycle_slots share a theta phase. cycle_cue adds one channel per theta cycle, on during the query,
    that tells which cycle the queried item was in (so the question stays answerable). A list of `items` symbols (drawn with repetition, so
    the content cannot tell the position) is shown one per gamma slot: item n is a one-step pulse at t = n * slot.
    A theta rhythm runs from the list onset, one cycle per list: theta_t = 2 pi t / (items * slot), and keeps running
    through the delay. Then, for `tail` read-out steps, the theta phase of slot n is replayed (theta = 2 pi n / items)
    together with a query flag; the label is item n.
    Returns real x (T, B, n_symbols + 1): symbol channels and the query flag; theta (T, B); labels (B,);
    queried positions (B,). The theta phase is given to the networks as input (and to theta-tagged fast weights)."""
    g = generator
    cyc = cycle_slots or items
    n_cycles = -(-items // cyc)
    T = items * slot + delay + tail
    x = torch.zeros(T, batch, n_symbols + 1 + (n_cycles if cycle_cue else 0))
    seq = torch.randint(0, n_symbols, (batch, items), generator=g)
    b = torch.arange(batch)
    for n in range(items):
        x[n * slot, b, seq[:, n]] = 1.0
    pos = torch.randint(0, items, (batch,), generator=g)
    theta = (2 * math.pi * torch.arange(T, dtype=torch.float32) / (cyc * slot)).view(T, 1).repeat(1, batch)
    theta[T - tail:] = (2 * math.pi * (pos % cyc).float() / cyc).view(1, batch)
    x[T - tail:, b, n_symbols] = 1.0
    if cycle_cue:
        x[T - tail:, b, n_symbols + 1 + pos // cyc] = 1.0
    return x, theta, seq[b, pos], pos


def input_transform(x: torch.Tensor, kind: str) -> torch.Tensor:
    """'complex' (unchanged) | 'magnitude' (|x|, phase removed) | 'reim' (real tensor [Re, Im]) |
    'real' (real tensor Re x only: a raw real oscillation, as from a microphone; I40) |
    'complex_real' (Re x as a complex tensor with zero imaginary part, for complex-weight models; I40)."""
    if kind == "complex":
        return x
    if kind == "real":
        return x.real
    if kind == "complex_real":
        return torch.complex(x.real, torch.zeros_like(x.real))
    if kind == "magnitude":
        return x.abs()
    if kind == "reim":
        return torch.cat([x.real, x.imag], dim=-1)
    raise ValueError(f"unknown input transform {kind!r}")


def transformed_features(features: int, kind: str) -> int:
    return features * 2 if kind == "reim" else features
