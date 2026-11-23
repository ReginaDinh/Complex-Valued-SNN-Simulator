"""Platform cost models driven by measured activity (spikes/events) of a forward pass.

The model records, per layer, how many inputs were active (events) and what kind they were.
Each platform turns that trace into operation counts and an energy estimate:

* digital_event   : event-driven digital chip (Loihi-like). Work happens per input event:
                    binary spike -> accumulate (1 add) per synapse; real-valued input -> MAC;
                    complex input / phasor spike -> complex MAC (Karatsuba 3 mul or 4 mul) per synapse.
                    Plus one weight fetch per synaptic op and a neuron update per neuron per step.
* digital_dense   : synchronous digital accelerator (FPGA/ASIC): every synapse is computed every step.
* phase_binned    : timing-coded cLIF spikes (CLIF(phase_bins=K)): a binary event in one of K slots.
                    Digital: each event adds (W_re, W_im) into the receiver's slot accumulator (2 adds,
                    no multiply); once per step the receiver combines its K accumulators with
                    e^{-j2pi k/K}: rotations by multiples of 90 deg are free (sign/swap), the others cost a
                    complex multiply by a constant. The sender only picks a slot (no sqrt/div).
                    Analog: 1-bit row pulses; conservative read-out with one ADC pass per slot.
* analog_crossbar : in-memory computing. Real layer: 1 read phase, 2 devices per weight, 1 ADC/output.
                    Complex layer: 2 read phases (drive Re then Im of the input) over the 4 devices
                    G_re+-, G_im+-; 2 ADC conversions per output per phase. Binary spikes drive rows
                    with 1-bit pulses; phasor/real-valued inputs need a multi-bit DAC per row.
                    Neuron updates are digital.

Key point the numbers expose: a cLIF phasor spike is *not* a binary event, so on digital hardware each
event costs a complex multiply instead of an add; on an analog crossbar it costs a DAC drive instead
of a 1-bit pulse. Whether CV-SNN saves energy depends on this, not on the model alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PLATFORMS = ("digital_event", "digital_dense", "analog_crossbar")

# Input kinds recorded in the layer trace
BINARY, REAL, COMPLEX, PHASE_BINNED = "binary", "real", "complex", "phase_binned"


@dataclass
class EnergyTable:
    add: float
    mul: float
    weight_read: float      # per weight fetch (scaled from a 64-bit SRAM word to the weight width)
    sqrt_div_mults: float
    device_read: float
    row_pulse: float
    row_dac: float
    adc: float
    precision: str


def load_energy_table(path: str | Path) -> EnergyTable:
    d = yaml.safe_load(Path(path).read_text())
    p = d["precision"]
    bits = {"int8": 8, "int16": 16, "int32": 32, "fp16": 16, "fp32": 32}[p]
    dig, ana = d["digital"], d["analog"]
    return EnergyTable(
        add=dig[p]["add"], mul=dig[p]["mul"], weight_read=dig["sram_read_64bit"] * bits / 64,
        sqrt_div_mults=dig["sqrt_div_as_mults"], device_read=ana["device_read"],
        row_pulse=ana["row_pulse_1bit"], row_dac=ana["row_dac"], adc=ana["adc_conversion"], precision=p,
    )


def _synapse_ops(kind: str, karatsuba: bool, complex_weights: bool = False) -> tuple[int, int, int]:
    """(mults, adds, weight_reads) for one synapse processing one input value. A binary or real input on a
    complex weight only touches (W_re, W_im): 2 adds (binary) or 2 multiply-adds (real)."""
    c = 2 if complex_weights else 1
    if kind == BINARY:
        return 0, c, c
    if kind == REAL:
        return c, c, c
    if kind == PHASE_BINNED:
        return 0, 2, 2
    # complex weight x complex input, accumulated into (re, im); Karatsuba with (wr + wi) precomputed
    return (3, 5, 2) if karatsuba else (4, 4, 2)


def _slot_combine_ops(k: int | None, karatsuba: bool) -> tuple[float, float]:
    """(mults, adds) per receiving neuron per step to combine K slot accumulators."""
    if not k:
        return 0, 0
    cm, ca = (3, 5) if karatsuba else (4, 2)
    free = 4 if k % 4 == 0 else (2 if k % 2 == 0 else 1)  # rotations by multiples of 90 deg are sign/swap
    nontrivial = max(0, k - free)
    return nontrivial * cm, nontrivial * ca + 2 * (k - 1)


def _neuron_ops(neuron: str, karatsuba: bool, sqrt_div_mults: float, binned_out: bool = False,
                binary_out: bool = False) -> tuple[float, float]:
    """(mults, adds) for one neuron update in one time step."""
    if neuron == "lif":
        return 1, 3                    # leak mult, add input, compare, reset subtract
    if neuron == "clif":
        cm, ca = (3, 5) if karatsuba else (4, 2)
        # beta*u (complex mult), + input (2 add), |u|^2 (2 mul 1 add), compare (1), soft reset 2 mul 2 add.
        # unit phasor for the emitted spike: sqrt + 2 div; a timing-coded spike only needs a slot
        # (a few comparisons) instead, a binary spike nothing beyond the comparison. Spike costs are counted
        # every step (upper bound); the output layer is costed as a phasor emitter (as in E-HW).
        emit_m, emit_a = (0, 3) if binned_out else (0, 0) if binary_out else (3 * sqrt_div_mults, 0)
        return cm + 2 + 2 + emit_m, ca + 2 + 1 + 1 + 2 + emit_a
    if neuron in ("prf", "csilif"):
        cm, ca = (3, 5) if karatsuba else (4, 2)
        # complex state update (complex mult by a constant), real-to-complex input scaling (2 mul),
        # add input (2 add), compare Re(u) with the threshold (1 add); binary spike, no reset
        return cm + 2, ca + 2 + 1
    if neuron == "cvnn":
        cm, ca = (3, 5) if karatsuba else (4, 2)
        return cm + 2 + 3 * sqrt_div_mults, ca + 2 + 1 + 1    # leak, |u|, modReLU scale
    raise ValueError(neuron)


def estimate_cost(trace: list[dict], table: EnergyTable, karatsuba: bool = True) -> dict[str, dict]:
    """trace: per layer {n_in, n_out, T, input_kind, neuron, in_events, complex_weights, encoder}
    (in_events = active inputs per sample summed over time). Returns per-platform counts and energy (pJ).
    Encoder layers (spiking sensor front end, fan-out 1, fixed gain) cost only their neuron updates."""
    out: dict[str, dict] = {}
    syn_layers = [L for L in trace if not L.get("encoder")]

    def neuron_cost(i: int) -> tuple[float, float]:
        L = trace[i]
        binned_out = i + 1 < len(trace) and trace[i + 1]["input_kind"] == PHASE_BINNED
        binary_out = i + 1 < len(trace) and trace[i + 1]["input_kind"] == BINARY
        nm, na = _neuron_ops(L["neuron"], karatsuba, table.sqrt_div_mults, binned_out, binary_out)
        cm, ca = _slot_combine_ops(L.get("phase_bins") if L["input_kind"] == PHASE_BINNED else None, karatsuba)
        return nm + cm, na + ca

    # digital, event-driven
    mul = add = reads = 0.0
    for i, L in enumerate(trace):
        m, a, r = _synapse_ops(L["input_kind"], karatsuba, L["complex_weights"])
        syn = 0 if L.get("encoder") else L["in_events"] * L["n_out"]
        nm, na = neuron_cost(i)
        upd = L["T"] * L["n_out"]
        mul += syn * m + upd * nm
        add += syn * a + upd * na
        reads += syn * r
    out["digital_event"] = {"mults": mul, "adds": add, "weight_reads": reads,
                            "energy_pj": mul * table.mul + add * table.add + reads * table.weight_read}

    # digital, dense synchronous
    mul = add = reads = 0.0
    for i, L in enumerate(trace):
        kind = L["input_kind"] if L["input_kind"] != BINARY else REAL  # dense hardware multiplies anyway
        if L["input_kind"] == PHASE_BINNED:
            kind = PHASE_BINNED  # slot accumulation is still add-only, but done for every synapse every step
        m, a, r = _synapse_ops(kind, karatsuba, L["complex_weights"])
        syn = 0 if L.get("encoder") else L["T"] * L["n_in"] * L["n_out"]
        nm, na = neuron_cost(i)
        upd = L["T"] * L["n_out"]
        mul += syn * m + upd * nm
        add += syn * a + upd * na
        reads += syn * r
    out["digital_dense"] = {"mults": mul, "adds": add, "weight_reads": reads,
                            "energy_pj": mul * table.mul + add * table.add + reads * table.weight_read}

    # analog crossbar (synapses) + digital neurons
    dev = rows = adc = 0.0
    n_mul = n_add = 0.0
    devices = 0
    for i, L in enumerate(trace):
        if L.get("encoder"):
            nm, na = neuron_cost(i)
            n_mul += L["T"] * L["n_out"] * nm
            n_add += L["T"] * L["n_out"] * na
            continue
        binned = L["input_kind"] == PHASE_BINNED
        # complex input on complex weights: drive Re then Im of the input (2 phases); a real or binary input
        # needs one phase (its columns read W_re and W_im); timing code: each event pulses once in its slot
        phases = 2 if (L["complex_weights"] and L["input_kind"] == COMPLEX) else 1
        dev_per_weight_per_phase = 4 if L["complex_weights"] else 2
        devices += L["n_in"] * L["n_out"] * dev_per_weight_per_phase
        active = L["in_events"]                        # row drives per phase (summed over time)
        rows += active * phases
        dev += active * phases * L["n_out"] * dev_per_weight_per_phase
        reads_per_step = L["phase_bins"] if binned else phases      # one ADC pass per slot (conservative)
        adc += L["T"] * L["n_out"] * reads_per_step * (2 if L["complex_weights"] else 1)
        nm, na = neuron_cost(i)
        n_mul += L["T"] * L["n_out"] * nm
        n_add += L["T"] * L["n_out"] * na
    row_e = sum(
        L["in_events"] * (table.row_pulse if L["input_kind"] in (BINARY, PHASE_BINNED)
                          else 2 * table.row_dac if (L["complex_weights"] and L["input_kind"] == COMPLEX) else table.row_dac)
        for L in syn_layers
    )
    out["analog_crossbar"] = {"devices": devices, "row_drives": rows, "device_reads": dev, "adc_conversions": adc,
                              "neuron_mults": n_mul, "neuron_adds": n_add,
                              "energy_pj": dev * table.device_read + row_e + adc * table.adc + n_mul * table.mul + n_add * table.add}
    return out
