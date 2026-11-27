from pathlib import Path

import pytest
import torch

from cvsnn.data.synthetic import input_transform, spectral_batch, transformed_features
from cvsnn.hardware import DeviceConfig, estimate_cost, load_energy_table
from cvsnn.hardware.nonideal import layer_scale, program_weight, read_out
from cvsnn.models import ModelConfig, SequenceClassifier

ENERGY = Path(__file__).resolve().parents[1] / "configs" / "energy" / "default.yaml"


def test_ideal_device_is_exact():
    torch.manual_seed(0)
    m = SequenceClassifier(16, 2, ModelConfig(kind="cvsnn"))
    x = spectral_batch(torch.tensor([0, 1, 1]), generator=torch.Generator().manual_seed(1))
    a, _ = m(x)
    m.set_hardware(DeviceConfig())
    b, _ = m(x)
    assert torch.equal(a, b)


def test_quantisation_gives_at_most_2_pow_bits_levels_per_device():
    w = torch.randn(64, 64)
    s = layer_scale(w)
    q = program_weight(w, DeviceConfig(weight_bits=3), None, s)
    devices = torch.cat([torch.relu(q) * s, torch.relu(-q) * s]).flatten()
    assert torch.unique(torch.round(devices * 1e6)).numel() <= 2**3
    assert (q - w).abs().max() <= 0.5 / (2**3 - 1) / s + 1e-6


def test_quantisation_passes_gradient_straight_through():
    w = torch.randn(8, 8, requires_grad=True)
    program_weight(w, DeviceConfig(weight_bits=2), None, layer_scale(w)).sum().backward()
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_programming_noise_has_expected_std_and_is_reproducible_per_chip():
    w = torch.full((200, 200), 0.5)
    s = layer_scale(w)  # 2.0, so G+ = 1 and noise in weight units = sqrt(prog_noise^2) / s
    cfg = DeviceConfig(prog_noise=0.05)
    q1 = program_weight(w, cfg, torch.Generator().manual_seed(3), s)
    q2 = program_weight(w, cfg, torch.Generator().manual_seed(3), s)
    assert torch.equal(q1, q2)
    # G- = 0 + noise clipped at 0 (half-normal), G+ = 1 + noise: check the G+ part dominates the spread
    assert 0.05 / 2.0 * 0.8 < (q1 - w).std().item() < 0.05 / 2.0 * 1.4


def test_adc_quantises_and_clips_to_full_scale():
    y = torch.linspace(-5, 5, 101)
    out = read_out(y, torch.ones(4), torch.tensor(1.0), DeviceConfig(adc_bits=2, adc_range=2.0), threshold=1.0, generator=None)
    assert out.max() <= 2.0 + 1e-6 and out.min() >= -2.0 - 1e-6
    assert torch.unique(out).numel() <= 4


def _trace(kind, inp, hidden=(8,)):
    torch.manual_seed(0)
    x = input_transform(spectral_batch(torch.tensor([0, 1]), generator=torch.Generator().manual_seed(2)), inp)
    m = SequenceClassifier(transformed_features(16, inp), 2, ModelConfig(kind=kind, hidden=list(hidden)))
    _, info = m(x, trace=True)
    return info["layers"]


def test_trace_records_activity():
    tr = _trace("cvsnn", "complex")
    assert [L["n_in"] for L in tr] == [16, 8] and [L["n_out"] for L in tr] == [8, 2]
    assert tr[0]["input_kind"] == "complex" and tr[1]["input_kind"] == "complex"
    assert tr[0]["in_events"] == 30 * 16  # dense encoded input
    assert 0 <= tr[1]["in_events"] <= 30 * 8  # spikes of layer 0
    lif = _trace("lif", "reim")
    assert lif[0]["input_kind"] == "real" and lif[1]["input_kind"] == "binary"


def test_cost_models_basic_properties():
    table = load_energy_table(ENERGY)
    tr = _trace("cvsnn", "complex")
    k = estimate_cost(tr, table, karatsuba=True)
    s = estimate_cost(tr, table, karatsuba=False)
    assert k["digital_dense"]["mults"] < s["digital_dense"]["mults"]  # Karatsuba saves multipliers
    assert k["analog_crossbar"]["devices"] == 4 * (16 * 8 + 8 * 2)
    silent = [dict(L, in_events=0) for L in tr]
    e = estimate_cost(silent, table)
    assert e["digital_event"]["weight_reads"] == 0  # no events, no synaptic work
    assert e["digital_dense"]["energy_pj"] == estimate_cost(tr, table)["digital_dense"]["energy_pj"]  # dense ignores activity
    # binary spikes are cheaper per event than phasor spikes on event-driven hardware
    lif = estimate_cost(_trace("lif", "reim"), table)
    assert lif["digital_event"]["mults"] < k["digital_event"]["mults"]


def test_noise_aware_forward_backward_finite():
    torch.manual_seed(0)
    m = SequenceClassifier(16, 2, ModelConfig(kind="cvsnn", hidden=[8]))
    m.set_hardware(DeviceConfig(weight_bits=4, prog_noise=0.05, read_noise=0.01, adc_bits=6))
    y = torch.tensor([0, 1])
    logits, _ = m(spectral_batch(y, generator=torch.Generator().manual_seed(0)))
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_phase_bins_quantise_spike_phase_and_keep_gradient():
    from cvsnn.layers import CLIF
    import math
    torch.manual_seed(0)
    n = CLIF(6, threshold=0.5, phase_bins=8)
    x = (torch.randn(5, 3, 6, dtype=torch.complex64)).requires_grad_()
    s, _, h = n(x)
    fired = h > 0
    ang = torch.angle(s[fired])
    k = ang / (2 * math.pi / 8)
    assert torch.allclose(k, torch.round(k), atol=1e-4)  # multiples of 2pi/8
    assert torch.allclose(s[fired].abs(), torch.ones_like(ang), atol=1e-5)
    s.abs().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_timing_code_removes_synaptic_multiplies():
    table = load_energy_table(ENERGY)
    torch.manual_seed(0)
    x = spectral_batch(torch.tensor([0, 1]), generator=torch.Generator().manual_seed(2))
    exact = SequenceClassifier(16, 2, ModelConfig(kind="cvsnn", hidden=[8]))
    timed = SequenceClassifier(16, 2, ModelConfig(kind="cvsnn", hidden=[8], phase_bins=4))
    timed.load_state_dict(exact.state_dict())
    _, ie = exact(x, trace=True)
    _, it = timed(x, trace=True)
    assert it["layers"][1]["input_kind"] == "phase_binned"
    # same spikes, but the spiking layer's synapses only add; K=4 combine is multiply-free
    ce = estimate_cost(ie["layers"][1:], table)["digital_event"]
    ct = estimate_cost(it["layers"][1:], table)["digital_event"]
    assert ct["mults"] < ce["mults"] and ct["energy_pj"] < ce["energy_pj"]


def test_encoder_front_end_has_fanout_one_and_no_synaptic_cost():
    table = load_energy_table(ENERGY)
    torch.manual_seed(0)
    m = SequenceClassifier(16, 8, ModelConfig(kind="cvsnn", hidden=[16], encoder=True, phase_bins=4))
    x = spectral_batch(torch.tensor([0, 1]), generator=torch.Generator().manual_seed(2))
    logits, info = m(x, trace=True)
    tr = info["layers"]
    assert logits.shape == (2, 8) and tr[0]["encoder"] and not tr[1]["encoder"]
    assert tr[1]["input_kind"] == "phase_binned" and tr[1]["n_in"] == 16
    enc_only = estimate_cost(tr[:1], table)
    assert enc_only["digital_event"]["weight_reads"] == 0 and enc_only["analog_crossbar"]["devices"] == 0
    assert m.real_mults_per_step() == 16 * 16 * 3 + 16 * 8 * 3  # encoder adds no synaptic multiplies


def test_k1_spikes_carry_no_phase():
    from cvsnn.layers import CLIF
    n = CLIF(4, threshold=0.3, phase_bins=1)
    s, _, h = n(torch.randn(6, 2, 4, dtype=torch.complex64))
    fired = h > 0
    assert fired.any() and torch.allclose(s[fired], torch.ones_like(s[fired]))
