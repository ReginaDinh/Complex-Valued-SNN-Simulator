"""H6: frequency sets, binary cLIF spikes, frozen frequencies, spike-rate penalty, cost of binary events."""

import math

import numpy as np
import pytest
import torch

from analysis.freq_metrics import combos
from cvsnn.frequencies import FREQ_SETS, additive_energy, frequency_set, grid_range, sidon_grid
from cvsnn.hardware import estimate_cost, load_energy_table
from cvsnn.layers import CLIF, LIF
from cvsnn.models import ModelConfig, SequenceClassifier, rate_penalty

LO, HI = 0.01, 1.5


@pytest.mark.parametrize("kind", [k for k in FREQ_SETS if k != "zero"])
def test_frequency_sets_lie_in_range(kind):
    w = frequency_set(kind, 32, LO, HI, bits=10, generator=torch.Generator().manual_seed(0))
    assert w.shape == (32,) and w.min() >= LO - 1e-12 and w.max() <= HI + 1e-12
    assert frequency_set("zero", 4, LO, HI).abs().sum() == 0


def test_golden_set_has_exact_order4_relation_and_independent_set_has_none():
    g = frequency_set("golden", 5, LO, HI).numpy()
    assert abs(g[0] - g[1] - g[3] + g[4]) < 1e-12        # omega_1 - omega_2 - omega_4 + omega_5 = 0
    ind = frequency_set("independent", 12, LO, HI).numpy()
    for M in (2, 3, 4):
        d = np.abs(combos(ind, M))
        d = np.abs(d - 2 * math.pi * np.round(d / (2 * math.pi)))
        assert d.min() > 1e-9


def test_harmonic_set_is_rationally_dependent():
    h = frequency_set("harmonic", 3, LO, HI).numpy()
    assert abs(h[0] - 2 * h[1] + h[2]) < 1e-12


def test_quantized_frequencies_lie_on_the_grid_and_can_merge():
    w = frequency_set("quantized", 32, LO, HI, bits=8, generator=torch.Generator().manual_seed(0))
    m = w * 2**8 / (2 * math.pi)
    assert torch.allclose(m, torch.round(m), atol=1e-9)
    assert len(torch.unique(m)) < 32          # 8 bits: log-uniform frequencies collide


def test_additive_energy_and_sidon_search():
    assert additive_energy(np.array([0, 1, 3, 7])) == 4 * 7    # Golomb ruler: the minimum N(2N - 1)
    assert additive_energy(np.array([0, 1, 2, 3])) > 4 * 7
    m_lo, m_hi = grid_range(LO, HI, 10)
    s = np.array(sidon_grid(32, m_lo, m_hi))
    assert len(set(s.tolist())) == 32 and s.min() >= m_lo and s.max() <= m_hi
    rng = np.random.default_rng(1)
    rand = [additive_energy(rng.choice(np.arange(m_lo, m_hi + 1), 32, replace=False)) for _ in range(5)]
    assert additive_energy(s) < min(rand)


def test_binary_spikes_are_real_events_with_surrogate_gradient():
    torch.manual_seed(0)
    n = CLIF(6, threshold=0.5, reset="none", spike_mode="binary", leak_mag_range=(0.9, 0.99), leak_phase_range=(0.1, 1.0)).double()
    x = (0.4 * torch.randn(20, 3, 6, dtype=torch.complex128)).requires_grad_()
    out, pot, h = n(x)
    assert not torch.is_complex(out) and torch.equal(out, h)
    assert set(out.unique().tolist()) <= {0.0, 1.0} and out.sum() > 0
    assert torch.equal(out.detach(), (pot.abs() >= 0.5).double())
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_binary_on_real_part_and_soft_reset_direction():
    n = CLIF(1, threshold=1.0, leak_mag=0.9, leak_phase=0.0, learn_leak=False, reset="soft", spike_mode="binary", binary_on="re").double()
    x = torch.zeros(3, 1, 1, dtype=torch.complex128)
    x[0, 0, 0] = complex(0.5, 2.0)        # |U| > V_th but Re U < V_th: no spike
    x[1, 0, 0] = complex(1.0, 0.0)
    out, pot, h = n(x)
    assert h[0, 0, 0] == 0 and h[1, 0, 0] == 1
    beta = torch.sigmoid(n.leak_mag_raw)[0].item()
    expected = (pot[1, 0, 0] - 1.0) * beta    # soft reset subtracts V_th along the real axis
    assert pot[2, 0, 0].real.item() == pytest.approx(expected.real.item()) and pot[2, 0, 0].imag.item() == pytest.approx(expected.imag.item())
    with pytest.raises(ValueError):
        CLIF(2, spike_mode="phasor", binary_on="re")


def test_frozen_frequencies_keep_the_set_while_magnitude_learns():
    n = CLIF(8, leak_mag_range=(0.9, 0.99), leak_phase_range=(LO, HI), phase_dist="golden", learn_phase=False)
    names = dict(n.named_parameters())
    assert "leak_mag_raw" in names and "leak_phase" not in names
    assert torch.allclose(n.leak_phase.double(), frequency_set("golden", 8, LO, HI), atol=1e-6)


def test_lif_leak_range_is_heterogeneous():
    torch.manual_seed(0)
    n = LIF(50, leak_range=(0.9, 0.999))
    b = torch.sigmoid(n.leak_raw)
    assert b.min() >= 0.9 - 1e-6 and b.max() <= 0.999 + 1e-6 and b.std() > 0.01


def test_rate_penalty_is_zero_below_target_and_reduces_rate():
    torch.manual_seed(0)
    cfg = ModelConfig(kind="cvsnn", hidden=[16], reset="none", spike_mode="binary", threshold=0.3,
                      leak_mag_range=(0.9, 0.99), leak_phase_range=(LO, HI), phase_dist="log")
    m = SequenceClassifier(8, 4, cfg)
    x = torch.complex(torch.randn(30, 16, 8), torch.randn(30, 16, 8))
    _, info = m(x)
    r = info["transmit_rate"].item()
    assert rate_penalty(info, r + 0.1, 5.0) == 0
    assert rate_penalty(info, None, 5.0) == 0
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    for _ in range(30):
        _, info = m(x)
        loss = rate_penalty(info, 0.0, 5.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert m(x)[1]["transmit_rate"].item() < r


def test_binary_cvsnn_trace_and_cost_have_no_synaptic_multiplies():
    torch.manual_seed(0)
    table = load_energy_table("configs/energy/default.yaml")
    x = torch.complex(torch.randn(20, 8, 6), torch.randn(20, 8, 6))
    base = dict(kind="cvsnn", hidden=[10], reset="none", threshold=0.5)
    mb = SequenceClassifier(6, 4, ModelConfig(**base, spike_mode="binary"))
    _, ib = mb(x, trace=True)
    L = ib["layers"][1]
    assert L["input_kind"] == "binary" and L["complex_weights"]
    cost, silent = estimate_cost([L], table)["digital_event"], estimate_cost([dict(L, in_events=0)], table)["digital_event"]
    assert cost["mults"] == silent["mults"]                      # synapses: no multiplies
    assert cost["adds"] - silent["adds"] == 2 * L["in_events"] * L["n_out"]  # W_re and W_im accumulated
    # one read phase (real input on complex weights) instead of two
    mp = SequenceClassifier(6, 4, ModelConfig(**base))
    _, ip = mp(x, trace=True)
    lp = dict(ip["layers"][1], in_events=L["in_events"])
    assert estimate_cost([L], table)["analog_crossbar"]["row_drives"] * 2 == estimate_cost([lp], table)["analog_crossbar"]["row_drives"]


@pytest.mark.parametrize("kind,inp", [("cvsnn", "complex"), ("prf", "reim"), ("cslif", "reim"), ("lif", "reim")])
def test_spiking_encoders(kind, inp):
    torch.manual_seed(0)
    extra = {"spike_mode": "binary", "reset": "none"} if kind == "cvsnn" else {}
    m = SequenceClassifier(12 if inp == "reim" else 6, 4, ModelConfig(kind=kind, hidden=[8], encoder=True, **extra))
    x = torch.complex(torch.randn(15, 4, 6), torch.randn(15, 4, 6))
    x = torch.cat([x.real, x.imag], -1) if inp == "reim" else x
    logits, info = m(x, trace=True)
    assert logits.shape == (4, 4) and torch.isfinite(logits).all()
    assert info["layers"][0]["encoder"] and info["layers"][1]["input_kind"] == "binary"


def test_split_layer_memory_neurons_never_reset_and_keep_their_init():
    torch.manual_seed(0)
    n = CLIF(4, threshold=0.5, leak_mag=0.85, leak_phase=0.15, reset="soft", mem_frac=0.5,
             leak_mag_range=(0.99, 0.999), leak_phase_range=(0.01, 1.5), phase_dist="log").double()
    b = torch.sigmoid(n.leak_mag_raw)
    assert torch.allclose(b[:2], torch.tensor(0.85, dtype=b.dtype), atol=1e-4) and (b[2:] >= 0.99 - 1e-6).all()
    assert n.reset_mask.tolist() == [1.0, 1.0, 0.0, 0.0]
    x = torch.zeros(5, 1, 4, dtype=torch.complex128)
    x[0] = 2.0
    s, pot, h = n(x)
    assert h[0].sum() == 4                                        # every neuron fires on the pulse
    beta = n.beta.detach()
    # processing neurons were reduced by the soft reset, memory neurons kept their full state
    assert torch.allclose(pot[1, 0, 2:], beta[2:] * 2.0) and (pot[1, 0, :2].abs() < (beta[:2] * 2.0).abs()).all()
    with pytest.raises(ValueError):
        CLIF(4, reset="none", mem_frac=0.5, leak_mag_range=(0.9, 0.99), leak_phase_range=(0.1, 1.0))


def test_real_synapse_cvsnn_has_real_weights_and_a_rotating_complex_state():
    torch.manual_seed(0)
    cfg = ModelConfig(kind="cvsnn", hidden=[8], spike_mode="binary", binary_on="re", real_synapses=True, reset="none",
                      leak_mag_range=(0.9, 0.99), leak_phase_range=(0.1, 1.0))
    m = SequenceClassifier(12, 4, cfg)
    assert all(isinstance(l, torch.nn.Linear) for l in m.linears)
    x = torch.randn(20, 3, 12)
    logits, info = m(x, trace=True)
    assert torch.isfinite(logits).all() and not info["layers"][1]["complex_weights"] and info["layers"][1]["input_kind"] == "binary"
    _, pot, _ = m.neurons[0](m.linears[0](x))
    assert torch.is_complex(pot) and pot.imag.abs().sum() > 0        # real current, complex (rotating) state
    with pytest.raises(ValueError):
        m(torch.complex(x, x))
    with pytest.raises(ValueError):
        SequenceClassifier(12, 4, ModelConfig(kind="cvsnn", real_synapses=True))  # phasor spikes need complex weights


@pytest.mark.parametrize("kw", [dict(spike_mode="phasor"), dict(spike_mode="binary", binary_on="re"), dict(phase_bins=4),
                                dict(mem_frac=0.5, leak_mag_range=(0.99, 0.999), leak_phase_range=(0.01, 1.5))])
def test_parallel_soft_reset_matches_the_loop_in_values_and_gradients(kw):
    torch.manual_seed(0)
    a = CLIF(5, threshold=0.6, leak_mag=0.9, leak_phase=0.3, reset="soft", **kw).double()
    b = CLIF(5, threshold=0.6, leak_mag=0.9, leak_phase=0.3, reset="soft", parallel_reset=True, **kw).double()
    b.load_state_dict(a.state_dict())
    x = (0.5 * torch.randn(120, 3, 5, dtype=torch.complex128)).requires_grad_()
    outs = []
    for n in (a, b):
        s, pot, h = n(x)
        loss = (pot.abs() ** 2).sum() + (s.abs() if torch.is_complex(s) else s).sum() * 0.3
        grads = torch.autograd.grad(loss, [x, *n.parameters()])
        outs.append((s.detach(), pot.detach(), h.detach(), grads))
    (s1, p1, h1, g1), (s2, p2, h2, g2) = outs
    assert torch.equal(h1, h2) and h1.sum() > 10
    assert torch.allclose(p1, p2, atol=1e-9) and torch.allclose(s1, s2, atol=1e-9)
    for u, v in zip(g1, g2):
        assert torch.allclose(u, v, atol=1e-7, rtol=1e-6)


def test_parallel_soft_reset_lif_matches_the_loop():
    torch.manual_seed(1)
    a = LIF(6, threshold=0.7, leak=0.9, reset="soft").double()
    b = LIF(6, threshold=0.7, leak=0.9, reset="soft", parallel_reset=True).double()
    b.load_state_dict(a.state_dict())
    x = (0.6 * torch.randn(150, 2, 6, dtype=torch.float64)).requires_grad_()
    res = []
    for n in (a, b):
        s, pot, _ = n(x)
        loss = (pot ** 2).sum() + s.sum()
        res.append((s.detach(), pot.detach(), torch.autograd.grad(loss, [x, *n.parameters()])))
    assert torch.equal(res[0][0], res[1][0]) and torch.allclose(res[0][1], res[1][1], atol=1e-9)
    for u, v in zip(res[0][2], res[1][2]):
        assert torch.allclose(u, v, atol=1e-7, rtol=1e-6)


def test_out_pairs_give_each_class_a_processing_and_a_memory_readout_neuron():
    torch.manual_seed(0)
    cfg = ModelConfig(kind="cvsnn", hidden=[8], reset="soft", mem_frac=0.5, out_pairs=True, spike_mode="binary",
                      binary_on="re", real_synapses=True, readout="re", leak_mag_range=(0.99, 0.999), leak_phase_range=(0.01, 1.5))
    m = SequenceClassifier(6, 4, cfg)
    last = m.neurons[-1]
    assert last.num_features == 8 and last.reset_mask.tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    logits, _ = m(torch.randn(10, 3, 6))
    assert logits.shape == (3, 4)
    with pytest.raises(ValueError):
        SequenceClassifier(6, 4, ModelConfig(kind="cvsnn", out_pairs=True))


def test_slot_lines_carry_the_timing_code_as_real_events():
    torch.manual_seed(0)
    n = CLIF(3, threshold=0.6, reset="soft", spike_mode="slots", phase_bins=4).double()
    x = 0.8 * torch.polar(torch.ones(12, 2, 3, dtype=torch.float64), torch.rand(12, 2, 3, dtype=torch.float64) * 6.28)
    out, pot, h = n(x)
    assert out.shape == (12, 2, 12) and not torch.is_complex(out)
    lines = out.view(12, 2, 3, 4)
    assert torch.equal(lines.sum(-1), h)                         # one line per spike
    k = torch.remainder(torch.round(torch.angle(pot) * 4 / (2 * math.pi)), 4).long()
    on = lines.argmax(-1)
    assert torch.equal(on[h > 0], k[h > 0])                      # on the line of its phase slot
    with pytest.raises(ValueError):
        CLIF(3, spike_mode="slots")


@pytest.mark.parametrize("front", ["binre", "k4"])
def test_complex_front_end_feeds_a_real_lif_core(front):
    torch.manual_seed(0)
    cfg = ModelConfig(kind="lif", hidden=[10], encoder=True, front=front, reset="none", leak_mag_range=(0.9, 0.99))
    m = SequenceClassifier(6, 4, cfg)
    assert isinstance(m.neurons[0], CLIF) and all(isinstance(n, LIF) for n in m.neurons[1:])
    assert m.linears[1].in_features == (24 if front == "k4" else 6)
    x = torch.complex(torch.randn(15, 3, 6), torch.randn(15, 3, 6))
    logits, info = m(x, trace=True)
    assert torch.isfinite(logits).all()
    L0, L1 = info["layers"][:2]
    assert L0["neuron"] == "clif" and L0["encoder"] and L1["input_kind"] == "binary" and L1["n_in"] == (24 if front == "k4" else 6)
    fixed = SequenceClassifier(6, 4, ModelConfig(**{**cfg.__dict__, "front_fixed": True}))
    assert not list(fixed.neurons[0].parameters())
    with pytest.raises(ValueError):
        SequenceClassifier(6, 4, ModelConfig(kind="cvsnn", encoder=True, front="k4"))


def test_two_stage_front_end_has_resonators_before_the_lif_core():
    torch.manual_seed(0)
    cfg = ModelConfig(kind="lif", hidden=[7], encoder=True, front="binre", front_hidden=[9], reset="soft")
    m = SequenceClassifier(6, 4, cfg)
    kinds = [type(n).__name__ for n in m.neurons]
    assert kinds == ["CLIF", "CLIF", "LIF", "LIF"]
    assert m.neurons[1].spike_mode == "binary" and m.neurons[1].binary_on == "re" and m.neurons[1].reset == "soft"
    assert isinstance(m.linears[1], torch.nn.Linear) and m.linears[1].in_features == 6 and m.linears[2].in_features == 9
    logits, info = m(torch.complex(torch.randn(12, 3, 6), torch.randn(12, 3, 6)), trace=True)
    assert torch.isfinite(logits).all() and [L["neuron"] for L in info["layers"]] == ["clif", "clif", "lif", "lif"]
    with pytest.raises(ValueError):
        SequenceClassifier(6, 4, ModelConfig(kind="lif", front_hidden=[4]))


def test_every_front_variant_gets_the_complex_sensor_signal_and_matched_size():
    from experiments import eh6
    from cvsnn.data.synthetic import transformed_features
    for j in eh6.transmit_jobs(1, False, front=True):
        if j["task"] != "fast" or j["seed"] or j["hp"] != {"encoder_gain": 1.0, "threshold": 0.5}:
            continue
        assert j["input"] == "complex", j["variant"]
        m = SequenceClassifier(transformed_features(16, j["input"]), 8, ModelConfig(**j["spec"]["model"]))
        assert 800 <= m.num_params() <= 860, (j["variant"], m.num_params())


def test_freeze_after_first_task_freezes_the_whole_periphery():
    from experiments.ecl import freeze_periphery
    m = SequenceClassifier(16, 8, ModelConfig(kind="lif", hidden=[5], encoder=True, front="k4", front_hidden=[6]))
    freeze_periphery(m)
    learn = {n for n, p in m.named_parameters() if p.requires_grad}
    assert learn == {"linears.2.weight", "linears.3.weight", "neurons.2.leak_raw", "neurons.3.leak_raw"}  # core + read-out


def test_memory_branch_runs_in_parallel_with_the_core_and_feeds_the_readout():
    torch.manual_seed(0)
    cfg = ModelConfig(kind="lif", hidden=[7], encoder=True, front="k4", front_hidden=[9], reset="soft", memory_branch=5)
    m = SequenceClassifier(6, 4, cfg)
    assert m.mem_neuron.reset == "none" and m.linears[-1].in_features == 12 and m.mem_linear.in_features == 9
    b = torch.sigmoid(m.mem_neuron.leak_raw)
    assert (b >= 0.99 - 1e-6).all()
    logits, info = m(torch.complex(torch.randn(10, 3, 6), torch.randn(10, 3, 6)), trace=True)
    assert torch.isfinite(logits).all() and len(info["layers"]) == 6 and info["layers"][-1]["n_in"] == 12
    logits.sum().backward()
    assert m.mem_linear.weight.grad is not None and m.mem_linear.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        SequenceClassifier(6, 4, ModelConfig(kind="lif", hidden=[7, 3], encoder=True, front="k4", memory_branch=5))


@pytest.mark.parametrize("mem_input,n_in", [("reim", 12), ("real", 6)])
def test_memory_branch_can_read_the_raw_input(mem_input, n_in):
    torch.manual_seed(0)
    cfg = ModelConfig(kind="lif", hidden=[7], encoder=True, front="k4", front_hidden=[9], reset="soft", memory_branch=5,
                      memory_from_input=True, memory_input=mem_input)
    m = SequenceClassifier(6, 4, cfg)
    assert m.mem_linear.in_features == n_in
    x = torch.complex(torch.randn(10, 3, 6), torch.randn(10, 3, 6)) if mem_input == "reim" else torch.rand(10, 3, 6)
    logits, info = m(x, trace=True)
    assert torch.isfinite(logits).all() and info["layers"][3]["n_in"] == n_in and info["layers"][3]["input_kind"] == "real"
    logits.sum().backward()
    assert m.mem_linear.weight.grad.abs().sum() > 0


def test_urgency_schedules_and_effects():
    from cvsnn.layers import Urgency
    u = Urgency("damping", 10, init=0.5)
    assert abs(u.strength().item() - 0.5) < 1e-5
    g_end = u.factors(9)[1].item()
    assert abs(g_end - math.exp(-0.5)) < 1e-5 and abs(u.factors(0)[1].item() - math.exp(-0.5 / 100)) < 1e-5
    uc = Urgency("damping_const", 10, init=0.5)
    assert all(abs(uc.factors(t)[1].item() - math.exp(-0.5)) < 1e-5 for t in (0, 4, 9, 20))
    th = Urgency("threshold", 10, init=0.3)
    assert abs(th.factors(9)[2].item() - 0.7) < 1e-5
    torch.manual_seed(0)
    x = torch.zeros(10, 1, 1, dtype=torch.complex128)
    x[0] = 0.5
    base = CLIF(1, threshold=10.0, leak_mag=0.95, leak_phase=0.2, learn_leak=False, reset="soft").double()
    damp = CLIF(1, threshold=10.0, leak_mag=0.95, leak_phase=0.2, learn_leak=False, reset="soft", urgency="damping", deadline=10).double()
    _, p0, _ = base(x)
    _, p1, _ = damp(x)
    assert p1[-1].abs() < p0[-1].abs()                              # the state settles towards the deadline
    lif = LIF(1, threshold=1.0, leak=0.9, reset="soft", urgency="threshold", deadline=5).double()
    v = torch.full((5, 1, 1), 0.2, dtype=torch.float64)
    s, _, _ = lif(v)
    (s.sum() + 0.0).backward()
    assert lif.urgency.raw.grad is not None
    with pytest.raises(ValueError):
        CLIF(2, reset="none", urgency="gain", deadline=5)
    with pytest.raises(ValueError):
        Urgency("panic", 5)


def test_binding_task_hides_the_label_in_the_cross_modal_phase_only():
    from cvsnn.data.synthetic import binding_batch
    g = torch.Generator().manual_seed(0)
    y = torch.tensor([1, 0] * 200)
    x = binding_batch(y, time_steps=30, freq=3.0, noise=0.0, generator=g)
    assert x.shape == (30, 400, 8)
    active = (x.abs().mean(0) > 0.5)
    assert (active[:, :4].sum(-1) == 2).all() and (active[:, 4:].sum(-1) == 2).all()  # 2 colours, 2 shapes
    # the channel magnitudes carry no label information: the same statistics for both classes
    m1, m0 = x[:, y == 1].abs().mean(), x[:, y == 0].abs().mean()
    assert abs(m1 - m0) < 0.02
    # in a noise-free sample, the lower colour is in phase with the lower shape iff y = 1
    for b in range(6):
        cols = active[b, :4].nonzero().view(-1)
        shps = active[b, 4:].nonzero().view(-1) + 4
        d = torch.angle(x[:, b, cols[0]] * torch.conj(x[:, b, shps[0]])).abs().max()
        assert (d < 1e-4) == bool(y[b])


def test_synaptic_intelligence_modes():
    from cvsnn.consolidation import SynapticIntelligence
    from cvsnn.layers import ComplexLinear
    torch.manual_seed(0)
    lin = ComplexLinear(4, 3)
    model = torch.nn.ModuleList([lin])
    for mode in ("cart", "polar", "mag", "phase"):
        si = SynapticIntelligence(model, mode, c=1.0)
        assert si.penalty() == 0.0
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(3):          # a few steps on a toy loss, then consolidate
            loss = (lin(torch.ones(2, 4, dtype=torch.complex64)).abs() ** 2).sum()
            opt.zero_grad()
            loss.backward()
            si.pre_step()
            opt.step()
            si.post_step()
        si.end_task()
        assert si.penalty() == 0 or float(si.penalty()) < 1e-6        # at the consolidated point
        with torch.no_grad():   # rotate every complex weight by 0.5 rad (phase change only)
            w = torch.complex(lin.weight_re, lin.weight_im) * complex(math.cos(0.5), math.sin(0.5))
            lin.weight_re.copy_(w.real); lin.weight_im.copy_(w.imag)
        rot = float(si.penalty())
        with torch.no_grad():   # undo, then scale the magnitudes by 1.5 (magnitude change only)
            w = w * complex(math.cos(-0.5), math.sin(-0.5)) * 1.5
            lin.weight_re.copy_(w.real); lin.weight_im.copy_(w.imag)
        scl = float(si.penalty())
        with torch.no_grad():
            w = w / 1.5
            lin.weight_re.copy_(w.real); lin.weight_im.copy_(w.imag)
        if mode == "mag":
            assert rot < 1e-6 and scl > 0          # magnitude consolidated, phase free
        if mode == "phase":
            assert scl < 1e-6 and rot > 0          # phase consolidated, magnitude free
        if mode in ("cart", "polar"):
            assert rot > 0 and scl > 0


def test_assoc_recall_task_and_fast_weights_recall_a_stored_value():
    from cvsnn.data.synthetic import assoc_recall_batch
    from cvsnn.layers import FastWeights
    x, y = assoc_recall_batch(16, pairs=3, generator=torch.Generator().manual_seed(0))
    assert x.shape == (85, 16, 17) and y.shape == (16,)
    assert torch.equal(x[:48, :, :8].sum((0, 2)), torch.full((16,), 3.0))      # 3 keys written
    assert torch.equal(x[80, :, 16], torch.ones(16))                           # query flag at the first tail step
    # a fast-weight layer whose projections are identities stores v k^H and returns A q: exact recall
    for complex_ in (False, True):
        fw = FastWeights(4, d=4, heads=1, complex_=complex_, decay=0.999999)
        with torch.no_grad():
            for proj in (fw.k, fw.v, fw.q):
                if complex_:
                    proj.weight_re.copy_(torch.eye(4)); proj.weight_im.zero_()
                else:
                    proj.weight.copy_(torch.eye(4))
        h = torch.zeros(3, 1, 4)
        h[0, 0, 0] = 1.0        # write key e0 -> value e0 (key and value from the same activity)
        h[2, 0, 0] = 1.0        # query e0
        y_ = fw(h)
        assert torch.allclose(y_[2, 0].real if complex_ else y_[2, 0], torch.tensor([2.0, 0, 0, 0]), atol=1e-3)


@pytest.mark.parametrize("kind", ["lif", "cvsnn"])
def test_model_with_fast_weights_runs_and_learns_gradients(kind):
    torch.manual_seed(0)
    cfg = ModelConfig(kind=kind, hidden=[10], reset="none", fast_weights={"d": 4, "heads": 2}, leak_mag_range=(0.9, 0.99),
                      **({"leak_phase_range": (0.01, 1.5), "phase_dist": "log"} if kind == "cvsnn" else {}))
    m = SequenceClassifier(17, 8, cfg)
    x = torch.rand(20, 3, 17)
    x = torch.complex(x, torch.zeros_like(x)) if kind == "cvsnn" else x
    logits, _ = m(x)
    logits.sum().backward()
    g = m.fw.k.weight_re.grad if kind == "cvsnn" else m.fw.k.weight.grad
    assert torch.isfinite(logits).all() and g is not None and g.abs().sum() > 0


def test_delta_rule_replaces_the_value_stored_under_a_key():
    from cvsnn.layers import FastWeights
    for complex_ in (False, True):
        fw = FastWeights(4, d=4, heads=1, complex_=complex_, decay=0.999999, rule="delta", gate=True)
        with torch.no_grad():
            fw.gate.weight.copy_(torch.tensor([[0.0, 0.0, 10.0, 10.0]]))  # write only when a value channel is on
            fw.gate.bias.fill_(-5.0)
            for proj in (fw.k, fw.q):
                if complex_:
                    proj.weight_re.copy_(torch.eye(4)); proj.weight_im.zero_()
                else:
                    proj.weight.copy_(torch.eye(4))
            vw = torch.zeros(4, 4); vw[1, 2] = 1.0; vw[2, 3] = 1.0    # input e2 -> value e1, input e3 -> value e2
            if complex_:
                fw.v.weight_re.copy_(vw); fw.v.weight_im.zero_()
            else:
                fw.v.weight.copy_(vw)
        h = torch.zeros(3, 1, 4)
        h[0, 0, 0] = 1.0; h[0, 0, 2] = 1.0    # key e0 (normalised with e2) -> value e1
        h[1, 0, 0] = 1.0; h[1, 0, 3] = 1.0    # same key part, new value e2: the delta rule overwrites
        h[2, 0, 0] = 1.0
        y = fw(h)[2, 0]
        y = y.real if complex_ else y
        assert y[2] > y[1]                    # the newer value dominates (a Hebbian sum would keep both)


@pytest.mark.parametrize("kind", ["lif", "cvsnn"])
def test_kwta_lets_at_most_k_hidden_neurons_spike_per_step(kind):
    torch.manual_seed(0)
    cfg = ModelConfig(kind=kind, hidden=[20], reset="soft", threshold=0.1, kwta=0.1)
    m = SequenceClassifier(8, 4, cfg)
    x = torch.rand(15, 3, 8) * 3
    x = torch.complex(x, torch.zeros_like(x)) if kind == "cvsnn" else x
    captured = {}
    h = m.neurons[0].register_forward_hook(lambda mod, i, o: captured.__setitem__("o", o))
    logits, info = m(x)
    h.remove()
    assert torch.isfinite(logits).all() and info["transmit_rate"] <= 0.1 + 1e-6


def test_i21_key_stats_ideal_keys():
    """Keys = one-hot of the key identity, queries = the queried key, gate = 1 exactly at the write times:
    no overlap, perfect match, no cross-talk, gate on time."""
    import torch
    from cvsnn.data.synthetic import assoc_recall_batch
    from experiments.ep1b import TASK, key_stats

    P, nk = 4, TASK["n_keys"]

    x, _ = assoc_recall_batch(16, pairs=P, generator=torch.Generator().manual_seed(0), **TASK)
    T, B, _ = x.shape
    ident = x[:, :, :nk]
    k = torch.zeros(T, B, 2, nk)
    k[..., :] = ident.unsqueeze(2)
    k[ident.sum(-1) == 0] = torch.eye(nk)[0] * 0 + 1 / nk ** 0.5
    beta = (ident.sum(-1) > 0).float().unsqueeze(-1).expand(T, B, 2).clone()      # silent steps: some unit vector
    beta[TASK["window"]:] = 0
    for cplx in (False, True):
        kk = torch.complex(k, torch.zeros_like(k)) if cplx else k
        s = key_stats(x, {"k": kk, "q": kk, "beta": beta}, P)
        assert s["key_overlap"] < 1e-6 and s["key_overlap_gate"] < 1e-6
        assert abs(s["gate_on_time"] - 1) < 1e-6
        assert abs(s["query_match"] - 1) < 1e-6 and s["query_cross"] < 1e-6


def _p3_model(kind: str, feedback: str, seed: int = 0):
    import torch
    from cvsnn.models import ModelConfig, SequenceClassifier

    torch.manual_seed(seed)
    extra = ({"reset": "none", "leak_mag_range": [0.9, 0.999], "leak_phase_range": [0.01, 1.5], "phase_dist": "log", "threshold": 0.5}
             if kind == "cvsnn" else {"reset": "soft", "threshold": 1.0})
    return SequenceClassifier(16 if kind == "cvsnn" else 32, 8, ModelConfig(kind=kind, hidden=[12], feedback=feedback, **extra))


def _p3_grads(model, x, y):
    import torch.nn.functional as F
    model.zero_grad()
    F.cross_entropy(model(x)[0], y).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_p3_random_feedback_changes_only_the_error_path():
    """Same weights: identical logits and identical read-out gradients; the first layer gets a different signal,
    and with B set to the forward weights it gets exactly the backprop gradient."""
    import torch

    for kind in ("cvsnn", "lif"):
        exact, rand = _p3_model(kind, "exact"), _p3_model(kind, "random")
        rand.load_state_dict({**exact.state_dict(), **{k: v for k, v in rand.state_dict().items() if k.startswith("fb")}})
        x = torch.randn(20, 6, 16, dtype=torch.cfloat) if kind == "cvsnn" else torch.randn(20, 6, 32)
        y = torch.randint(0, 8, (6,))
        assert torch.allclose(exact(x)[0], rand(x)[0])
        ge, gr = _p3_grads(exact, x, y), _p3_grads(rand, x, y)
        first = [n for n in ge if n.startswith("linears.0")]
        last = [n for n in ge if n.startswith("linears.1")]
        assert first and last
        assert all(torch.allclose(ge[n], gr[n], atol=1e-6) for n in last)
        assert not all(torch.allclose(ge[n], gr[n], atol=1e-6) for n in first)
        with torch.no_grad():   # symmetric feedback: B = W -> exact gradient
            if kind == "cvsnn":
                rand.fb1_re.copy_(rand.linears[1].weight_re)
                rand.fb1_im.copy_(rand.linears[1].weight_im)
            else:
                rand.fb1.copy_(rand.linears[1].weight)
        gs = _p3_grads(rand, x, y)
        assert all(torch.allclose(ge[n], gs[n], atol=1e-5) for n in ge)


def test_p3_plastic_gate_only_masks_weight_learning():
    """Gates leave the values unchanged; gate = 1 everywhere gives the ungated gradients; gate = 0 everywhere stops
    every synaptic weight from learning but leaves the neuron parameters' gradients unchanged."""
    import torch
    from cvsnn.models import ModelConfig, SequenceClassifier

    for kind, gates in (("cvsnn", ("theta", "phase_rel", "random")), ("lif", ("theta", "random"))):
        base = _p3_model(kind, "random")
        x = torch.randn(20, 6, 16, dtype=torch.cfloat) if kind == "cvsnn" else torch.randn(20, 6, 32)
        y = torch.randint(0, 8, (6,))
        g0 = _p3_grads(base, x, y)
        for gate in gates:
            m = SequenceClassifier(base.linears[0].in_features, 8, ModelConfig(**{**base.cfg.__dict__, "plastic_gate": gate}))
            m.load_state_dict(base.state_dict())
            assert torch.allclose(m(x)[0], base(x)[0])
            m._plastic_gate = lambda neuron, syn, theta: torch.ones(syn.shape)
            g1 = _p3_grads(m, x, y)
            assert all(torch.allclose(g0[n], g1[n], atol=1e-6) for n in g0)
            m._plastic_gate = lambda neuron, syn, theta: torch.zeros(syn.shape)
            gz = _p3_grads(m, x, y)
            for n, gr in gz.items():
                if n.startswith("linears."):
                    assert torch.count_nonzero(gr) == 0, n
                else:
                    assert torch.allclose(gr, g0[n], atol=1e-6), n
            assert any(not n.startswith("linears.") for n in gz)


def test_p3_plastic_gates_open_half_the_time():
    import torch
    from cvsnn.models import ModelConfig, SequenceClassifier

    m = _p3_model("cvsnn", "random")
    m2 = SequenceClassifier(16, 8, ModelConfig(**{**m.cfg.__dict__, "plastic_gate": "phase_rel"}))
    seen = []
    orig = m2._plastic_gate
    m2._plastic_gate = lambda neuron, syn, theta: seen.append(orig(neuron, syn, theta)) or seen[-1]
    m2(torch.randn(40, 32, 16, dtype=torch.cfloat))
    assert all(0.35 < g.mean().item() < 0.65 for g in seen) and len(seen) == 2
    th = torch.arange(30).view(-1, 1, 1) * 2 * 3.141592653589793 / 10 + torch.rand(1, 8, 1) * 6.283
    assert abs((torch.cos(th) > 0).float().mean().item() - 0.5) < 0.05


def test_i28b_theta_tag_orthogonal_slots_and_task():
    """Complex theta tag with harmonics 0..7: the same key written at the 8 slot phases of one theta cycle gives 8
    mutually orthogonal keys; real tag keeps norms; the serial-recall labels are the items at the queried slot."""
    import math
    import torch
    from cvsnn.data.synthetic import serial_recall_batch
    from cvsnn.layers import FastWeights

    fw = FastWeights(5, d=8, heads=1, complex_=True, theta_tag=True)
    u = torch.polar(torch.ones(8) / 8 ** 0.5, torch.rand(8) * 6.283).view(1, 1, 1, 8).repeat(8, 1, 1, 1)  # (T=8, B=1, H=1, d)
    th = (2 * math.pi * torch.arange(8) / 8).view(8, 1)
    k = fw._tag(u, th)[:, 0, 0]                                      # (8, 8)
    G = (torch.conj(k) @ k.T).abs()
    assert torch.allclose(G, torch.eye(8), atol=1e-6)
    fr = FastWeights(5, d=11, heads=2, complex_=False, theta_tag=True)
    z = torch.randn(6, 3, 2, 11)
    assert torch.allclose(fr._tag(z, torch.rand(6, 3) * 6.283).norm(dim=-1), z.norm(dim=-1), atol=1e-5)
    x, theta, y, pos = serial_recall_batch(16, generator=torch.Generator().manual_seed(0))
    b = torch.arange(16)
    assert torch.equal(x[pos + 4, b, :8].argmax(-1), y)
    assert torch.allclose(theta[-1], 2 * math.pi * pos.float() / 8) and torch.all(x[-5:, :, 8] == 1)


def test_i28c_list_longer_than_a_theta_cycle():
    import math
    import torch
    from cvsnn.data.synthetic import serial_recall_batch

    x, theta, y, pos = serial_recall_batch(64, items=12, cycle_slots=8, cycle_cue=True, generator=torch.Generator().manual_seed(1))
    b = torch.arange(64)
    assert x.shape[-1] == 8 + 1 + 2 and torch.equal(x[pos + 4, b, :8].argmax(-1), y)
    assert torch.allclose(theta[-1], 2 * math.pi * (pos % 8).float() / 8)
    assert torch.equal(x[-1, :, 9:].argmax(-1), pos // 8)


def test_i26_rotate_frac_masks_omega():
    import torch
    from cvsnn.layers import CLIF

    torch.manual_seed(0)
    n = CLIF(16, leak_mag_range=(0.9, 0.999), leak_phase_range=(0.01, 1.5), phase_dist="log", reset="none", rotate_frac=0.25)
    assert int((n.phase != 0).sum()) == 4 and torch.all(n.beta.imag[n.rot_mask == 0] == 0)
    full = CLIF(16, leak_mag_range=(0.9, 0.999), leak_phase_range=(0.01, 1.5), phase_dist="log", reset="none")
    assert full.rot_mask is None and torch.equal(full.phase, full.leak_phase)


def test_i23_phi_ladder():
    import math
    import torch
    from cvsnn.frequencies import frequency_set

    w = frequency_set("phi_ladder", 32, 0.01, 1.5)
    r = torch.unique(w)
    assert len(r) == 11 and r.min() >= 0.01 - 1e-12 and r.max() <= 1.5
    assert torch.allclose(r[2:], r[1:-1] + r[:-2], atol=1e-12)         # neighbouring rungs add exactly
    assert torch.allclose(r[1:] / r[:-1], torch.full((10,), (1 + 5 ** 0.5) / 2, dtype=torch.float64))


def test_i23b_controls():
    import torch
    from cvsnn.frequencies import frequency_set

    w2 = frequency_set("ladder2", 32, 0.01, 1.5)
    r2 = torch.unique(w2)
    assert len(r2) == 8 and torch.allclose(r2[1:] / r2[:-1], torch.full((7,), 2.0, dtype=torch.float64))
    wb = frequency_set("banded_log", 32, 0.01, 1.5, generator=torch.Generator().manual_seed(0))
    assert len(torch.unique(wb)) == 11 and wb.min() >= 0.01 and wb.max() <= 1.5


def test_i28a_five_bands():
    import math
    import torch
    from cvsnn.frequencies import frequency_set

    b = frequency_set("brain5", 32, 0.01, 1.5)
    assert torch.allclose(torch.unique(b), torch.tensor([2.0, 6, 10, 20, 40], dtype=torch.float64) * 2 * math.pi / 1000)
    l5 = torch.unique(frequency_set("ladder5", 32, 0.01, 1.5))
    assert len(l5) == 5 and abs(l5[0].item() - 0.01) < 1e-12 and abs(l5[-1].item() - 1.5) < 1e-9


def test_fastweights_chunked_equals_loop():
    """The chunked scan gives the loop's outputs and gradients (real and complex, hebb and delta, with and without
    the write gate, theta tag, a length that is not a multiple of the chunk), up to float rounding."""
    import torch
    from cvsnn.layers import FastWeights

    for cplx, rule, gate, tag in [(True, "delta", True, False), (False, "delta", True, False), (True, "hebb", False, False),
                                  (False, "hebb", True, False), (True, "delta", False, True), (False, "delta", True, True)]:
        torch.manual_seed(0)
        a = FastWeights(6, d=5, heads=2, complex_=cplx, decay=0.9, rule=rule, gate=gate, theta_tag=tag).double()
        b = FastWeights(6, d=5, heads=2, complex_=cplx, decay=0.9, rule=rule, gate=gate, theta_tag=tag, chunk=4).double()
        b.load_state_dict(a.state_dict())
        with torch.no_grad():
            a.decay_raw.copy_(torch.tensor([0.3, 2.0]))
            b.decay_raw.copy_(torch.tensor([0.3, 2.0]))
            if gate:
                for m in (a, b):
                    m.gate.bias.fill_(0.5)
        h = torch.randn(23, 3, 6, dtype=torch.cdouble if cplx else torch.double)
        th = torch.rand(23, 3, dtype=torch.double) * 6.28 if tag else None
        ya, yb = a(h, th), b(h, th)
        assert torch.allclose(ya, yb, atol=1e-10), (cplx, rule, gate, tag, (ya - yb).abs().max())
        (ya.abs() ** 2).sum().backward()
        (yb.abs() ** 2).sum().backward()
        for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            assert torch.allclose(pa.grad, pb.grad, atol=1e-8), (cplx, rule, gate, tag, n)


def test_i23c_band_variants():
    import torch
    from cvsnn.frequencies import frequency_set

    w = frequency_set("phi_jitter", 32, 0.01, 1.5, generator=torch.Generator().manual_seed(0))
    lad = frequency_set("phi_ladder", 32, 0.01, 1.5)
    assert len(torch.unique(w)) >= 30 and torch.all((w / lad).log().abs() <= 0.05 + 1e-9)  # a few clamp to lo
    b = frequency_set("banded_strat", 32, 0.01, 1.5, generator=torch.Generator().manual_seed(0))
    u = torch.unique(b)
    assert len(u) == 11
    strata = torch.floor(torch.log(u / 0.01) / torch.log(torch.tensor(150.0, dtype=torch.float64)) * 11)
    assert torch.equal(strata, torch.arange(11, dtype=torch.float64))  # one band in every log-stratum


def test_i29_context_gate():
    """Half of the hidden neurons are active per context; silenced neurons pass nothing and their input weights get no
    gradient; neighbouring contexts share more neurons than opposite ones with the phase gate."""
    import torch
    import torch.nn.functional as F
    from cvsnn.models import ModelConfig, SequenceClassifier

    for gate in ("random", "phase"):
        torch.manual_seed(0)
        m = SequenceClassifier(16, 8, ModelConfig(kind="cvsnn", hidden=[40], reset="none", threshold=0.5, context_gate=gate, n_contexts=6))
        mk = m.ctx_masks[0]
        assert mk.shape == (6, 40) and 0.3 < mk.mean().item() < 0.7
        x = torch.randn(20, 4, 16, dtype=torch.cfloat)
        F.cross_entropy(m(x, context=2)[0], torch.zeros(4, dtype=torch.long)).backward()
        off = mk[2] == 0
        g = m.linears[0].weight_re.grad
        assert torch.all(g[off] == 0) and g[~off].abs().sum() > 0
        ov = lambda a, b: (mk[a] * mk[b]).sum().item()
        assert ov(0, 1) > ov(0, 3)  # phase gate: neighbours overlap more than opposite contexts


def test_i40_real_input_transforms():
    import torch
    from cvsnn.data.synthetic import input_transform, transformed_features
    x = torch.polar(torch.ones(3, 2, 4), torch.rand(3, 2, 4))
    r = input_transform(x, "real")
    assert not r.is_complex() and torch.equal(r, x.real) and transformed_features(4, "real") == 4
    c = input_transform(x, "complex_real")
    assert c.is_complex() and torch.equal(c.real, x.real) and c.imag.abs().max() == 0


def test_i41d_deep_lr_mult_splits_the_optimizer():
    from experiments.ecl import _run
    job = {"key": "t", "variant": "t", "seed": 0, "steps": 2, "kind": "cvsnn", "hidden": [6, 4], "input": "complex", "lr": 1e-3,
           "extra": {"reset": "none", "threshold": 0.5}}
    a = _run(job)
    b = _run({**job, "deep_lr_mult": 1.0})  # same lr in both groups: same result as one group
    assert a["forgetting"] == b["forgetting"] and a["final_avg_acc"] == b["final_avg_acc"]


def test_i41f_phasor_bundle_readback():
    import torch
    g = torch.Generator().manual_seed(0)
    x = torch.polar(torch.ones(30, 8, 16), torch.rand(30, 8, 16, generator=g))
    keys = torch.polar(torch.ones_like(x.real), 2 * torch.pi * torch.rand(x.real.shape, generator=g))
    bound = (keys * x).reshape(30, 2, 4, 16)
    dec = (keys.conj().reshape(bound.shape) * bound.sum(2, keepdim=True)).reshape(x.shape)
    err = dec - x                                                  # crosstalk of the 3 other samples in each bundle
    assert abs(err.abs().pow(2).mean().item() - 3.0) < 0.3         # power = K - 1 for unit-magnitude samples
    # a bundle of one is lossless
    one = (keys.conj()) * (keys * x))
    assert torch.allclose(one, x, atol=1e-5)


def test_i42_replay_controls_run():
    from experiments.ecl import _run
    base = {"key": "t", "variant": "t", "seed": 0, "steps": 2, "kind": "cvsnn", "hidden": [6], "input": "complex", "lr": 1e-3,
            "extra": {"reset": "none", "threshold": 0.5}}
    for rep in ({"per_task": 4, "frac": 0.25, "noise_match": 2}, {"per_task": 4, "frac": 0.25, "bundle_real": 2},
                {"per_task": 4, "frac": 0.25, "bundle": 2}):
        r = _run({**base, "replay": rep})
        assert 0.0 <= r["final_avg_acc"] <= 1.0


def test_i45_keep_A_at_matches_the_read_out():
    import torch
    from cvsnn.layers import FastWeights
    torch.manual_seed(0)
    fw = FastWeights(5, d=4, heads=2, complex_=True, rule="delta", gate=True)
    h = torch.randn(7, 3, 5)
    fw.keep_A_at = 4
    y = fw(h).view(7, 3, 2, 4)
    A, q = fw.last["A"], fw.last["q"]
    assert torch.allclose((A @ q[4].unsqueeze(-1)).squeeze(-1), y[4], atol=1e-6)


def test_i52_fw_protection_options_run():
    from experiments.ecl import _run
    from experiments.ep1b import FW2
    base = {"key": "t", "variant": "t", "seed": 0, "steps": 2, "kind": "lif", "hidden": [6], "input": "reim", "lr": 1e-3,
            "extra": {"reset": "none", "threshold": 0.5, "fast_weights": {**FW2, "d": 4}}}
    a = _run(base)
    b = _run({**base, "fw_lr_mult": 1.0})  # same lr in both groups: same result
    assert a["forgetting"] == b["forgetting"] and a["final_avg_acc"] == b["final_avg_acc"]
    c = _run({**base, "fw_freeze_after_first": True})
    assert 0.0 <= c["final_avg_acc"] <= 1.0


def test_i54e_reverse_order_and_16_pairs_run():
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 2, 1, reverse=True) if x["base"] == "lif_fw2_d11" and x["two_heads"]][0]
    r = _run(j)
    assert [h["after"] for h in r["history"]] == [f"ecl{t}" for t in range(5)] + ["recall"]
    assert r["recall_learned"] == r["recall_final"] and "eclfirst" in r["variant"]
    k = [x for x in phase_jobs(1, 2, 1, pairs=16) if x["base"] == "lif_fw2_d11" and not x["two_heads"]][0]
    r16 = _run(k)
    assert r16["pairs"] == 16 and r16["history"][0]["after"] == "recall"


def test_i56_key_sep_penalty_and_option():
    import torch
    from experiments.ep1b import FW2, TASK, _run, key_sep_penalty
    from cvsnn.data.synthetic import assoc_recall_batch
    x, _ = assoc_recall_batch(4, pairs=4, generator=torch.Generator().manual_seed(0), **TASK)
    T, B, H, d = x.shape[0], 4, 2, 8
    eye = torch.zeros(T, B, H, d)
    shown = x[:TASK["window"], :, :TASK["n_keys"]].sum(-1) > 0
    for b in range(B):  # orthogonal unit keys at the write times, identical keys elsewhere
        ts = torch.nonzero(shown[:, b]).flatten()
        for i, t in enumerate(ts):
            eye[t, b, :, i] = 1.0
    same = torch.zeros(T, B, H, d)
    same[..., 0] = 1.0
    assert key_sep_penalty(eye, x, 4, TASK, "write").item() == 0.0
    assert abs(key_sep_penalty(same, x, 4, TASK, "write").item() - 1.0) < 1e-6
    g = torch.Generator().manual_seed(1)
    assert abs(key_sep_penalty(same, x, 4, TASK, "placebo", g).item() - 1.0) < 1e-6
    job = {"key": "t", "pairs": 4, "variant": "lif_fw2_d11", "hp": {}, "seed": 0, "steps": 3, "lr": 3e-3,
           "model": {"kind": "lif", "hidden": [8], "readout_last": 5, "reset": "none", "fast_weights": {**FW2, "d": 4}}}
    a, b0 = _run(job), _run({**job, "key_sep": {"mu": 0.0, "where": "write"}})  # strength 0: the usual training, bit for bit
    assert a["test_acc"] == b0["test_acc"] and a["val_acc"] == b0["val_acc"]
    for where in ("placebo", "silent", "direction"):
        c = _run({**job, "key_sep": {"mu": 1.0, "where": where}})
        assert 0.0 <= c["test_acc"] <= 1.0
    dirp = key_sep_penalty(same, x, 4, TASK, "direction").item()  # collapsed keys: finite, non-negative, not forced to 1
    assert 0.0 <= dirp < 8.0


def test_i54j_i58_lif_short_and_si_options():
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 2, 1, lrs=[(1e-2, 3e-4)], variants=("lif_short_slowfw",)) if x["two_heads"]][0]
    a = _run(j)
    assert a["params"] > 0 and len(a["history"]) == 6
    b, c = _run({**j, "si_c": 0.0}), _run({**j, "si_c": 1.0})  # SI with strength 0: the usual training, bit for bit
    assert b["ecl_forgetting"] == a["ecl_forgetting"] and b["recall_final"] == a["recall_final"]
    assert 0.0 <= c["ecl_final_avg"] <= 1.0


def test_i61_adlif_matches_clif_with_real_input():
    """A cLIF neuron with real input, 0/1 spikes on Re U and rotation omega has the same subthreshold dynamics as an
    adLIF neuron with alpha = beta_w = r cos(omega), a = r^2 sin^2(omega), b = 0 (w = sin(omega) Im U)."""
    import torch
    from cvsnn.layers import CLIF, LIF
    torch.manual_seed(0)
    x = torch.randn(40, 3, 5) * 0.6
    for reset in ("none", "soft"):
        c = CLIF(5, 1.0, 0.95, 0.15, True, reset, spike_mode="binary", binary_on="re")
        a = LIF(5, 1.0, 0.95, True, reset, adapt={"omega": 0.15, "b": 0.0})
        _, u_c, h_c = c(torch.complex(x, torch.zeros_like(x)))
        h_a, u_a, _ = a(x)
        assert torch.allclose(u_c.real, u_a, atol=1e-5), reset
        assert torch.equal((h_c > 0.5).float(), (h_a > 0.5).float()), reset


def test_i61b_standard_adlif_runs_and_clamps():
    import torch
    from cvsnn.layers import LIF
    n = LIF(6, 0.5, 0.85, True, "soft", adapt={"style": "bittar"})
    with torch.no_grad():
        n.a.fill_(5.0)       # outside (-1, 1): clamped in the forward pass
    s, u, _ = n(torch.randn(30, 2, 6))
    assert s.shape == (30, 2, 6) and torch.isfinite(u).all()
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 2, 1, lrs=[(1e-2, 3e-4)], variants=("adlif_std_slowfw",)) if x["two_heads"]][0]
    assert 0.0 <= _run(j)["ecl_final_avg"] <= 1.0


def test_i61c_freeze_keeps_neuron_parameters():
    import torch
    from experiments.ei54 import _run, phase_jobs
    from cvsnn.layers import CLIF
    j = [x for x in phase_jobs(1, 3, 1, lrs=[(1e-2, 3e-4)], variants=("cand_long_frz_slowfw",)) if x["two_heads"]][0]
    assert j["freeze_neuron"] and j["base"] == "cand_long"
    r = _run(j)
    assert 0.0 <= r["ecl_final_avg"] <= 1.0


def test_i63_shd_binning_is_right_aligned():
    import numpy as np
    from cvsnn.data.shd import bin_spikes
    t = np.array([0.0, 0.6, 1.1, 1.1, 1.105])
    u = np.array([0, 699, 21, 22, 350])
    x = bin_spikes(t, u, t_steps=85, channels=32, dt=0.012)
    assert x.shape == (85, 32) and x.dtype == np.uint8
    assert x[84, 0] == 1 and x[84, 1] == 1 and x[84, 16] == 1  # the last spike (and those within dt of it) in the last step
    assert x[84 - int(0.505 // 0.012), 31] == 1                 # unit 699 -> group 31
    assert x.sum() == 4                                         # t = 0 is more than 85 * dt before the last spike: dropped
    assert bin_spikes(np.array([]), np.array([])).sum() == 0


def test_i63_split_shd_runs():
    import pytest
    from experiments.ei54 import ROOT, _run, phase_jobs
    if not (ROOT / "data/shd.npz").exists():
        pytest.skip("needs research/data/shd.npz (tools/convert_shd.py)")
    j = [x for x in phase_jobs(1, 2, 2, lrs=[(1e-2, 1e-3)], variants=("cand_long_fw003",)) if x["two_heads"]][0]
    r = _run({**j, "shd": True, "hidden": 40})
    assert 0.0 <= r["ecl_final_avg"] <= 1.0 and len(r["history"][-1]["ecl"]) == 5


def test_i64_replay_leaves_the_first_task_unchanged():
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 2, 3, lrs=[(1e-2, 1e-3)], variants=("cand_long_fw003",)) if x["two_heads"]][0]
    a, b = _run(j), _run({**j, "replay": 5})
    assert a["history"][1] == b["history"][1]        # no buffer during the first later-phase task: identical training
    assert a["history"][-1] != b["history"][-1]      # later tasks train on current + replayed samples


def test_i65_se_adlif_and_brf():
    import math
    import torch
    from cvsnn.layers import LIF
    torch.manual_seed(0)
    # SE-adLIF: one step by hand (the adaptation uses the membrane after the update and reset)
    n = LIF(3, 0.5, 0.85, True, "hard", adapt={"style": "se"})
    x = torch.tensor([[[0.2, 5.0, -0.1]]])
    s, pot, _ = n(x)
    alpha = torch.exp(-1.0 / (5.0 + 20.0 * n.th_u))
    u_hat = (1 - alpha) * n.in_gain * x[0, 0]
    assert torch.allclose(pot[0, 0], u_hat, atol=1e-6) and s[0, 0, 1] == 1 and s[0, 0, 0] == 0
    with torch.no_grad():
        n.a_hat.fill_(3.0)                  # outside [0, 1): clamped in the forward pass
    assert torch.isfinite(n(torch.randn(40, 2, 3))[1]).all()
    # BRF: on the divergence boundary (b' = 0, no spikes) |1 + p + i omega| = 1; with b' > 0 the state decays
    b = LIF(4, 1e9, 0.85, True, "none", adapt={"style": "brf"})
    om = b.omega_p.detach().abs()
    p = -1.0 + torch.sqrt(1.0 - om ** 2)
    assert torch.allclose((1 + p) ** 2 + om ** 2, torch.ones(4), atol=1e-6)
    x = torch.zeros(200, 1, 4)
    x[0] = 1.0
    _, ur, _ = b(x)
    assert ur[-1].abs().max() < 0.05 and 0.05 <= float(om.min()) and float(om.max()) <= 0.10
    # a spike raises the threshold and the damping (smooth reset), but only by delta * q (delta = 0.01): one step after a
    # spike the neuron keeps most of its state (the first version used q unscaled and wiped it: |u| fell to ~3%)
    c = LIF(1, 0.5, 0.85, True, "none", adapt={"style": "brf"})
    x = torch.zeros(30, 1, 1)
    x[0] = 2.0
    s, ur, _ = c(x)
    assert s[0].sum() == 1 and ur[1].abs() > 0.9 * ur[0].abs()
    from experiments.ei54 import _run, phase_jobs
    for v in ("se_adlif_slowfw", "brf_slowfw"):
        j = [x for x in phase_jobs(1, 2, 1, lrs=[(1e-2, 3e-4)], variants=(v,)) if x["two_heads"]][0]
        assert 0.0 <= _run(j)["ecl_final_avg"] <= 1.0


def test_i66_standard_shd_runs():
    import numpy as np
    import pytest
    from cvsnn.data.shd import bin_spikes
    x = bin_spikes(np.array([0.0, 0.005, 0.999, 1.5]), np.array([0, 4, 699, 3]), t_steps=100, channels=140, dt=0.01, align="left")
    assert x[0, 0] == 2 and x[99, 139] == 1 and x.sum() == 3        # 1.5 s is past the 100 x 10 ms window: dropped
    from experiments.ei66 import ROOT, _run, jobs
    if not (ROOT / "data/shd140.npz").exists():
        pytest.skip("needs research/data/shd140.npz (tools/convert_shd.py ... --channels 140 --dt 0.01 --steps 100 --align left)")
    r = _run({**jobs(1, epochs=1, lrs=(3e-3,), cells=("cand_long_fw003",))[0], "max_steps": 2, "hidden": 20})
    assert 0.0 <= r["test"] <= 1.0 and len(r["curve"]) == 1


def test_i69_ten_tasks_extend_the_five_task_sequence():
    import pytest
    from experiments.ei54 import ROOT, _run, phase_jobs
    if not (ROOT / "data/mnist.npz").exists():
        pytest.skip("needs research/data/mnist.npz")
    j = [x for x in phase_jobs(1, 2, 2, lrs=[(1e-2, 1e-3)], variants=("cand_long_fw003",)) if x["two_heads"]][0]
    a, b = _run(j), _run({**j, "n_tasks": 10})
    assert b["n_tasks"] == 10 and len(b["history"][-1]["ecl"]) == 10 and len(b["seen_avg"]) == 10
    assert b["history"][1]["ecl"][:5] == a["history"][1]["ecl"]  # tasks 0-4 are the same tasks, the same training up to task 1


def test_i71_phase1_cache_gives_identical_results(tmp_path):
    from experiments.ei54 import _run, phase_jobs
    for v, extra in (("cand_long_fw003", {}), ("lif_short_fw003", {"replay": 4})):
        j = {**[x for x in phase_jobs(1, 20, 3, lrs=[(1e-2, 1e-3)], variants=(v,)) if x["two_heads"]][0], **extra, "cache_dir": str(tmp_path)}
        a = _run({**j, "no_cache": True})
        b = _run(j)                      # computes phase 1 and saves it
        assert len(list(tmp_path.glob("*.pt"))) >= 1
        c = _run(j)                      # loads phase 1
        keys = ("recall_learned", "recall_final", "ecl_final_avg", "ecl_forgetting", "val_score", "history")
        assert all(a[k] == b[k] == c[k] for k in keys), v


def test_i72_babi_encoding_and_run():
    import pytest
    import torch
    from cvsnn.data.babi import encode_batch, steps_of
    a = {"sents": [[0, 1], [2]], "q": [3], "a": 0}
    b = {"sents": [[1]], "q": [0], "a": 1}
    x = encode_batch([a, b], 4)
    assert x.shape == (steps_of(2), 2, 5)
    assert x[0, 0, 0] == 1 and x[0, 0, 1] == 1 and x[2, 0, 2] == 1 and x[4, 0, 3] == 1 and x[4, 0, 4] == 1  # sentence, sentence, question + flag
    assert x[:2, 1].sum() == 0 and x[2, 1, 1] == 1                  # the shorter story is padded at the start
    assert torch.equal(x[-3:], torch.zeros_like(x[-3:]))            # three empty steps before the end of the read-out window
    from experiments.ei72 import ROOT, _run, jobs
    if not (ROOT / "data/babi1k.json").exists():
        pytest.skip("needs research/data/babi1k.json (tools/convert_babi.py)")
    r = _run({**jobs([1], 1, epochs=1, lrs=(1e-3,), cells=("cand_long_fw003",), hidden=20)[0], "max_steps": 2})
    assert 0.0 <= r["test"] <= 1.0


def test_appendix_selection_ignores_held_out_seeds(tmp_path):
    """Adding the hold-out seeds 5-9 at the chosen setting must not move the choice to a setting that has only seeds 0-4."""
    import json
    from experiments.ei54 import analyse
    v = "cand_long_fw003|2heads|phaselr"
    base = {"variant": v, "params": 1, "recall_learned": 1.0, "recall_final": 1.0, "ecl_final_avg": 1.0, "ecl_learned_avg": 1.0,
            "ecl_forgetting": 0.0, "ecl_end_of_phase": 1.0}
    rows = [{**base, "lr": "0.01/0.0003", "seed": s, "val_score": 0.99 if s < 5 else 0.90} for s in range(10)]
    rows += [{**base, "lr": "0.01/0.0001", "seed": s, "val_score": 0.98} for s in range(5)]
    analyse(rows, tmp_path)
    sel = json.loads((tmp_path / "selected.json").read_text())
    assert sel[v]["lr"] == "0.01/0.0003" and sel[v]["seeds"] == 10
    assert sel["best|cand_long|2heads|phaselr"]["lr"] == "0.01/0.0003"


def test_n10_save_model_records_neuron_params_and_n11_extra_cells_freeze(tmp_path):
    import torch
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 5, 2, lrs=[(1e-2, 3e-4)], variants=("cand_long_fw003",)) if x["two_heads"]][0]
    a = _run({**j, "no_cache": True})
    b = _run({**j, "no_cache": True, "save_model": str(tmp_path / "m.pt")})

    assert all(a[k] == b[k] for k in ("recall_learned", "recall_final", "ecl_final_avg", "val_score", "history"))  # recording changes nothing
    sv = torch.load(tmp_path / "m.pt", weights_only=False)
    assert set(sv) >= {"init", "phase1", "final", "state_dict"} and len(sv["init"]) == 1
    assert torch.allclose(sv["init"][0]["omega"], torch.full_like(sv["init"][0]["omega"], 0.15))
    assert not torch.equal(sv["init"][0]["omega"], sv["final"][0]["omega"])          # omega is learned
    for v, mult in (("cand_long_frz_fw001", 0.01), ("adlif_long_frz_fw03", 0.3)):
        jf = [x for x in phase_jobs(1, 3, 2, lrs=[(3e-2, 3e-4)], variants=(v,)) if x["two_heads"]][0]
        assert jf["freeze_neuron"] and jf["fw_lr_mult"] == mult
    jf = [x for x in phase_jobs(1, 3, 2, lrs=[(3e-2, 3e-4)], variants=("cand_long_frz_fw001",)) if x["two_heads"]][0]
    sv = _run({**jf, "no_cache": True, "save_model": str(tmp_path / "f.pt")}) and torch.load(tmp_path / "f.pt", weights_only=False)
    assert torch.equal(sv["init"][0]["omega"], sv["final"][0]["omega"]) and torch.equal(sv["init"][0]["r"], sv["final"][0]["r"])


def test_n12_save_model_records_lif_memory():
    import tempfile
    from pathlib import Path

    import torch
    from experiments.ei54 import _run, phase_jobs
    j = [x for x in phase_jobs(1, 5, 2, lrs=[(1e-1, 1e-3)], variants=("lif_fw2_d11_fw001",)) if x["two_heads"]][0]
    with tempfile.TemporaryDirectory() as d:
        _run({**j, "no_cache": True, "save_model": str(Path(d) / "m.pt")})
        sv = torch.load(Path(d) / "m.pt", weights_only=False)
    assert len(sv["init"]) == 1 and torch.all(sv["init"][0]["omega"] == 0)
    assert 0.89 < float(sv["init"][0]["r"].min()) and float(sv["init"][0]["r"].max()) < 0.9995  # the |beta| ring [0.9, 0.999]
    assert not torch.equal(sv["init"][0]["r"], sv["final"][0]["r"])                          # the LIF leak is learned


def test_n19_gated_rnn_reference_matches_parameters_and_runs():
    import torch
    from cvsnn.models import ModelConfig, SequenceClassifier
    from experiments.ei54 import C_ECL, EXTRA_MODELS, _matched_hidden, _run, phase_jobs
    for name in ("gru", "lstm"):
        cfg = {**EXTRA_MODELS[name], "threshold": 0.5, "hidden": [30]}
        h = _matched_hidden(cfg, C_ECL + 17, 16)
        p_rnn = SequenceClassifier(C_ECL + 17, 16, ModelConfig(**{**cfg, "hidden": [h]})).num_params()
        p_lif = SequenceClassifier(C_ECL + 17, 16, ModelConfig(**{**cfg, "rnn_cell": None})).num_params()
        assert abs(p_rnn - p_lif) / p_lif < 0.05                    # matched to the LIF network within 5%
        m = SequenceClassifier(C_ECL + 17, 16, ModelConfig(**{**cfg, "hidden": [h]}))
        assert isinstance(m.linears[0], torch.nn.Identity) and m.fw is not None  # same plastic synapse, non-spiking hidden layer
    j = [x for x in phase_jobs(1, 3, 2, lrs=[(3e-3, 1e-3)], variants=("lstm_fw003",)) if x["two_heads"]][0]
    r = _run({**j, "no_cache": True})
    assert 0.0 <= r["recall_final"] <= 1.0 and r["params"] == p_rnn


def test_n9_long_memory_rotating_network_with_soft_reset():
    from experiments.ei54 import EXTRA_MODELS, LONGRESET_VARIANTS, MODELS, NOROT_LRS, _run, phase_jobs
    cfg = {**MODELS, **EXTRA_MODELS}["cand_long_reset"]
    long = {**MODELS, **EXTRA_MODELS}["cand_long"]
    assert cfg["reset"] != "none" and cfg["leak_mag_range"] == long["leak_mag_range"] == [0.9, 0.999]  # only the reset differs
    assert {k: v for k, v in cfg.items() if k != "reset"} == {k: v for k, v in long.items() if k != "reset"}
    js = [j for v in LONGRESET_VARIANTS for j in phase_jobs(5, 2, 1, lrs=NOROT_LRS, variants=(v,)) if j["two_heads"]]
    ref = [j for v in ("cand_long_slowfw", "cand_long_fw003", "cand_long_fw001") for j in phase_jobs(5, 2, 1, lrs=NOROT_LRS, variants=(v,)) if j["two_heads"]]
    assert len(js) == len(ref) == 150 and {j["base"] for j in js} == {"cand_long_reset"}  # the same grid as I54i
    r = _run([j for j in js if j["seed"] == 0][0])
    assert 0.0 <= r["ecl_final_avg"] <= 1.0
