"""Sequence classifiers with a shared interface, so models differ only in the neuron type.

Every model maps a time-major input (T, B, F) to class logits (B, C). The last layer always has
C units and the read-out is the time average of a per-unit score (same idea as the browser engine):

* cvsnn : |U_pre| / V_th + 0.5 * spike    (complex spiking; readout='re': Re U_pre / V_th + 0.5 * spike)
* lif   : V_pre / V_th + 0.5 * spike      (real spiking)
* cvnn  : |modReLU(U)|                    (complex, non-spiking)
* lru   : y                               (complex-diagonal SSM, non-spiking)
* prf   : Re(u) / V_th + 0.5 * spike      (complex state, binary spikes; Huang et al. 2024)
* csilif: (2 Re(u) - V_th) + 0.5 * spike  (complex state, binary spikes; REACT, Keime et al. 2026)

``logits = temperature * mean_t(score)``; temperature 2.5 matches the JS engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import math

import torch
import torch.nn as nn

from .hardware.nonideal import DeviceConfig, layer_scale, program_weight, read_out
from .layers import CLIF, LIF, LRU, PRF, ComplexLeakyRNN, ComplexLinear, CSiLIF, FastWeights, GatedRNN, complex_magnitude

KINDS = ("cvsnn", "lif", "cvnn", "lru", "prf", "csilif")


@dataclass
class ModelConfig:
    kind: str = "cvsnn"
    hidden: list[int] = field(default_factory=list) # [] = single layer (like the JS engine)
    threshold: float = 1.0
    leak_mag: float = 0.85
    leak_phase: float = 0.15
    learn_leak: bool = True
    reset: str = "soft"
    surrogate: str = "von_mises"
    sharpness: float = 2.5
    spike_mode: str = "phasor"    # cvsnn: "phasor" | "graded" | "binary" (0/1 events, phase stays in the neuron)
    binary_on: str = "mag"        # cvsnn binary spikes: threshold on |U| ("mag") or on Re U ("re")
    phase_bins: int | None = None # timing-coded spikes (cvsnn hidden layers); None = exact phasor
    karatsuba: bool = True
    init: str = "js"
    temperature: float = 2.5
    lru_state: int = 32
    lru_r_min: float = 0.4
    lru_r_max: float = 0.99
    lru_max_phase: float = 3.141592653589793
    # heterogeneous initialisation (None = every neuron gets leak_mag / leak_phase); leak_mag_range also
    # applies to LIF (per-neuron time constants)
    leak_mag_range: tuple | None = None
    leak_phase_range: tuple | None = None
    phase_dist: str = "uniform"   # one of cvsnn.frequencies.FREQ_SETS
    phase_bits: int | None = None # grid of the 'quantized' / 'sidon' frequency sets
    learn_phase: bool = True      # False: frequencies stay at their initial set, |beta| still learns
    mem_frac: float | None = None # cvsnn: split layers into processing + memory neurons (see CLIF)
    # with mem_frac: the output layer gets one processing and one memory neuron per class (score = their sum), so
    # every class has a read-out neuron that can hold information through a delay
    out_pairs: bool = False
    # cvsnn with real synaptic weights (H6e): cLIF neurons keep a complex, rotating state, but every synapse is a
    # real weight on real inputs ([Re, Im] or |x|) and binary spikes, as in PRF / C-SiLIF
    real_synapses: bool = False
    # soft reset (detached) computed as a no-grad pass + FFT recurrence: same values and gradients, much faster on
    # long sequences (cvsnn and lif hidden/output layers)
    parallel_reset: bool = False
    # E-FRONT: a complex front end (one cLIF resonator per sensor channel, E-HW dynamics: soft reset, |beta| 0.85,
    # omega 0.15) feeding a real LIF core (kind='lif', encoder=True). "binre": 0/1 spikes on Re U; "k4": timing code
    # as 4 real slot lines per channel. front_fixed: the resonators are not trained (a fixed sensor front end)
    front: str | None = None
    front_fixed: bool = False
    # E-FRONT v2: resonator layers between the front end and the LIF core that combine the channels (cLIF, real
    # synapses, 0/1 spikes on Re U, soft reset, E-HW leak); `hidden` then lists the LIF core layers
    front_hidden: list[int] = field(default_factory=list)
    # Q1''(c): a memory branch of integrators (LIF, no reset, long heterogeneous time constants) in parallel with
    # the (single) LIF core layer; it reads the same input as the core and the read-out layer sees both
    memory_branch: int = 0
    memory_leak_range: tuple = (0.99, 0.9999)
    # I15 (c'): the memory branch reads the raw input ([Re, Im] of complex sensors), in parallel with the whole
    # periphery, instead of the output of the resonator layer that combines channels
    memory_from_input: bool = False
    memory_input: str = "reim"
    # P1b: Hebbian fast-weight layer after the last hidden layer ({"d": 8, "heads": 2}); the read-out layer sees
    # the hidden spikes and the fast-weight output
    fast_weights: dict | None = None
    # selective sparsity (k-winners-take-all): in every hidden layer only the kwta fraction of neurons with the
    # strongest drive (|U| for cvsnn, V for lif) may send spikes at each step; the others are silenced
    kwta: float | None = None
    # H5: deadline-dependent urgency in every non-encoder cLIF / LIF layer ('damping' | 'threshold' | 'gain'; control
    # 'damping_const' = the same extra leak at every step), with the
    # deadline in steps (see cvsnn.layers.Urgency)
    urgency: str | None = None
    deadline: int | None = None   # with memory_from_input: "reim" ([Re, Im] of complex sensors) or "real" (real input)
    readout: str = "mag"          # cvsnn read-out score: |U| / V_th ("mag", phase-invariant, >= 0) or Re U / V_th ("re", signed)
    brf_gamma: float = 0.9
    brf_strength: float = 0.01
    # PRF / C-SiLIF baselines
    prf_tau: float = 2.0
    prf_theta_max: float = 0.7853981633974483
    prf_dt_range: tuple = (0.001, 0.1)
    csilif_lambda_im: float = 3.141592653589793
    csilif_rho_range: tuple = (0.001, 0.1)
    # Spiking sensor front end: each input channel drives its own neuron (fixed gain, not learned), so
    # everything downstream sees only spikes, as on a neuromorphic chip fed by spiking sensors.
    encoder: bool = False
    encoder_gain: float = 1.0
    encoder_reset: str | None = None  # reset of the encoder neurons (None = same as the other layers)
    # Read-out window: 0 = average over the whole sequence; R > 0 = only the last R steps
    # (memory tasks must use R > 0, otherwise the read-out can see the inputs directly)
    readout_last: int = 0
    # P3/N2: how the error reaches the layers before the read-out. "exact": backprop (for these nets, with no
    # connections between neurons of a layer and a detached reset, this equals e-prop with symmetric feedback,
    # Bellec et al. 2020, i.e. an online rule: eligibility trace x learning signal). "random": e-prop with random
    # feedback: the learning signal travels back through a fixed random matrix B instead of the transpose of the
    # forward weights (feedback alignment, Lillicrap et al. 2016), so no synapse needs to know another's weight.
    # Forward values are unchanged; every weight update stays local (presynaptic trace x postsynaptic signal).
    feedback: str = "exact"
    # P3 step 3: phase-gated plasticity. Synaptic weights may change only at gated (time, sample, neuron) points; the
    # values, the learning signal passed to earlier layers and the learning of neuron parameters are unchanged.
    #   None        : no gate
    #   "theta"     : a global slow rhythm theta_t = 2 pi t / theta_period + phi (phi random per sample); synapses learn
    #                 in one half cycle, cos theta_t > 0 (separate encoding phase, Hasselmo et al. 2002)
    #   "phase_rel" : synapses onto neuron j learn when j's membrane phase is within +-90 deg of theta_t (cvsnn only)
    #   "random"    : control, an input-independent random gate open half of the time (per time, sample, neuron)
    rotate_frac: float | None = None  # I25: only this fraction of the cLIF neurons rotate (the rest omega = 0); None = all
    input_norm: bool = False          # I70: cLIF input scaled by sqrt(1 - |beta|^2) (LRU-style), see CLIF
    lif_adapt: dict | None = None     # I61: kind 'lif' neurons become adLIF (adaptation current); see LIF
    rnn_cell: str | None = None       # N19: kind 'lif' hidden layers become a non-spiking GRU / LSTM (read-out stays LIF)
    # I29: context-dependent gating of the hidden layers (Masse et al. 2018, XdG), the task given as context: only a
    # subset of hidden neurons is active (and so learns) in each task. "random": a fixed random half per task;
    # "phase": task t gets a theta phase 2 pi t / n_contexts, neuron j a preferred phase psi_j, and j is active when
    # cos(phase_t - psi_j) > 0 (half of the neurons; neighbouring tasks share more neurons)
    context_gate: str | None = None
    n_contexts: int = 5
    plastic_gate: str | None = None
    theta_period: int = 10


def rate_penalty(info: dict, target: float | None, strength: float) -> torch.Tensor | float:
    """strength * max(0, r - target): r = mean firing rate of the spikes sent between layers. Pushes a model
    towards sparse activity (fewer synaptic events) without forcing it below the target."""
    if target is None or strength == 0 or "transmit_rate" not in info:
        return 0.0
    return strength * torch.relu(info["transmit_rate"] - target)


def neuron_out_features(neuron: nn.Module) -> int:
    if isinstance(neuron, (CLIF, PRF, CSiLIF, GatedRNN)):
        return neuron.num_features
    if isinstance(neuron, LRU):
        return neuron.D.shape[0]
    if isinstance(neuron, LIF):
        return neuron.leak_raw.shape[0]
    return neuron.leak_mag_raw.shape[0] # ComplexLeakyRNN


def _rng(r):
    return None if r is None else (float(r[0]), float(r[1]))


class ChannelEncoder(nn.Module):
    """Fixed per-channel gain: channel i drives encoder neuron i only (fan-out 1, not learned)."""

    def __init__(self, gain: float):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain


class SequenceClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, cfg: ModelConfig):
        super().__init__()
        if cfg.kind not in KINDS:
            raise ValueError(f"unknown model kind {cfg.kind!r}; expected one of {KINDS}")
        if cfg.encoder and cfg.kind not in ("cvsnn", "lif", "prf", "csilif"):
            raise ValueError("encoder is only defined for spiking models (cvsnn, lif, prf, csilif)")
        if cfg.readout not in ("mag", "re"):
            raise ValueError("readout must be 'mag' or 're'")
        if cfg.real_synapses and (cfg.kind != "cvsnn" or cfg.spike_mode != "binary"):
            raise ValueError("real_synapses needs kind='cvsnn' with binary spikes (a real weight cannot take a phasor)")
        if cfg.front_hidden and not cfg.front:
            raise ValueError("front_hidden needs a front end (front='binre' | 'k4')")
        if cfg.front and (cfg.kind != "lif" or not cfg.encoder or cfg.front not in ("binre", "k4")):
            raise ValueError("front ('binre' | 'k4') needs kind='lif' (the core) and encoder=True")
        if cfg.out_pairs and not cfg.mem_frac:
            raise ValueError("out_pairs needs mem_frac (one processing and one memory read-out neuron per class)")
        self.cfg = cfg
        self.num_classes = num_classes
        dims = [in_features * (4 if cfg.front == "k4" else 1), *cfg.front_hidden, *cfg.hidden,
                num_classes * (2 if cfg.out_pairs else 1)]
        self.linears = nn.ModuleList()
        self.neurons = nn.ModuleList()
        if cfg.encoder:
            self.linears.append(ChannelEncoder(cfg.encoder_gain))
            if cfg.front:
                kw = dict(spike_mode="binary", binary_on="re") if cfg.front == "binre" else dict(spike_mode="slots", phase_bins=4)
                self.neurons.append(CLIF(in_features, cfg.threshold, 0.85, 0.15, not cfg.front_fixed, cfg.encoder_reset or "soft",
                                         surrogate=cfg.surrogate, sharpness=cfg.sharpness, parallel_reset=cfg.parallel_reset, **kw))
            elif cfg.kind == "cvsnn":
                # the encoder emits the same kind of spike as the hidden layers (binary variants send no phase at all)
                self.neurons.append(CLIF(in_features, cfg.threshold, cfg.leak_mag, cfg.leak_phase, cfg.learn_leak,
                                         cfg.encoder_reset or cfg.reset,
                                         surrogate=cfg.surrogate, sharpness=cfg.sharpness, phase_bins=cfg.phase_bins,
                                         spike_mode=cfg.spike_mode if cfg.spike_mode == "binary" else "phasor",
                                         binary_on=cfg.binary_on))
            elif cfg.kind == "lif":
                self.neurons.append(LIF(in_features, cfg.threshold, cfg.leak_mag, cfg.learn_leak, cfg.encoder_reset or cfg.reset,
                                        surrogate=cfg.surrogate, sharpness=cfg.sharpness))
            elif cfg.kind == "prf":
                self.neurons.append(PRF(in_features, cfg.threshold, cfg.prf_tau, cfg.prf_theta_max, tuple(cfg.prf_dt_range),
                                        sharpness=cfg.sharpness))
            else:
                self.neurons.append(CSiLIF(in_features, cfg.threshold, cfg.csilif_lambda_im, tuple(cfg.csilif_rho_range),
                                           sharpness=cfg.sharpness))
        for layer, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if layer < len(cfg.front_hidden):     # E-FRONT v2: resonators that combine the channels
                lin = nn.Linear(d_in, d_out, bias=False)
                nn.init.normal_(lin.weight, 0.0, (1.0 / d_in) ** 0.5)
                self.linears.append(lin)
                self.neurons.append(CLIF(d_out, cfg.threshold, 0.85, 0.15, True, "soft", surrogate=cfg.surrogate,
                                         sharpness=cfg.sharpness, spike_mode="binary", binary_on="re",
                                         parallel_reset=cfg.parallel_reset))
                continue
            if cfg.kind == "cvsnn":
                if cfg.real_synapses:
                    lin = nn.Linear(d_in, d_out, bias=False)
                    nn.init.normal_(lin.weight, 0.0, (1.0 / d_in) ** 0.5)
                    self.linears.append(lin)
                else:
                    self.linears.append(ComplexLinear(d_in, d_out, karatsuba=cfg.karatsuba, init=cfg.init))
                self.neurons.append(
                    CLIF(d_out, cfg.threshold, cfg.leak_mag, cfg.leak_phase, cfg.learn_leak, cfg.reset,
                         surrogate=cfg.surrogate, sharpness=cfg.sharpness, spike_mode=cfg.spike_mode,
                         phase_bins=cfg.phase_bins, leak_mag_range=_rng(cfg.leak_mag_range),
                         leak_phase_range=_rng(cfg.leak_phase_range), phase_dist=cfg.phase_dist,
                         brf_gamma=cfg.brf_gamma, brf_strength=cfg.brf_strength, binary_on=cfg.binary_on,
                         learn_phase=cfg.learn_phase, phase_bits=cfg.phase_bits, mem_frac=cfg.mem_frac,
                         parallel_reset=cfg.parallel_reset, urgency=cfg.urgency, deadline=cfg.deadline,
                         rotate_frac=cfg.rotate_frac, input_norm=cfg.input_norm)
                )
            elif cfg.kind == "cvnn":
                self.linears.append(ComplexLinear(d_in, d_out, karatsuba=cfg.karatsuba, init=cfg.init))
                self.neurons.append(ComplexLeakyRNN(d_out, cfg.leak_mag, cfg.leak_phase))
            elif cfg.kind == "lif" and cfg.rnn_cell and layer < len(dims) - 2:   # N19: non-spiking hidden layer
                self.linears.append(nn.Identity())
                self.neurons.append(GatedRNN(d_in, d_out, cfg.rnn_cell))
            elif cfg.kind == "lif":
                lin = nn.Linear(d_in, d_out, bias=False)
                nn.init.normal_(lin.weight, 0.0, (1.5 / d_in) ** 0.5 * 0.7)
                self.linears.append(lin)
                self.neurons.append(LIF(d_out, cfg.threshold, cfg.leak_mag, cfg.learn_leak, cfg.reset,
                                        surrogate=cfg.surrogate, sharpness=cfg.sharpness, leak_range=_rng(cfg.leak_mag_range),
                                        parallel_reset=cfg.parallel_reset, urgency=cfg.urgency, deadline=cfg.deadline,
                                        adapt=cfg.lif_adapt))
            elif cfg.kind in ("prf", "csilif"):
                lin = nn.Linear(d_in, d_out, bias=False)
                nn.init.normal_(lin.weight, 0.0, (1.0 / d_in) ** 0.5)
                self.linears.append(lin)
                if cfg.kind == "prf":
                    self.neurons.append(PRF(d_out, cfg.threshold, cfg.prf_tau, cfg.prf_theta_max, tuple(cfg.prf_dt_range),
                                            sharpness=cfg.sharpness))
                else:
                    self.neurons.append(CSiLIF(d_out, cfg.threshold, cfg.csilif_lambda_im, tuple(cfg.csilif_rho_range),
                                               sharpness=cfg.sharpness))
            else: # lru
                self.linears.append(nn.Identity())
                self.neurons.append(LRU(d_in, cfg.lru_state, d_out, cfg.lru_r_min, cfg.lru_r_max, cfg.lru_max_phase))
        self.mem_linear = self.mem_neuron = None
        if cfg.memory_branch:
            if not cfg.front or len(cfg.hidden) != 1:
                raise ValueError("memory_branch needs a front end and exactly one core layer (hidden=[n])")
            if cfg.memory_from_input:
                d_core_in = in_features * (2 if cfg.memory_input == "reim" else 1)
            else:
                d_core_in = self.linears[-2].in_features
            self.mem_from_input = cfg.memory_from_input
            self.mem_linear = nn.Linear(d_core_in, cfg.memory_branch, bias=False)
            nn.init.normal_(self.mem_linear.weight, 0.0, (1.5 / d_core_in) ** 0.5 * 0.7)
            self.mem_neuron = LIF(cfg.memory_branch, cfg.threshold, cfg.leak_mag, cfg.learn_leak, "none",
                                  surrogate=cfg.surrogate, sharpness=cfg.sharpness, leak_range=_rng(cfg.memory_leak_range))
            out = nn.Linear(cfg.hidden[-1] + cfg.memory_branch, dims[-1], bias=False)   # read-out sees core + memory
            nn.init.normal_(out.weight, 0.0, (1.5 / out.in_features) ** 0.5 * 0.7)
            self.linears[-1] = out
        self.fw = None
        if cfg.fast_weights:
            if cfg.kind not in ("cvsnn", "lif") or not cfg.hidden or cfg.memory_branch:
                raise ValueError("fast_weights needs kind cvsnn or lif, a hidden layer, and no memory_branch")
            self.fw = FastWeights(cfg.hidden[-1], cfg.fast_weights.get("d", 8), cfg.fast_weights.get("heads", 2),
                                  complex_=cfg.fast_weights.get("complex", cfg.kind == "cvsnn"), decay=cfg.fast_weights.get("decay", 0.99),
                                  rule=cfg.fast_weights.get("rule", "hebb"), gate=cfg.fast_weights.get("gate", False),
                                  theta_tag=cfg.fast_weights.get("theta_tag", False), chunk=cfg.fast_weights.get("chunk"))
            n_in = cfg.hidden[-1] + self.fw.out_features
            if cfg.kind == "cvsnn" and not cfg.real_synapses:
                self.linears[-1] = ComplexLinear(n_in, dims[-1], karatsuba=cfg.karatsuba, init=cfg.init)
            else:
                out = nn.Linear(n_in, dims[-1], bias=False)
                nn.init.normal_(out.weight, 0.0, (1.5 / n_in) ** 0.5 * 0.7)
                self.linears[-1] = out
        if cfg.plastic_gate not in (None, "theta", "phase_rel", "random"):
            raise ValueError("plastic_gate must be None, 'theta', 'phase_rel' or 'random'")
        if cfg.plastic_gate == "phase_rel" and cfg.kind != "cvsnn":
            raise ValueError("plastic_gate='phase_rel' needs neurons with a phase (kind='cvsnn')")
        if cfg.plastic_gate and (self.fw is not None or self.mem_neuron is not None or cfg.kwta):
            raise ValueError("plastic_gate is defined for plain layer stacks")
        if cfg.context_gate not in (None, "random", "phase", "phase_shuffled"):
            raise ValueError("context_gate must be None, 'random', 'phase' or 'phase_shuffled'")
        if cfg.context_gate:
            g = torch.Generator().manual_seed(torch.initial_seed() + 4241)   # fixed per seed, does not touch the global RNG
            masks = []
            for n in [neuron_out_features(nr) for nr in self.neurons[:-1]]:
                if cfg.context_gate == "random":
                    m = torch.stack([(torch.randperm(n, generator=g) < n // 2).float() for _ in range(cfg.n_contexts)])
                else:   # "phase"; "phase_shuffled" (I29b): the same phases, given to the tasks in a random order
                    psi = torch.rand(n, generator=g) * 2 * math.pi
                    order = torch.randperm(cfg.n_contexts, generator=g) if cfg.context_gate == "phase_shuffled" else torch.arange(cfg.n_contexts)
                    ph = 2 * math.pi * order.float() / cfg.n_contexts
                    m = (torch.cos(ph.view(-1, 1) - psi.view(1, -1)) > 0).float()
                masks.append(m)                                              # (n_contexts, n)
            self.ctx_masks = masks
        if cfg.feedback not in ("exact", "random"):
            raise ValueError("feedback must be 'exact' or 'random'")
        if cfg.feedback == "random":
            if self.fw is not None or self.mem_neuron is not None:
                raise ValueError("feedback='random' is defined for plain layer stacks (no fast_weights / memory_branch)")
            g = torch.Generator().manual_seed(torch.initial_seed() + 7919)   # fixed per seed, does not touch the global RNG
            for i, lin in enumerate(self.linears):
                if i == 0:
                    continue
                if isinstance(lin, ComplexLinear):
                    for part in ("re", "im"):
                        w = getattr(lin, f"weight_{part}").detach()
                        self.register_buffer(f"fb{i}_{part}", torch.randn(w.shape, generator=g) * w.std())
                elif isinstance(lin, nn.Linear):
                    self.register_buffer(f"fb{i}", torch.randn(lin.weight.shape, generator=g) * lin.weight.detach().std())
                else:
                    raise ValueError("feedback='random' needs Linear / ComplexLinear synapses after the first layer")
        self.device_cfg = DeviceConfig()
        self.hw_generator: torch.Generator | None = None

    def set_hardware(self, cfg: DeviceConfig | None, generator: torch.Generator | None = None) -> None:
        """Map synaptic layers onto a (simulated) crossbar with these non-idealities; None = ideal.
        Pass a seeded generator to sample one fixed chip instance; None resamples on every call."""
        self.device_cfg = cfg or DeviceConfig()
        self.hw_generator = generator

    def _synapse(self, lin: nn.Module, x: torch.Tensor, threshold: float) -> torch.Tensor:
        cfg, g = self.device_cfg, self.hw_generator
        if cfg.ideal or isinstance(lin, (nn.Identity, ChannelEncoder)):
            return lin(x)
        if isinstance(lin, ComplexLinear):
            s = layer_scale(lin.weight_re, lin.weight_im)
            y = lin.forward_with(x, program_weight(lin.weight_re, cfg, g, s), program_weight(lin.weight_im, cfg, g, s))
        else:
            s = layer_scale(lin.weight)
            y = torch.nn.functional.linear(x, program_weight(lin.weight, cfg, g, s))
        return read_out(y, x, s, cfg, threshold, g)

    def _synapse_frozen(self, lin: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """lin(x) with the weights detached (ideal synapses only)."""
        if not self.device_cfg.ideal:
            raise ValueError("plastic_gate needs ideal synapses")
        if isinstance(lin, ComplexLinear):
            return lin.forward_with(x, lin.weight_re.detach(), lin.weight_im.detach())
        return torch.nn.functional.linear(x, lin.weight.detach(), None if lin.bias is None else lin.bias.detach())

    def _plastic_gate(self, neuron: nn.Module, syn: torch.Tensor, theta: torch.Tensor | None) -> torch.Tensor:
        """Real 0/1 gate broadcastable to syn (T, B, N): 1 where the synapses onto that neuron may learn."""
        kind = self.cfg.plastic_gate
        if kind == "random":
            return (torch.rand(syn.shape) < 0.5).float()
        if kind == "theta":
            return (torch.cos(theta) > 0).float()
        with torch.no_grad():   # the neuron's own membrane phase (neurons are stateless, so this equals the real pass)
            _, pot, _ = neuron(syn.detach())
        return (torch.cos(torch.angle(pot) - theta) > 0).float()

    def _random_feedback(self, i: int, h: torch.Tensor) -> torch.Tensor:
        if hasattr(self, f"fb{i}_re"):
            br, bi = getattr(self, f"fb{i}_re"), getattr(self, f"fb{i}_im")
            if not torch.is_complex(h):
                h = torch.complex(h, torch.zeros_like(h))
            re = torch.nn.functional.linear(h.real, br) - torch.nn.functional.linear(h.imag, bi)
            im = torch.nn.functional.linear(h.real, bi) + torch.nn.functional.linear(h.imag, br)
            return torch.complex(re, im)
        return torch.nn.functional.linear(h, getattr(self, f"fb{i}"))

    def forward(self, x: torch.Tensor, trace: bool = False, theta: torch.Tensor | None = None,
                context: int | None = None) -> tuple[torch.Tensor, dict]:
        """Returns (logits, info). theta (T, B): the theta phase for fast weights with theta_tag (I28b). With trace=True, info["layers"] holds per-layer activity for the
        hardware cost models (see cvsnn.hardware.cost)."""
        cfg = self.cfg
        if (cfg.kind in ("lif", "prf", "csilif") or cfg.real_synapses) and torch.is_complex(x) and not cfg.front:
            raise ValueError(f"{cfg.kind} expects real input: apply an input transform ('magnitude' or 'reim') first")
        h = x
        if self.mem_neuron is not None and self.mem_from_input:
            # raw input for the memory branch: [Re, Im] of complex sensors, or the real input as it is
            if cfg.memory_input == "reim":
                x_mem = torch.cat([x.real, x.imag], -1) if torch.is_complex(x) else torch.cat([x, torch.zeros_like(x)], -1)
            else:
                x_mem = x.real if torch.is_complex(x) else x
        gate_theta = None   # the gating rhythm of plastic_gate (P3 step 3); not the theta argument (I28b)
        if cfg.plastic_gate in ("theta", "phase_rel") and torch.is_grad_enabled():
            T_, B_ = x.shape[0], x.shape[1]
            phi = torch.rand(1, B_, 1) * 2 * math.pi
            gate_theta = 2 * math.pi * torch.arange(T_).view(-1, 1, 1) / cfg.theta_period + phi   # (T, B, 1)
        total_spikes = 0.0
        total_units = 0
        last_pot = last_fire = None
        layers = []
        sent = []   # firing rate of every layer whose spikes travel to another layer (with surrogate gradient)
        names = ((CLIF, "clif"), (LIF, "lif"), (PRF, "prf"), (CSiLIF, "csilif"), (ComplexLeakyRNN, "cvnn"), (LRU, "lru"))
        for i, (lin, neuron) in enumerate(zip(self.linears, self.neurons)):
            if trace:
                T, B, n_in = h.shape
                if i == 0:
                    kind = "complex" if torch.is_complex(h) else "real"
                else:
                    kind = {"cvsnn": "complex", "lif": "binary", "cvnn": "complex", "lru": "real",
                            "prf": "binary", "csilif": "binary"}[cfg.kind]
                    if cfg.kind == "cvsnn" and cfg.phase_bins is not None:
                        kind = "phase_binned"
                    if cfg.kind == "cvsnn" and cfg.spike_mode == "binary":
                        kind = "binary"
                active = (h.abs() > 0).sum().item() / B
                layers.append({"n_in": n_in, "n_out": neuron_out_features(neuron), "T": T, "input_kind": kind,
                               "phase_bins": cfg.phase_bins if kind == "phase_binned" else None,
                               "neuron": next(n for t, n in names if isinstance(neuron, t)), "in_events": active,
                               "complex_weights": isinstance(lin, ComplexLinear),
                               "encoder": isinstance(lin, ChannelEncoder)})
            h_in = h
            thr = getattr(neuron, "threshold", 1.0)
            pre = h.detach() if cfg.feedback == "random" and i > 0 else h
            syn = self._synapse(lin, pre, thr)
            if cfg.plastic_gate and torch.is_grad_enabled() and not isinstance(lin, (nn.Identity, ChannelEncoder)):
                g = self._plastic_gate(neuron, syn, gate_theta)
                syn = g * syn + (1 - g) * self._synapse_frozen(lin, pre)   # same value; weight gradient only where g = 1
            if cfg.feedback == "random" and i > 0:
                # value = lin(h); gradient to h only through the fixed random B (feedback alignment); gradient to the
                # weights of lin = postsynaptic signal x presynaptic activity, as with exact feedback
                fb = self._random_feedback(i, h)
                syn = syn + (fb - fb.detach())
            out, pot, fire = neuron(syn)
            if cfg.kwta and i < len(self.neurons) - 1 and not isinstance(lin, ChannelEncoder):
                drive = complex_magnitude(pot) if torch.is_complex(pot) else pot
                k = max(1, round(cfg.kwta * drive.shape[-1]))
                top = drive.detach().topk(k, dim=-1).indices
                keep = torch.zeros_like(drive).scatter_(-1, top, 1.0)
                out = out * (keep.to(out.dtype) if torch.is_complex(out) else keep)
                fire = fire * keep
            if cfg.context_gate and i < len(self.neurons) - 1:
                if context is None:
                    raise ValueError("context_gate needs the task as context")
                m = self.ctx_masks[i][context]
                out = out * (m.to(out.dtype) if torch.is_complex(out) else m)
                fire = fire * m
            total_spikes += float(fire.detach().sum())
            total_units += fire.numel()
            if i < len(self.neurons) - 1:
                sent.append(fire.mean())
            last_pot, last_fire = pot, fire
            h = out
            if self.fw is not None and i == len(self.neurons) - 2:   # fast-weight synapses after the last hidden layer
                h = torch.cat([h, self.fw(h, theta) if self.fw.theta_tag else self.fw(h)], dim=-1)
            if self.mem_neuron is not None and i == len(self.neurons) - 2:   # memory branch next to the core layer
                if self.mem_from_input:
                    h_in = x_mem
                if trace:
                    layers.append({"n_in": h_in.shape[-1], "n_out": self.cfg.memory_branch, "T": h_in.shape[0],
                                   "input_kind": "real" if self.mem_from_input else "binary", "phase_bins": None, "neuron": "lif",
                                   "in_events": (h_in.abs() > 0).sum().item() / h_in.shape[1],
                                   "complex_weights": False, "encoder": False})
                m_out, _, m_fire = self.mem_neuron(self._synapse(self.mem_linear, h_in, self.mem_neuron.threshold))
                total_spikes += float(m_fire.detach().sum())
                total_units += m_fire.numel()
                sent.append(m_fire.mean())
                h = torch.cat([h, m_out], dim=-1)
            if cfg.kind == "lru" and i < len(self.neurons) - 1:
                h = torch.nn.functional.gelu(h)
        if cfg.kind == "cvsnn":
            amp = last_pot.real if cfg.readout == "re" else complex_magnitude(last_pot)
            score = amp / cfg.threshold + 0.5 * last_fire
        elif cfg.kind == "lif":
            score = last_pot / cfg.threshold + 0.5 * last_fire
        elif cfg.kind == "cvnn":
            score = complex_magnitude(h)   # modReLU output of the last layer (uses its bias)
        elif cfg.kind == "prf":
            score = last_pot.real / cfg.threshold + 0.5 * last_fire
        elif cfg.kind == "csilif":
            score = (2 * last_pot.real - self.neurons[-1].vth) + 0.5 * last_fire
        else:
            score = h
        if cfg.out_pairs:   # memory neurons are the last ones of a split layer (CLIF mem_frac = 0.5)
            score = score[..., :self.num_classes] + score[..., self.num_classes:]
        if cfg.readout_last > 0:
            score = score[-cfg.readout_last:]
        logits = cfg.temperature * score.mean(dim=0)
        spiking = cfg.kind in ("cvsnn", "lif", "prf", "csilif")
        info = {"spike_rate": total_spikes / max(1, total_units) if spiking else 0.0}
        if spiking and sent:
            info["transmit_rate"] = torch.stack(sent).mean()   # differentiable: for a spike-rate penalty
        if trace:
            info["layers"] = layers
        return logits, info

    def real_mults_per_step(self) -> int:
        """Real multiplications per sample per time step in the synaptic (linear) layers."""
        total = 0
        for lin, neuron in zip(self.linears, self.neurons):
            if isinstance(lin, ComplexLinear):
                total += lin.real_mults_per_sample()
            elif isinstance(lin, nn.Linear):
                total += lin.in_features * lin.out_features
            elif isinstance(lin, ChannelEncoder):
                total += 0   # sensor front end: fixed gain, no synaptic array
            if isinstance(neuron, LRU):
                total += neuron.B.real_mults_per_sample() + neuron.C.real_mults_per_sample()
        if self.mem_linear is not None:
            total += self.mem_linear.in_features * self.mem_linear.out_features
        return total

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
