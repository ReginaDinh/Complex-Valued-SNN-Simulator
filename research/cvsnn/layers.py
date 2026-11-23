"""Layers: complex linear (Karatsuba / standard), cLIF neuron, and baseline neurons.

Conventions
-----------
* Sequences are time-major: ``(T, B, N)``.
* Complex tensors use torch complex dtypes; autograd applies the conjugate-Wirtinger convention,
  so ``loss.backward()`` gives the correct gradient for real-valued losses of complex parameters.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequencies import FREQ_SETS, frequency_set
from .surrogate import spike

_EPS = 1e-12


def complex_magnitude(z: torch.Tensor) -> torch.Tensor:
    """|z| with a finite gradient at z = 0."""
    return torch.sqrt(z.real * z.real + z.imag * z.imag + _EPS)


class ComplexLinear(nn.Module):
    """y = W x (+ b) with complex W, stored as (W_re, W_im).

    ``karatsuba=True`` uses 3 real matmuls instead of 4. The result is mathematically identical;
    the benefit is fewer multipliers in hardware, not speed on CPU/GPU.

    ``init``:
      * ``"js"``: same scheme as the browser engine: |w| ~ (0.3 + 0.7 U) * sqrt(1.5 / fan_in),
        phase ~ U(-pi, pi).
      * ``"glorot"``: complex Glorot (Trabelsi et al., 2018): Rayleigh magnitude with
        Var(w) = 2 / (fan_in + fan_out), uniform phase.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, karatsuba: bool = True, init: str = "js"):
        super().__init__()
        self.in_features, self.out_features, self.karatsuba = in_features, out_features, karatsuba
        self.weight_re = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_im = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias_re = nn.Parameter(torch.zeros(out_features))
            self.bias_im = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_re", None)
            self.register_parameter("bias_im", None)
        self.reset_parameters(init)

    @torch.no_grad()
    def reset_parameters(self, init: str = "js") -> None:
        shape = self.weight_re.shape
        if init == "js":
            mag = (0.3 + 0.7 * torch.rand(shape)) * math.sqrt(1.5 / self.in_features)
        elif init == "glorot":
            sigma = 1.0 / math.sqrt(self.in_features + self.out_features)
            mag = sigma * torch.sqrt(-2.0 * torch.log(torch.rand(shape).clamp_min(1e-12)))  # Rayleigh
        else:
            raise ValueError(f"unknown init {init!r}")
        phase = (torch.rand(shape) * 2 - 1) * math.pi
        self.weight_re.copy_(mag * torch.cos(phase))
        self.weight_im.copy_(mag * torch.sin(phase))

    @property
    def weight(self) -> torch.Tensor:
        return torch.complex(self.weight_re, self.weight_im)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with(x, self.weight_re, self.weight_im)

    def forward_with(self, x: torch.Tensor, wr: torch.Tensor, wi: torch.Tensor) -> torch.Tensor:
        """Same as forward, with explicit (e.g. hardware-perturbed) weights."""
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        xr, xi = x.real, x.imag
        if self.karatsuba:
            k1 = F.linear(xr, wr)
            k2 = F.linear(xi, wi)
            k3 = F.linear(xr + xi, wr + wi)
            re, im = k1 - k2, k3 - k1 - k2
        else:
            re = F.linear(xr, wr) - F.linear(xi, wi)
            im = F.linear(xr, wi) + F.linear(xi, wr)
        if self.bias_re is not None:
            re, im = re + self.bias_re, im + self.bias_im
        return torch.complex(re, im)

    def real_mults_per_sample(self) -> int:
        return self.in_features * self.out_features * (3 if self.karatsuba else 4)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def linear_recurrence(x: torch.Tensor, beta: torch.Tensor | None = None, *, mag: torch.Tensor | None = None,
                      phase: torch.Tensor | None = None) -> torch.Tensor:
    """u_t = beta * u_{t-1} + x_t (u_{-1} = 0) for all t at once, by FFT convolution with the kernel beta^k.
    x: (T, B, N) complex; give either beta (N,) complex or, preferably, its polar parts mag (N,) in (0, 1) and
    phase (N,): this avoids recovering them with abs/angle, whose gradients are undefined at 0.
    Exact up to float rounding; O(T log T)."""
    T = x.shape[0]
    if mag is None:
        mag, phase = beta.abs(), torch.angle(beta)
    k = torch.arange(T, device=x.device, dtype=mag.dtype).unsqueeze(1)            # (T, 1)
    # beta^k = exp(k (log|beta| + j phase)): a single complex exp has a well-defined gradient even when
    # |beta|^k underflows to 0 (torch.polar's backward divides by the magnitude -> 0/0 = NaN)
    log_beta = torch.complex(torch.log(mag.clamp_min(1e-12)), phase).unsqueeze(0)   # (1, N)
    kernel = torch.exp(k.to(log_beta.dtype) * log_beta)                             # (T, N)
    n = 2 * T
    X = torch.fft.fft(x, n=n, dim=0)
    K = torch.fft.fft(kernel.to(x.dtype), n=n, dim=0).unsqueeze(1)                  # (n, 1, N)
    return torch.fft.ifft(X * K, dim=0)[:T]


def ring_magnitudes(num_features: int, lo: float, hi: float) -> torch.Tensor:
    """Raw logits of |beta| drawn uniformly on the ring r^2 ~ U(lo^2, hi^2) (LRU-style)."""
    r = torch.sqrt(torch.rand(num_features) * (hi**2 - lo**2) + lo**2).clamp(1e-4, 0.9999)
    return torch.log(r / (1 - r))


def init_leak(num_features: int, leak_mag: float, leak_phase: float,
              mag_range: tuple[float, float] | None, phase_range: tuple[float, float] | None,
              phase_dist: str = "uniform", phase_bits: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-neuron (raw |beta| logits, omega). Ranges give heterogeneous neurons (all four closest papers
    initialise frequencies per neuron); otherwise every neuron gets (leak_mag, leak_phase).
    |beta| is drawn uniformly on the ring r^2 ~ U(r_min^2, r_max^2) (LRU-style). ``phase_dist`` is one of
    ``cvsnn.frequencies.FREQ_SETS`` (``phase_bits`` for the hardware-grid sets 'quantized' and 'sidon')."""
    if mag_range is None:
        raw = torch.full((num_features,), _logit(min(leak_mag, 0.999)))
    else:
        raw = ring_magnitudes(num_features, *mag_range)
    if phase_range is None:
        phase = torch.full((num_features,), float(leak_phase))
    else:
        lo, hi = phase_range
        if phase_dist in ("uniform", "log"):   # float32 draws, as in the E-MEM / E-CL v2 runs (reproducible)
            u = torch.rand(num_features)
            phase = torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if phase_dist == "log" else lo + u * (hi - lo)
        elif phase_dist in FREQ_SETS:
            phase = frequency_set(phase_dist, num_features, lo, hi, bits=phase_bits).float()
        else:
            raise ValueError(f"unknown phase_dist {phase_dist!r}; expected one of {FREQ_SETS}")
    return raw, phase


class Urgency(nn.Module):
    """H5: a deadline-dependent modulation with one learnable strength c. With g_t = (min(t + 1, D) / D)^2 (0 at the
    start, 1 at the deadline D), the neuron sees one of (Cisek et al. 2009 urgency gating; Drugowitsch et al. 2012
    collapsing bounds; 'damping' is the user's idea of a convergence pressure near the deadline):
      'damping'  : |beta_t| = |beta| exp(-c g_t)   (c >= 0; the state settles as the deadline approaches)
      'threshold': V_th,t = V_th (1 - c g_t)       (0 <= c <= 0.9; collapsing bound)
      'gain'     : I_t (1 + c g_t)                 (c >= 0; growing urgency signal)
      'damping_const': |beta_t| = |beta| exp(-c)       (control: the same extra leak at every step, no deadline schedule)"""

    KINDS = ("damping", "threshold", "gain", "damping_const")

    def __init__(self, kind: str, deadline: int, init: float = 0.1):
        super().__init__()
        if kind not in self.KINDS or deadline < 1:
            raise ValueError(f"urgency must be one of {self.KINDS} with a deadline >= 1")
        self.kind, self.deadline = kind, deadline
        raw = _logit(init / 0.9) if kind == "threshold" else math.log(math.expm1(init))
        self.raw = nn.Parameter(torch.tensor(float(raw)))

    def strength(self) -> torch.Tensor:
        return 0.9 * torch.sigmoid(self.raw) if self.kind == "threshold" else F.softplus(self.raw)

    def factors(self, t: int) -> tuple:
        """(input gain, leak factor, threshold scale) at step t."""
        g = 1.0 if self.kind == "damping_const" else (min(t + 1, self.deadline) / self.deadline) ** 2
        c = self.strength()
        one = torch.ones((), dtype=c.dtype)
        if self.kind == "gain":
            return 1 + c * g, one, one
        if self.kind in ("damping", "damping_const"):
            return one, torch.exp(-c * g), one
        return one, one, 1 - c * g


class CLIF(nn.Module):
    """Complex leaky integrate-and-fire neuron (cLIF), unrolled over time (full BPTT).

        U_pre[t] = beta * U[t-1] + I[t],        beta = |beta| e^{j omega}
        S[t]     = H(|U_pre[t]| - V_th) * e^{j arg U_pre[t]}       (spike_mode="phasor")
                   H * e^{j q_K(arg U_pre[t])}                     (phasor + phase_bins=K)
                   H(|U_pre[t]| - V_th) * U_pre[t]                 (spike_mode="graded")
                   H(|U_pre[t]| - V_th)  or  H(Re U_pre[t] - V_th) (spike_mode="binary", binary_on="mag" | "re")
                   K real 0/1 lines, the event on the line of its phase slot   (spike_mode="slots", phase_bins=K)
        A binary spike is a real 0/1 event: the phase stays inside the neuron (as in PRF / C-SiLIF), the
        next layer's complex weights turn it into a complex current. With binary_on="re" the spike time
        within the rotation still reflects the phase, at the resolution of one step.
        soft reset: U[t] = U_pre[t] - V_th * H * e^{j arg U_pre[t]}  (keeps the phase)
        hard reset: U[t] = U_pre[t] * (1 - H)
        no reset:   U[t] = U_pre[t]  (linear state; spikes are only a read-out, as in PRF / C-SiLIF;
                    computed for all t at once by FFT convolution)
        brf reset:  no subtraction. A refractory trace q[t] = g q[t-1] + H raises the threshold to V_th + q
                    and temporarily increases damping, beta_t = beta * exp(-s q[t-1]) (phase unchanged),
                    adapted from the balanced resonate-and-fire neuron (Higuchi et al., ICML 2024, Eq. 9-12).
                    In BRF the extra damping enters as delta * q with time step delta = 0.01, hence s = 0.01

    ``phase_bins=K`` models a *timing code* for neuromorphic hardware: the spike is a binary event sent
    in one of K sub-step time slots, slot k = round(arg U * K / 2pi) mod K. A receiver that accumulates
    each slot separately and combines them with e^{-j 2 pi k / K} sees exactly e^{j q_K(theta)} with the
    phase quantised to multiples of 2pi/K, so synapses only add (see cvsnn.hardware.cost). Training uses a
    straight-through gradient through the quantiser. The neuron's own reset uses its exact phase.

    Same forward dynamics as ``ComplexLIFNeuron`` in ``src/simulationEngine.ts``.
    |beta| is parameterised through a sigmoid so it stays in (0, 1) while learning. ``learn_phase=False``
    keeps the frequencies omega fixed at their initial set (H6: compare frequency sets) while |beta| learns.

    ``mem_frac=p`` splits the layer into two populations (H6c): the last round(p N) neurons are *memory* neurons
    (no reset, |beta| on the ring ``leak_mag_range``, frequencies from ``leak_phase_range`` / ``phase_dist``), the
    others are *processing* neurons (the given soft or hard ``reset``, homogeneous ``leak_mag`` / ``leak_phase``).
    Both populations emit the same kind of spike.
    """

    def __init__(
        self,
        num_features: int,
        threshold: float = 1.0,
        leak_mag: float = 0.85,
        leak_phase: float = 0.15,
        learn_leak: bool = True,
        reset: str = "soft",
        detach_reset: bool = True,
        surrogate: str = "von_mises",
        sharpness: float = 2.5,
        spike_mode: str = "phasor",
        phase_bins: int | None = None,
        leak_mag_range: tuple[float, float] | None = None,
        leak_phase_range: tuple[float, float] | None = None,
        phase_dist: str = "uniform",
        brf_gamma: float = 0.9,
        brf_strength: float = 0.01,
        binary_on: str = "mag",
        learn_phase: bool = True,
        phase_bits: int | None = None,
        mem_frac: float | None = None,
        parallel_reset: bool = False,
        urgency: str | None = None,
        deadline: int | None = None,
        rotate_frac: float | None = None,
        input_norm: bool = False,
    ):
        super().__init__()
        self.parallel_reset = parallel_reset
        # I70: LRU-style input normalisation (Orvieto et al. 2023): the input is scaled by sqrt(1 - |beta|^2), so a
        # neuron with |beta| near 1 no longer accumulates ~1/(1 - |beta|) times more than a fast one (equal state
        # variance for white-noise input); computed from the current |beta| and detached (a scale, not a gradient path)
        self.input_norm = input_norm
        if urgency is not None and (reset not in ("soft", "hard") or parallel_reset):
            raise ValueError("urgency needs a step-by-step soft or hard reset (no parallel_reset)")
        self.urgency = Urgency(urgency, deadline) if urgency else None
        if phase_bins is not None and (phase_bins < 1 or spike_mode not in ("phasor", "slots")):
            raise ValueError("phase_bins needs spike_mode='phasor' or 'slots' and at least 1 bin (1 = no phase in the spike)")
        if spike_mode == "slots" and not phase_bins:
            raise ValueError("spike_mode='slots' needs phase_bins")
        self.phase_bins = phase_bins
        if reset not in ("soft", "hard", "none", "brf"):
            raise ValueError("reset must be 'soft', 'hard', 'none' or 'brf'")
        self.brf_gamma, self.brf_strength = brf_gamma, brf_strength
        if spike_mode not in ("phasor", "graded", "binary", "slots"):
            raise ValueError("spike_mode must be 'phasor', 'graded', 'binary' or 'slots'")
        if binary_on not in ("mag", "re") or (binary_on == "re" and spike_mode != "binary"):
            raise ValueError("binary_on must be 'mag' or 're' ('re' only with spike_mode='binary')")
        self.num_features, self.threshold = num_features, max(0.05, threshold)
        self.reset, self.detach_reset = reset, detach_reset
        self.surrogate, self.sharpness, self.spike_mode, self.binary_on = surrogate, sharpness, spike_mode, binary_on
        if mem_frac is None:
            raw_mag, raw_phase = init_leak(num_features, leak_mag, leak_phase, leak_mag_range, leak_phase_range, phase_dist, phase_bits)
            self.register_buffer("reset_mask", None)
        else:
            if reset not in ("soft", "hard") or leak_mag_range is None or leak_phase_range is None:
                raise ValueError("mem_frac needs reset 'soft' or 'hard' (processing neurons) and leak ranges (memory neurons)")
            n_mem = round(mem_frac * num_features)
            raw_mag, raw_phase = init_leak(num_features, leak_mag, leak_phase, None, None)
            if n_mem:
                m_mag, m_phase = init_leak(n_mem, leak_mag, leak_phase, leak_mag_range, leak_phase_range, phase_dist, phase_bits)
                raw_mag[num_features - n_mem:], raw_phase[num_features - n_mem:] = m_mag, m_phase
            mask = torch.ones(num_features)
            mask[num_features - n_mem:] = 0.0                  # 1 = processing neuron (resets), 0 = memory neuron
            self.register_buffer("reset_mask", mask)
        if learn_leak:
            self.leak_mag_raw = nn.Parameter(raw_mag)
        else:
            self.register_buffer("leak_mag_raw", raw_mag)
        if learn_leak and learn_phase:
            self.leak_phase = nn.Parameter(raw_phase)
        else:
            self.register_buffer("leak_phase", raw_phase)
        # I25: only a random fraction of the neurons rotate; the others keep omega = 0 (their phase is masked, so it
        # cannot learn to rotate either)
        if rotate_frac is None:
            self.register_buffer("rot_mask", None)
        else:
            mask = torch.zeros(num_features)
            mask[torch.randperm(num_features)[:round(rotate_frac * num_features)]] = 1.0
            self.register_buffer("rot_mask", mask)

    @property
    def phase(self) -> torch.Tensor:
        """The rotation per step omega of every neuron (leak_phase, masked by rot_mask when only some rotate)."""
        return self.leak_phase if self.rot_mask is None else self.leak_phase * self.rot_mask

    @property
    def beta(self) -> torch.Tensor:
        return torch.polar(torch.sigmoid(self.leak_mag_raw), self.phase)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (T, B, N) complex input current.

        Returns ``(spikes, u_pre, fired)``: complex spikes, pre-reset potentials (both (T, B, N))
        and the real 0/1 spike indicator H (with surrogate gradient).
        """
        T, B, N = x.shape
        if not torch.is_complex(x):   # e.g. binary spikes or real sensor values feeding an encoder neuron
            x = torch.complex(x, torch.zeros_like(x))
        if self.input_norm:
            g = torch.sqrt(torch.clamp(1 - torch.sigmoid(self.leak_mag_raw.detach()) ** 2, min=1e-6))
            x = x * g.to(x.real.dtype)
        beta = self.beta.to(x.dtype)
        if self.reset == "none":
            mag = torch.sigmoid(self.leak_mag_raw).to(x.real.dtype)
            u_pre = linear_recurrence(x, mag=mag, phase=self.phase.to(x.real.dtype))   # exact, parallel
            m = complex_magnitude(u_pre)
            h = spike(self._drive(u_pre, m), self.threshold, self.surrogate, self.sharpness)
            return self._out(h, u_pre, m), u_pre, h
        if self.reset == "soft" and self.detach_reset and self.parallel_reset:
            return self._soft_reset_parallel(x)
        u = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        q = torch.zeros(B, N, dtype=x.real.dtype, device=x.device)     # BRF refractory trace
        spikes, pots, fires = [], [], []
        for t in range(T):
            if self.reset == "brf":
                u_pre = beta * torch.exp(-self.brf_strength * q).to(x.dtype) * u + x[t]
                m = complex_magnitude(u_pre)
                h = spike(self._drive(u_pre, m) - q, self.threshold, self.surrogate, self.sharpness)   # threshold V_th + q
                s = self._out(h, u_pre, m)
                u = u_pre
                q = self.brf_gamma * q + (h.detach() if self.detach_reset else h)
                spikes.append(s)
                pots.append(u_pre)
                fires.append(h)
                continue
            if self.urgency is None:
                u_pre = beta * u + x[t]
                thr_scale = 1.0
            else:
                gain, leak, thr_scale = self.urgency.factors(t)
                u_pre = beta * leak.to(x.dtype) * u + x[t] * gain.to(x.dtype)
            m = complex_magnitude(u_pre)
            h = spike(self._drive(u_pre, m) / thr_scale, self.threshold, self.surrogate, self.sharpness)
            # soft reset moves the state back along the direction in which the threshold was crossed
            phasor = u_pre / m if self.binary_on == "mag" else torch.ones_like(u_pre)
            s = self._out(h, u_pre, m)
            hr = h if self.reset_mask is None else h * self.reset_mask   # memory neurons of a split layer never reset
            if self.reset == "soft":
                sub = self.threshold * thr_scale * hr * phasor
                u = u_pre - (sub.detach() if self.detach_reset else sub)
            else:
                keep = 1.0 - (hr.detach() if self.detach_reset else hr)
                u = u_pre * keep
            spikes.append(s)
            pots.append(u_pre)
            fires.append(h)
        return torch.stack(spikes), torch.stack(pots), torch.stack(fires)

    def _soft_reset_parallel(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Soft reset with a detached reset term, computed as: (1) one pass without autograd that finds the exact
        spikes and reset vectors r_t; (2) u_pre = linear recurrence of (x_t - beta r_{t-1}) by FFT, with gradients
        (u_pre_t = beta (u_pre_{t-1} - r_{t-1}) + x_t; beta keeps its gradient, r does not).
        Because r is detached, the values and the gradients equal those of the step-by-step loop (up to float
        rounding), at a fraction of the cost on long sequences. Forward spike values come from pass (1), so the
        reset and the emitted spikes always agree; their gradient is the surrogate at u_pre from pass (2)."""
        T, B, N = x.shape
        mag = torch.sigmoid(self.leak_mag_raw).to(x.real.dtype)
        phase = self.phase.to(x.real.dtype)
        with torch.no_grad():
            beta = torch.polar(mag, phase).to(x.dtype)
            u = torch.zeros(B, N, dtype=x.dtype, device=x.device)
            fired, resets = [], []
            for t in range(T):
                u_pre = beta * u + x[t]
                m = complex_magnitude(u_pre)
                h = (self._drive(u_pre, m) >= self.threshold).to(m.dtype)
                hr = h if self.reset_mask is None else h * self.reset_mask
                r = self.threshold * hr * (u_pre / m if self.binary_on == "mag" else torch.ones_like(u_pre))
                u = u_pre - r
                fired.append(h)
                resets.append(r)
            H = torch.stack(fired)
            R = torch.stack(resets)
            R = torch.cat([torch.zeros_like(R[:1]), R[:-1]])        # r_{t-1} enters u_pre_t
        u_pre = linear_recurrence(x - torch.polar(mag, phase).to(x.dtype) * R, mag=mag, phase=phase)
        m = complex_magnitude(u_pre)
        h_sur = spike(self._drive(u_pre, m), self.threshold, self.surrogate, self.sharpness)
        h = H + (h_sur - h_sur.detach())
        return self._out(h, u_pre, m), u_pre, h

    def _drive(self, u_pre: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Quantity compared with the threshold: |U| (default) or Re U (binary_on='re')."""
        return u_pre.real if self.binary_on == "re" else m

    def _out(self, h: torch.Tensor, u_pre: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Output sent to the next layer: the real 0/1 event itself, the event times the spike's value, or
        (spike_mode='slots') the timing code as K real 0/1 lines per neuron: the event appears on the line of
        its slot k = round(arg U K / 2 pi) mod K, so a receiver with real weights has one weight per slot."""
        if self.spike_mode == "binary":
            return h
        if self.spike_mode == "slots":
            K = self.phase_bins
            k = torch.remainder(torch.round(torch.angle(u_pre.detach()) * K / (2 * math.pi)), K).long()
            lines = torch.nn.functional.one_hot(k, K).to(h.dtype) * h.unsqueeze(-1)
            return lines.flatten(-2)
        return h * self._emit(u_pre, m)

    def _emit(self, u_pre: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Value carried by a spike: unit phasor, phase-quantised phasor (timing code) or graded U."""
        if self.spike_mode == "graded":
            return u_pre
        phasor = u_pre / m
        if self.phase_bins is None:
            return phasor
        step = 2 * math.pi / self.phase_bins
        theta_q = torch.round(torch.angle(u_pre.detach()) / step) * step
        return phasor + (torch.polar(torch.ones_like(m), theta_q).to(phasor.dtype) - phasor).detach()


class FastWeights(nn.Module):
    """P1b: Hebbian fast weights (plastic synapses that change while the network runs). From the presynaptic
    activity h_t the layer forms a key k_t, a value v_t and a query q_t (learned projections; keys and queries
    normalised per head) and updates, per head, a synaptic matrix
        A_t = lambda A_{t-1} + v_t k_t^H        (k^H: conjugate transpose; real codes: k^T)
    and outputs the current y_t = A_t q_t. lambda (one per head, learnable) sets how fast the synapses forget.
    The projections are learned by BPTT; A itself is never a parameter. This is a linear-attention /
    fast-weight-programmer layer; with complex codes the stored associations are phase relations."""

    def __init__(self, in_features: int, d: int = 8, heads: int = 2, complex_: bool = False, decay: float = 0.99,
                 rule: str = "hebb", gate: bool = False, theta_tag: bool = False, chunk: int | None = None):
        """rule 'hebb': A <- lambda A + v k^H. rule 'delta' (Schlag et al. 2021): A <- lambda A + beta (v - A k) k^H,
        which replaces the value stored under a key instead of piling up. gate: a learned write strength
        beta_t = sigmoid(w . |h_t| + b) per head (b starts at -2: write rarely unless the input says so).
        theta_tag (I28b, a theta-gamma phase code): keys and queries are rotated by harmonics of a theta phase theta_t
        passed to forward: complex, dimension r is multiplied by e^{j r theta} (r = 0 .. d-1, a ladder from theta up
        to gamma); real, dimension pair p is rotated by (p + 1) theta (RoPE-style), a leftover dimension is not
        rotated. A key written at phase theta then matches a query made at the same phase, whatever the content.
        chunk (speed): None = the step-by-step loop (default, reproduces earlier runs bit for bit); an integer C =
        the same recurrence computed C steps at a time (see _scan_chunks); equal up to float rounding."""
        super().__init__()
        if rule not in ("hebb", "delta"):
            raise ValueError("rule must be 'hebb' or 'delta'")
        if chunk is not None and chunk < 1:
            raise ValueError("chunk must be None or a positive number of steps")
        self.d, self.heads, self.complex_, self.rule, self.theta_tag, self.chunk = d, heads, complex_, rule, theta_tag, chunk
        hd = d * heads
        make = (lambda: ComplexLinear(in_features, hd, karatsuba=False, init="glorot")) if complex_ else (lambda: nn.Linear(in_features, hd, bias=False))
        self.k, self.v, self.q = make(), make(), make()
        self.decay_raw = nn.Parameter(torch.full((heads,), _logit(decay)))
        self.gate = None
        if gate:
            self.gate = nn.Linear(in_features, heads)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.0)

    @property
    def out_features(self) -> int:
        return self.d * self.heads

    def _tag(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Rotate (T, B, H, d) keys or queries by harmonics of theta (T, B); norms are unchanged."""
        th = theta.view(*theta.shape, 1, 1).to(z.real.dtype)
        if self.complex_:
            r = torch.arange(self.d, dtype=th.dtype)
            return z * torch.polar(torch.ones_like(th * r), th * r).to(z.dtype)
        n = self.d // 2
        x, y = z[..., 0:2 * n:2], z[..., 1:2 * n:2]
        ang = th * torch.arange(1, n + 1, dtype=th.dtype)
        c, s_ = torch.cos(ang), torch.sin(ang)
        rot = torch.stack([x * c - y * s_, x * s_ + y * c], -1).flatten(-2)
        return torch.cat([rot, z[..., 2 * n:]], -1)

    def _scan_chunks(self, k, q, v, beta, lam) -> torch.Tensor:
        """The recurrence of forward, C steps at a time. Inside a chunk that starts from state S (state before it),
        with 0-based steps t and write vectors u_t (u_t = beta_t (v_t - S_{t-1} k_t) for the delta rule):
            S_{t-1} = lam^t S + sum_{j<t} lam^{t-1-j} u_j k_j^H
        so the delta writes solve one unit lower-triangular system per chunk,
            u_t + beta_t sum_{j<t} lam^{t-1-j} (k_j^H k_t) u_j = beta_t (v_t - lam^t S k_t),
        and  y_t = lam^{t+1} S q_t + sum_{j<=t} lam^{t-j} (k_j^H q_t) u_j,  S_end = lam^C S + sum_j lam^{C-1-j} u_j k_j^H.
        Only non-negative powers of lam appear. k, q, v: (T, B, H, d); beta: (T, B, H) or None; lam: (H,).
        Returns (T, B, H, d)."""
        T, B, H, d = v.shape
        K, Q, V = (z.permute(1, 2, 0, 3) for z in (k, q, v))                      # (B, H, T, d)
        bt = None if beta is None else beta.permute(1, 2, 0).to(V.dtype)          # (B, H, T)
        loglam = torch.log(lam).view(1, H, 1, 1)
        S = torch.zeros(B, H, d, d, dtype=V.dtype, device=V.device)
        tr = lambda z: z.transpose(-1, -2)
        ys = []
        for s0 in range(0, T, self.chunk):
            Kc, Qc, Vc = K[:, :, s0:s0 + self.chunk], Q[:, :, s0:s0 + self.chunk], V[:, :, s0:s0 + self.chunk]
            C = Kc.shape[2]
            t = torch.arange(C, dtype=lam.dtype, device=V.device)
            diff = t.view(-1, 1) - t.view(1, -1)                                  # t - j
            pw = lambda e, mask: (torch.exp(e * loglam) * mask).to(V.dtype)       # lam^e where mask, else 0
            L1 = pw(diff - 1, (diff >= 1).to(lam.dtype))                          # lam^{t-1-j}, j < t
            L2 = pw(diff, (diff >= 0).to(lam.dtype))                              # lam^{t-j},   j <= t
            p0 = torch.exp(t.view(1, 1, -1, 1) * loglam).to(V.dtype)              # lam^t
            Kh = torch.conj(Kc) if self.complex_ else Kc
            if self.rule == "delta":
                KK = Kc @ tr(Kh)                                                  # [t, j] = k_j^H k_t
                b = torch.ones_like(Vc[..., 0]) if bt is None else bt[:, :, s0:s0 + C]
                Lmat = torch.eye(C, dtype=V.dtype, device=V.device) + b.unsqueeze(-1) * KK * L1
                R = b.unsqueeze(-1) * (Vc - p0 * (Kc @ tr(S)))
                U = torch.linalg.solve_triangular(Lmat, R, upper=False, unitriangular=True)
            else:
                U = Vc if bt is None else bt[:, :, s0:s0 + C].unsqueeze(-1) * Vc
            ys.append(p0 * torch.exp(loglam).to(V.dtype) * (Qc @ tr(S)) + (Qc @ tr(Kh) * L2) @ U)
            w = torch.exp((C - 1 - t).view(1, 1, -1, 1) * loglam).to(V.dtype)     # lam^{C-1-j}
            S = torch.exp(C * loglam).to(V.dtype) * S + tr(U * w) @ Kh
        return torch.cat(ys, 2).permute(2, 0, 1, 3)

    def forward(self, h: torch.Tensor, theta: torch.Tensor | None = None) -> torch.Tensor:
        T, B, _ = h.shape
        if self.theta_tag and theta is None:
            raise ValueError("theta_tag needs the theta phase (T, B)")
        if self.complex_ and not torch.is_complex(h):
            h = torch.complex(h, torch.zeros_like(h))
        if not self.complex_ and torch.is_complex(h):
            raise ValueError("real fast weights need real input")
        H, d = self.heads, self.d
        unit = lambda z: z / (torch.sqrt((z.real ** 2 + z.imag ** 2).sum(-1, keepdim=True) + 1e-6) if torch.is_complex(z)
                              else torch.sqrt((z ** 2).sum(-1, keepdim=True) + 1e-6))
        k = unit(self.k(h).view(T, B, H, d))
        q = unit(self.q(h).view(T, B, H, d))
        v = self.v(h).view(T, B, H, d)
        k_raw, q_raw = k, q
        if self.theta_tag:
            k, q = self._tag(k, theta), self._tag(q, theta)
        lam = torch.sigmoid(self.decay_raw).view(1, H, 1, 1)
        if self.gate is not None:
            mag = torch.sqrt(h.real ** 2 + h.imag ** 2 + 1e-12) if torch.is_complex(h) else h
            beta = torch.sigmoid(self.gate(mag))              # (T, B, H)
        self.last = {"k": k.detach(), "q": q.detach(), "beta": beta.detach() if self.gate is not None else None,   # for analysis (I21)
                     "v": v.detach(), "k_raw": k_raw.detach(), "q_raw": q_raw.detach()}                           # (I28, before the theta tag)
        if getattr(self, "keep_live", False):   # I56 (mechanism test): the keys with their graph, for a training penalty
            self.live_k = k
        if self.chunk:
            return self._scan_chunks(k, q, v, beta if self.gate is not None else None, torch.sigmoid(self.decay_raw)).reshape(T, B, H * d)
        A = torch.zeros(B, H, d, d, dtype=v.dtype, device=v.device)
        ys = []
        for t in range(T):
            kt = torch.conj(k[t]) if self.complex_ else k[t]
            write = v[t] if self.rule == "hebb" else v[t] - (A @ k[t].unsqueeze(-1)).squeeze(-1)
            if self.gate is not None:
                write = write * beta[t].unsqueeze(-1).to(write.dtype)
            A = lam.to(A.dtype) * A + write.unsqueeze(-1) * kt.unsqueeze(-2)
            if getattr(self, "keep_A_at", None) == t:   # I45 (analysis): keep the synaptic matrix after step t
                self.last["A"] = A.detach()
            ys.append((A @ q[t].unsqueeze(-1)).squeeze(-1))
        return torch.stack(ys).reshape(T, B, H * d)


class LIF(nn.Module):
    """Real-valued LIF baseline: V_pre = beta V + I, S = H(V_pre - V_th), soft/hard/no reset."""

    def __init__(
        self,
        num_features: int,
        threshold: float = 1.0,
        leak: float = 0.85,
        learn_leak: bool = True,
        reset: str = "soft",
        detach_reset: bool = True,
        surrogate: str = "von_mises",
        sharpness: float = 2.5,
        leak_range: tuple[float, float] | None = None,
        parallel_reset: bool = False,
        urgency: str | None = None,
        deadline: int | None = None,
        adapt: dict | None = None,
    ):
        super().__init__()
        self.parallel_reset = parallel_reset
        if urgency is not None and (reset not in ("soft", "hard") or parallel_reset):
            raise ValueError("urgency needs a step-by-step soft or hard reset (no parallel_reset)")
        self.urgency = Urgency(urgency, deadline) if urgency else None
        self.threshold, self.reset, self.detach_reset = max(0.05, threshold), reset, detach_reset
        self.surrogate, self.sharpness = surrogate, sharpness
        # leak_range: per-neuron time constants on the same ring as CLIF (fair control for heterogeneous cLIF)
        raw = torch.full((num_features,), _logit(min(leak, 0.999))) if leak_range is None else ring_magnitudes(num_features, *leak_range)
        if learn_leak:
            self.leak_raw = nn.Parameter(raw)
        else:
            self.register_buffer("leak_raw", raw)
        # I61 (adLIF, Bittar & Garner 2022 / Baronig et al. 2025): an adaptation current w coupled to the membrane,
        #   u_t = alpha u_{t-1} + I_t - w_{t-1},   w_t = beta_w w_{t-1} + a u_{t-1} + b z_{t-1}
        # (u_{t-1}: membrane after reset). With alpha = beta_w = r cos(omega), a = r^2 sin^2(omega), b = 0 this is exactly
        # the subthreshold dynamics of a cLIF neuron with |beta| = r, rotation omega, real input and spikes on Re U
        # (w = sin(omega) * Im U). adapt = {"omega": init rotation, "b": init spike-triggered adaptation}; alpha, beta_w,
        # a and b are then learned freely (the cLIF keeps alpha = beta_w and a = r^2 sin^2 omega tied to (r, omega)).
        self.adapt = adapt is not None
        self.adapt_style = (adapt or {}).get("style", "matched")
        if self.adapt and self.adapt_style == "bittar":
            # I61b: the standard adLIF of Bittar & Garner (2022), also the adLIF baseline of Baronig et al. (2025):
            #   u_t = alpha (u_{t-1} - theta s_{t-1}) + (1 - alpha) (g I_t - w_{t-1}),   w_t = beta (w_{t-1}) + (1 - beta) (a u_{t-1} + b s_{t-1})
            # with alpha in [e^-1/5, e^-1/25], beta in [e^-1/30, e^-1/120], a in [-1, 1], b in [0, 2] (clamped), initialised
            # uniformly in those ranges. g = 1 / (1 - alpha_init) is a fixed per-neuron input gain that undoes the (1 - alpha)
            # input scaling at initialisation, so that the input weights start at the same scale as the LIF (otherwise
            # the network is silent at the start on this task's sparse pulses); it is not learned.
            self.lim = {"alpha": (math.exp(-1 / 5), math.exp(-1 / 25)), "beta": (math.exp(-1 / 30), math.exp(-1 / 120)),
                        "a": (-1.0, 1.0), "b": (0.0, 2.0)}
            u = lambda lo, hi: lo + (hi - lo) * torch.rand(num_features)
            al = u(*self.lim["alpha"])
            self.alpha_p = nn.Parameter(al)
            self.beta_p = nn.Parameter(u(*self.lim["beta"]))
            self.a = nn.Parameter(u(*self.lim["a"]))
            self.b = nn.Parameter(u(*self.lim["b"]))
            self.register_buffer("in_gain", 1.0 / (1.0 - al))
            self.leak_raw = nn.Parameter(torch.zeros(1), requires_grad=False) if learn_leak else self.leak_raw   # unused
        elif self.adapt and self.adapt_style == "se":
            # I65: SE-adLIF (Baronig et al. 2025, Nat. Commun.): adLIF discretised with symplectic Euler, i.e. the adaptation
            # current uses the membrane AFTER this step's update and reset:
            #   u^_t = alpha u_{t-1} + (1 - alpha) (g I_t - w_{t-1}),  s_t = H(u^_t - theta),  u_t = u^_t (1 - s_t)  (reset to 0)
            #   w_t = beta w_{t-1} + (1 - beta) (a u_t + b s_t)
            # with alpha = exp(-1/tau_u), beta = exp(-1/tau_w) and their reparameterisation (SHD setting, Table 7):
            # tau_u = 5 + 20 th_u, tau_w = 60 + 240 th_w (th in [0, 1]), a = q a^ (a^ in [0, 1]), b = q b^ (b^ in [0, 2]), q = 120;
            # all initialised uniformly in their ranges and clamped in the forward pass. g = 1 / (1 - alpha_init): the same
            # fixed input gain as the "bittar" style (keeps the input weights at the LIF scale at initialisation).
            u = lambda lo, hi: lo + (hi - lo) * torch.rand(num_features)
            self.th_u, self.th_w = nn.Parameter(u(0.0, 1.0)), nn.Parameter(u(0.0, 1.0))
            self.a_hat, self.b_hat = nn.Parameter(u(0.0, 1.0)), nn.Parameter(u(0.0, 2.0))
            self.q = 120.0
            self.register_buffer("in_gain", 1.0 / (1.0 - torch.exp(-1.0 / (5.0 + 20.0 * self.th_u.detach()))))
            self.leak_raw = nn.Parameter(torch.zeros(1), requires_grad=False) if learn_leak else self.leak_raw   # unused
        elif self.adapt and self.adapt_style == "brf":
            # I65: balanced resonate-and-fire (Higuchi, Kairat, Bohte & Otte 2024), complex state u = x + i y, forward Euler
            # with step delta (here delta = 1 step, parameters per step):
            #   u_t = u_{t-1} + delta ((b_t + i omega) u_{t-1} + I_t),   b_t = p(omega) - b' - q_{t-1},
            #   p(omega) = (-1 + sqrt(1 - (delta omega)^2)) / delta      (divergence boundary: |1 + delta (p + i omega)| = 1)
            #   s_t = H(Re u_t - theta - q_{t-1}),   q_t = gamma q_{t-1} + s_t,  gamma = 0.9   (no reset: smooth reset + refractory)
            # omega and b' are learned (|.| taken in the forward pass, as in their code); initialised as their SHD run
            # (github.com/AdaptiveAILab/brf-neurons, experiments/SHD/shd_train.py: omega ~ U(5, 10), b' ~ U(2, 3), delta = 0.01;
            # the paper's table says 4 ms), i.e. per step omega ~ U(0.05, 0.10), b' ~ U(0.02, 0.03). Input enters with gain 1
            # (their delta-scaled input is a rescaling of the input weights), as for the cLIF.
            u = lambda lo, hi: lo + (hi - lo) * torch.rand(num_features)
            self.omega_p, self.boff_p = nn.Parameter(u(0.05, 0.10)), nn.Parameter(u(0.02, 0.03))
            self.gamma = 0.9
            # fix 2026-11-19: in their update the refractory term q enters the damping b, which is multiplied by delta
            # (u + delta b u), so per step it acts as delta * q = 0.01 q; the first version used q unscaled (100x too
            # strong: with frequent spikes 1 + b reached about -9 and the state blew up). Results before the fix (I65,
            # I67, I68 BRF rows, variants brf_*) are invalid; rerun as brfv2_*.
            self.q_scale = 0.01
            self.leak_raw = nn.Parameter(torch.zeros(1), requires_grad=False) if learn_leak else self.leak_raw   # unused
        elif self.adapt:
            r = torch.sigmoid(raw)
            om = float(adapt.get("omega", 0.15))
            c = r * math.cos(om)
            self.leak_raw = nn.Parameter(torch.log(c / (1 - c))) if learn_leak else self.leak_raw   # alpha
            self.wleak_raw = nn.Parameter(torch.log(c / (1 - c)))                                  # beta_w
            self.a = nn.Parameter((r * math.sin(om)) ** 2)
            self.b = nn.Parameter(torch.full_like(r, float(adapt.get("b", 0.0))))

    def _forward_bittar(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, N = x.shape
        c = lambda p, k: torch.clamp(p, *self.lim[k])
        alpha, beta, a, b = c(self.alpha_p, "alpha"), c(self.beta_p, "beta"), c(self.a, "a"), c(self.b, "b")
        u = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        w = torch.zeros_like(u)
        s = torch.zeros_like(u)
        spikes, pots = [], []
        for t in range(T):
            rs = self.threshold * s if self.reset == "soft" else 0.0
            rs = rs.detach() if (self.detach_reset and torch.is_tensor(rs)) else rs
            u_new = alpha * (u - rs) + (1 - alpha) * (self.in_gain * x[t] - w)
            w = beta * w + (1 - beta) * (a * u + b * s)
            u = u_new
            s = spike(u, self.threshold, self.surrogate, self.sharpness)
            spikes.append(s)
            pots.append(u)
        z = torch.stack(spikes)
        return z, torch.stack(pots), z

    def _forward_se(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, N = x.shape
        alpha = torch.exp(-1.0 / (5.0 + 20.0 * self.th_u.clamp(0.0, 1.0)))
        beta = torch.exp(-1.0 / (60.0 + 240.0 * self.th_w.clamp(0.0, 1.0)))
        a, b = self.q * self.a_hat.clamp(0.0, 1.0), self.q * self.b_hat.clamp(0.0, 2.0)
        u = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        w = torch.zeros_like(u)
        spikes, pots = [], []
        for t in range(T):
            u_hat = alpha * u + (1 - alpha) * (self.in_gain * x[t] - w)
            s = spike(u_hat, self.threshold, self.surrogate, self.sharpness)
            if self.reset == "hard":
                u = u_hat * (1.0 - (s.detach() if self.detach_reset else s))
            elif self.reset == "soft":
                u = u_hat - self.threshold * (s.detach() if self.detach_reset else s)
            else:
                u = u_hat
            w = beta * w + (1 - beta) * (a * u + b * s)          # symplectic Euler: the updated (reset) membrane
            spikes.append(s)
            pots.append(u_hat)
        z = torch.stack(spikes)
        return z, torch.stack(pots), z

    def _forward_brf(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, N = x.shape
        om = self.omega_p.abs().clamp(max=0.99)
        p = -1.0 + torch.sqrt(1.0 - om ** 2)
        boff = self.boff_p.abs()
        ur = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        ui = torch.zeros_like(ur)
        q = torch.zeros_like(ur)
        spikes, pots = [], []
        for t in range(T):
            bt = p - boff - self.q_scale * q
            ur, ui = ur + bt * ur - om * ui + x[t], ui + om * ur + bt * ui
            s = spike(ur - q, self.threshold, self.surrogate, self.sharpness)
            q = self.gamma * q + s
            spikes.append(s)
            pots.append(ur)
        z = torch.stack(spikes)
        return z, torch.stack(pots), z

    def _forward_adapt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, N = x.shape
        if self.adapt_style == "bittar":
            return self._forward_bittar(x)
        if self.adapt_style == "se":
            return self._forward_se(x)
        if self.adapt_style == "brf":
            return self._forward_brf(x)
        alpha, beta_w = torch.sigmoid(self.leak_raw), torch.sigmoid(self.wleak_raw)
        u = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        w = torch.zeros_like(u)
        z = torch.zeros_like(u)
        spikes, pots = [], []
        for t in range(T):
            u_pre = alpha * u + x[t] - w
            h = spike(u_pre, self.threshold, self.surrogate, self.sharpness)
            w = beta_w * w + self.a * u + self.b * z             # uses u_{t-1} (after reset) and z_{t-1}
            if self.reset == "soft":
                sub = self.threshold * h
                u = u_pre - (sub.detach() if self.detach_reset else sub)
            elif self.reset == "hard":
                u = u_pre * (1.0 - (h.detach() if self.detach_reset else h))
            else:
                u = u_pre
            z = h
            spikes.append(h)
            pots.append(u_pre)
        s = torch.stack(spikes)
        return s, torch.stack(pots), s

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, N = x.shape
        if self.adapt:
            return self._forward_adapt(x)
        beta = torch.sigmoid(self.leak_raw)
        if self.reset == "none":   # leaky integrator, spikes only read out (control used in REACT, Table IV)
            v_pre = linear_recurrence(torch.complex(x, torch.zeros_like(x)), mag=beta.to(x.dtype), phase=torch.zeros_like(beta).to(x.dtype)).real
            h = spike(v_pre, self.threshold, self.surrogate, self.sharpness)
            return h, v_pre, h
        if self.reset == "soft" and self.detach_reset and self.parallel_reset:   # see CLIF._soft_reset_parallel
            with torch.no_grad():
                v = torch.zeros(B, N, dtype=x.dtype, device=x.device)
                fired = []
                for t in range(T):
                    v_pre = beta * v + x[t]
                    h = (v_pre >= self.threshold).to(x.dtype)
                    v = v_pre - self.threshold * h
                    fired.append(h)
                H = torch.stack(fired)
                R = torch.cat([torch.zeros_like(H[:1]), H[:-1]]) * self.threshold
            xc = torch.complex(x - beta * R, torch.zeros_like(x))
            v_pre = linear_recurrence(xc, mag=beta.to(x.dtype), phase=torch.zeros_like(beta).to(x.dtype)).real
            h_sur = spike(v_pre, self.threshold, self.surrogate, self.sharpness)
            h = H + (h_sur - h_sur.detach())
            return h, v_pre, h
        v = torch.zeros(B, N, dtype=x.dtype, device=x.device)
        spikes, pots = [], []
        for t in range(T):
            if self.urgency is None:
                v_pre = beta * v + x[t]
                thr_scale = 1.0
            else:
                gain, leak, thr_scale = self.urgency.factors(t)
                v_pre = beta * leak * v + x[t] * gain
            h = spike(v_pre / thr_scale, self.threshold, self.surrogate, self.sharpness)
            if self.reset == "soft":
                sub = self.threshold * thr_scale * h
                v = v_pre - (sub.detach() if self.detach_reset else sub)
            else:
                v = v_pre * (1.0 - (h.detach() if self.detach_reset else h))
            spikes.append(h)
            pots.append(v_pre)
        s = torch.stack(spikes)
        return s, torch.stack(pots), s


class ComplexLeakyRNN(nn.Module):
    """Non-spiking complex baseline (CVNN): same leaky complex recurrence, modReLU output, no reset."""

    def __init__(self, num_features: int, leak_mag: float = 0.85, leak_phase: float = 0.15):
        super().__init__()
        self.leak_mag_raw = nn.Parameter(torch.full((num_features,), _logit(leak_mag)))
        self.leak_phase = nn.Parameter(torch.full((num_features,), float(leak_phase)))
        self.bias = nn.Parameter(torch.full((num_features,), -0.5))   # modReLU bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = linear_recurrence(x, mag=torch.sigmoid(self.leak_mag_raw).to(x.real.dtype),
                              phase=self.leak_phase.to(x.real.dtype))   # no reset: linear, parallel
        m = complex_magnitude(u)
        # modReLU; the 1e-6 keeps d(u/|u|)/du bounded when |u| -> 0 (otherwise gradients explode)
        out = torch.relu(m + self.bias) / (m + 1e-6) * u
        return out, u, torch.zeros_like(out.real)


class LRU(nn.Module):
    """Linear Recurrent Unit (Orvieto et al., 2023), the non-spiking complex-diagonal SSM baseline.

        h_t = lambda * h_{t-1} + gamma * (B x_t),   lambda = exp(-exp(nu) + j theta)
        y_t = Re(C h_t) + D x_t
    with |lambda| initialised uniformly on the ring [r_min, r_max] and gamma = sqrt(1 - |lambda|^2).
    """

    def __init__(self, in_features: int, state_features: int, out_features: int, r_min: float = 0.4, r_max: float = 0.99, max_phase: float = math.pi):
        super().__init__()
        u1, u2 = torch.rand(state_features), torch.rand(state_features)
        nu = torch.log(-0.5 * torch.log(u1 * (r_max**2 - r_min**2) + r_min**2))
        theta = u2 * max_phase
        self.nu_log, self.theta = nn.Parameter(nu), nn.Parameter(theta)
        lam_abs = torch.exp(-torch.exp(nu))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - lam_abs**2)))
        self.B = ComplexLinear(in_features, state_features, init="glorot", karatsuba=False)
        self.C = ComplexLinear(state_features, out_features, init="glorot", karatsuba=False)
        self.D = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, _ = x.shape
        lam = torch.polar(torch.exp(-torch.exp(self.nu_log)), self.theta)
        gamma = torch.exp(self.gamma_log)
        bx = self.B(x) * gamma
        h = torch.zeros(B, lam.shape[0], dtype=bx.dtype, device=x.device)
        hs = []
        for t in range(T):
            h = lam * h + bx[t]
            hs.append(h)
        hs = torch.stack(hs)
        xr = x.real if torch.is_complex(x) else x
        y = self.C(hs).real + F.linear(xr, self.D)
        return y, hs, torch.zeros_like(y)


class GatedRNN(nn.Module):
    """Non-spiking reference (N19): a standard GRU or LSTM hidden layer (torch.nn.GRU / nn.LSTM, one layer, own input and
    recurrent weights) in place of a spiking layer. Returns (h, h, zeros) like the other layers: the output h_t in
    (-1, 1) feeds the plastic synapse and the read-out exactly as spikes do."""

    def __init__(self, in_features: int, num_features: int, cell: str = "gru"):
        super().__init__()
        if cell not in ("gru", "lstm"):
            raise ValueError("cell must be 'gru' or 'lstm'")
        self.num_features = num_features
        self.rnn = (nn.GRU if cell == "gru" else nn.LSTM)(in_features, num_features)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(x)
        return h, h, torch.zeros_like(h)


class PRF(nn.Module):
    """Parallel Resonate-and-Fire neuron (Huang et al., arXiv 2410.03530, Eq. 19/27):

        u_t = exp(Delta (-1/tau + j theta)) u_{t-1} + Delta c_t,   s_t = H(Re u_t - V_th)

    complex state, real input current c_t, binary spikes, no reset (computed by FFT convolution).
    Per-neuron theta ~ U(0, theta_max) and log Delta ~ U(log dt_min, log dt_max) (both learnable); tau fixed
    (paper: tau = 2; theta range chosen per task)."""

    def __init__(self, num_features: int, threshold: float = 1.0, tau: float = 2.0, theta_max: float = math.pi / 4,
                 dt_range: tuple[float, float] = (0.001, 0.1), surrogate: str = "atan", sharpness: float = 2.5):
        super().__init__()
        self.num_features, self.threshold, self.tau = num_features, threshold, tau
        self.surrogate, self.sharpness = surrogate, sharpness
        self.theta = nn.Parameter(torch.rand(num_features) * theta_max)
        lo, hi = dt_range
        self.log_dt = nn.Parameter(math.log(lo) + torch.rand(num_features) * (math.log(hi) - math.log(lo)))

    @property
    def beta(self) -> torch.Tensor:
        dt = torch.exp(self.log_dt)
        return torch.polar(torch.exp(-dt / self.tau), self.theta * dt)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xr = x.real if torch.is_complex(x) else x
        dt = torch.exp(self.log_dt)
        drive = torch.complex(xr * dt, torch.zeros_like(xr))
        u = linear_recurrence(drive, mag=torch.exp(-dt / self.tau).to(xr.dtype), phase=(self.theta * dt).to(xr.dtype))
        h = spike(u.real, self.threshold, self.surrogate, self.sharpness)
        return h, u, h


class CSiLIF(nn.Module):
    """C-SiLIF neuron of REACT (Keime et al., arXiv 2609.19204, Eq. 2-5; from Fabre et al. 2026):

        u_k = alpha u_{k-1} + b z_k,   alpha = exp((-e^{lambda_re} + j lambda_im) e^{rho} Delta t),
        s_k = H(2 Re u_k - theta)

    complex state, complex input gain b, binary spikes, learnable threshold, no reset (FFT convolution).
    Initialisation as in the paper: e^{lambda_re} = 0.5, lambda_im shared, e^{rho} log-uniform in
    [1e-3, 1e-1] (per-neuron time scale), b_re ~ U(0, 1), b_im = 0. Delta t = 1 step here."""

    def __init__(self, num_features: int, threshold: float = 1.0, lambda_im: float = math.pi,
                 rho_range: tuple[float, float] = (1e-3, 1e-1), surrogate: str = "atan", sharpness: float = 2.5):
        super().__init__()
        self.num_features, self.surrogate, self.sharpness = num_features, surrogate, sharpness
        self.lambda_re = nn.Parameter(torch.full((num_features,), math.log(0.5)))
        self.lambda_im = nn.Parameter(torch.full((num_features,), float(lambda_im)))
        lo, hi = rho_range
        self.rho = nn.Parameter(math.log(lo) + torch.rand(num_features) * (math.log(hi) - math.log(lo)))
        self.b_re = nn.Parameter(torch.rand(num_features))
        self.b_im = nn.Parameter(torch.zeros(num_features))
        self.vth = nn.Parameter(torch.full((num_features,), float(threshold)))

    @property
    def threshold(self) -> float:
        return float(self.vth.detach().mean())

    @property
    def beta(self) -> torch.Tensor:
        d = torch.exp(self.rho)
        return torch.polar(torch.exp(-torch.exp(self.lambda_re) * d), self.lambda_im * d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xr = x.real if torch.is_complex(x) else x
        drive = torch.complex(xr * self.b_re, xr * self.b_im)
        d = torch.exp(self.rho)
        u = linear_recurrence(drive, mag=torch.exp(-torch.exp(self.lambda_re) * d).to(xr.dtype),
                              phase=(self.lambda_im * d).to(xr.dtype))
        # learnable per-neuron threshold: H(2 Re u - theta) = H((2 Re u - theta) + 1 - 1) with the scalar spike fn
        h = spike(2 * u.real - self.vth + 1.0, 1.0, self.surrogate, self.sharpness)
        return h, u, h
