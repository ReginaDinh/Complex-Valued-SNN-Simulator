"""P2: synaptic consolidation for continual learning.

Synaptic Intelligence (Zenke, Poole & Ganguli, ICML 2017): every synapse keeps, during a task, the path integral
omega = sum over steps of (-g * delta theta), i.e. how much its own changes lowered the loss. At the end of a task
its importance grows, Omega += omega / (Delta^2 + xi) (Delta: its total change over the task), and later training
pays c * sum Omega (theta - theta*)^2 for moving important synapses away from their consolidated values theta*.
Everything is per synapse and computed online.

For complex weights (ComplexLinear: weight_re, weight_im) the coordinates can be:
  'cart'  : (Re w, Im w), the usual choice when a complex weight is stored as two reals
  'polar' : (|w|, arg w), both consolidated
  'mag'   : only |w| consolidated, the phase stays free (hypothesis: magnitude = slow, phase = fast plasticity)
  'phase' : only arg w consolidated, the magnitude stays free
Every other parameter (leaks, real weights) always uses the cartesian form. Phase distances use
2 (1 - cos(phi - phi*)), which equals (phi - phi*)^2 for small differences and respects the wrap-around.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .layers import ComplexLinear

MODES = ("cart", "polar", "mag", "phase")


def _wrap(d: torch.Tensor) -> torch.Tensor:
    return torch.remainder(d + math.pi, 2 * math.pi) - math.pi


class SynapticIntelligence:
    def __init__(self, model: nn.Module, mode: str = "cart", c: float = 0.1, xi: float = 0.1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode, self.c, self.xi = mode, c, xi
        self.pairs = []                   # (weight_re, weight_im) of complex layers, in polar modes
        paired = set()
        if mode != "cart":
            for m in model.modules():
                if isinstance(m, ComplexLinear):
                    self.pairs.append((m.weight_re, m.weight_im))
                    paired |= {id(m.weight_re), id(m.weight_im)}
        self.single = [p for p in model.parameters() if p.requires_grad and id(p) not in paired]
        z = lambda t: torch.zeros_like(t)
        self.omega = [z(p) for p in self.single]
        self.Omega = [z(p) for p in self.single]
        self.star = [p.detach().clone() for p in self.single]
        self.start = [p.detach().clone() for p in self.single]
        self.p_omega = [(z(a), z(a)) for a, _ in self.pairs]       # (magnitude, phase)
        self.p_Omega = [(z(a), z(a)) for a, _ in self.pairs]
        self.p_star = [self._polar(a, b) for a, b in self.pairs]
        self.p_start = [self._polar(a, b) for a, b in self.pairs]
        self._pre = None
        self.consolidated = False

    @staticmethod
    def _polar(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sqrt(a.detach() ** 2 + b.detach() ** 2 + 1e-12), torch.atan2(b.detach(), a.detach())

    def penalty(self) -> torch.Tensor | float:
        if not self.consolidated or self.c == 0:
            return 0.0
        total = 0.0
        for p, Om, st in zip(self.single, self.Omega, self.star):
            total = total + (Om * (p - st) ** 2).sum()
        for (a, b), (Om_m, Om_p), (m_s, ph_s) in zip(self.pairs, self.p_Omega, self.p_star):
            m = torch.sqrt(a ** 2 + b ** 2 + 1e-12)
            ph = torch.atan2(b, a)
            if self.mode in ("polar", "mag"):
                total = total + (Om_m * (m - m_s) ** 2).sum()
            if self.mode in ("polar", "phase"):
                total = total + (Om_p * 2 * (1 - torch.cos(ph - ph_s))).sum()
        return self.c * total

    def pre_step(self) -> None:
        """Call after backward (and clipping), before optimizer.step(): remember coordinates and their gradients."""
        sing = [(p.detach().clone(), torch.zeros_like(p) if p.grad is None else p.grad.detach.clone()) for p in self.single]
        prs = []
        for a, b in self.pairs:
            m, ph = self._polar(a, b)
            ga = torch.zeros_like(a) if a.grad is None else a.grad.detach()
            gb = torch.zeros_like(b) if b.grad is None else b.grad.detach()
            g_m = ga * torch.cos(ph) + gb * torch.sin(ph)              # dL/d|w|
            g_ph = m * (-ga * torch.sin(ph) + gb * torch.cos(ph))      # dL/d arg w
            prs.append((m, ph, g_m, g_ph))
        self._pre = (sing, prs)

    def post_step(self) -> None:
        """Call after optimizer.step(): accumulate omega += -g * delta."""
        sing, prs = self._pre
        for i, (p, (old, g)) in enumerate(zip(self.single, sing)):
            self.omega[i] += -g * (p.detach() - old)
        for i, ((a, b), (m0, ph0, g_m, g_ph)) in enumerate(zip(self.pairs, prs)):
            m1, ph1 = self._polar(a, b)
            om_m, om_p = self.p_omega[i]
            om_m += -g_m * (m1 - m0)
            om_p += -g_ph * _wrap(ph1 - ph0)

    def end_task(self) -> None:
        """Consolidate: Omega += omega / (Delta^2 + xi); new reference values; reset the path integrals."""
        for i, p in enumerate(self.single):
            d = p.detach() - self.start[i]
            self.Omega[i] += torch.relu(self.omega[i]) / (d ** 2 + self.xi)
            self.star[i] = p.detach().clone()
            self.start[i] = p.detach().clone()
            self.omega[i].zero_()
        for i, (a, b) in enumerate(self.pairs):
            m, ph = self._polar(a, b)
            m0, ph0 = self.p_start[i]
            Om_m, Om_p = self.p_Omega[i]
            om_m, om_p = self.p_omega[i]
            Om_m += torch.relu(om_m) / ((m - m0) ** 2 + self.xi)
            Om_p += torch.relu(om_p) / (_wrap(ph - ph0) ** 2 + self.xi)
            self.p_star[i] = (m, ph)
            self.p_start[i] = (m, ph)
            om_m.zero_()
            om_p.zero_()
        self.consolidated = True
