"""Device non-idealities of an analog crossbar, applied to synaptic weights and column currents.

Mapping (per real weight component): w -> differential conductance pair (G+, G-), normalised so the
largest |w| in the layer maps to G_max = 1:

    G+ = max(w, 0) * s,  G- = max(-w, 0) * s,  s = 1 / max|w|
    quantise each device to 2**weight_bits levels in [0, 1]     (straight-through gradient)
    + programming noise N(0, prog_noise)  (static per programmed chip; clipped at 0)
    * drift (t / t0)^(-drift_nu)          (PCM-style power-law conductance drift)
    w_eff = (G+ - G-) / s

A complex weight uses two pairs (Re, Im), i.e. the 4 devices G_re+, G_re-, G_im+, G_im-.

Column read-out: read noise ~ N(0, read_noise * sqrt(2) * ||x|| / s) added to each output current
(each device in the column contributes independent noise), then an ADC with 2**adc_bits levels over
[-adc_range * V_th, +adc_range * V_th] (full scale tied to the firing threshold).

All noise magnitudes are relative to G_max. Defaults are "ideal" (no effect). These are first-order
models meant to be replaced by measured device statistics when hardware data is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceConfig:
    weight_bits: int | None = None  # conductance levels per device = 2**bits; None = continuous
    prog_noise: float = 0.0         # std of programming error, fraction of G_max
    read_noise: float = 0.0         # std of per-read device noise, fraction of G_max
    drift_nu: float = 0.0           # drift exponent (PCM ~0.05; 0 = none)
    drift_time: float = 1.0         # t / t0 at evaluation time
    adc_bits: int | None = None     # column ADC resolution; None = ideal
    adc_range: float = 2.0          # ADC full scale = ±adc_range * V_th

    @property
    def ideal(self) -> bool:
        return (self.weight_bits is None and self.prog_noise == 0 and self.read_noise == 0
                and self.drift_nu == 0 and self.adc_bits is None)


def _ste(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Forward q, backward identity (straight-through estimator)."""
    return x + (q - x).detach()


def _quantise_unit(g: torch.Tensor, bits: int) -> torch.Tensor:
    levels = 2**bits - 1
    return _ste(g, torch.round(g.clamp(0, 1) * levels) / levels)


def layer_scale(*ws: torch.Tensor) -> torch.Tensor:
    """s such that the largest |w| over all components of a layer maps to G_max = 1."""
    return 1.0 / torch.stack([w.detach().abs().max() for w in ws]).max().clamp_min(1e-12)


def program_weight(w: torch.Tensor, cfg: DeviceConfig, generator: torch.Generator | None, s: torch.Tensor) -> torch.Tensor:
    """Effective weight after mapping a real weight tensor onto differential device pairs (scale s)."""
    if cfg.weight_bits is None and cfg.prog_noise == 0 and cfg.drift_nu == 0:
        return w
    gp, gn = torch.relu(w) * s, torch.relu(-w) * s
    if cfg.weight_bits is not None:
        gp, gn = _quantise_unit(gp, cfg.weight_bits), _quantise_unit(gn, cfg.weight_bits)
    if cfg.prog_noise > 0:
        gp = (gp + cfg.prog_noise * torch.randn(gp.shape, generator=generator).to(gp.device)).clamp_min(0)
        gn = (gn + cfg.prog_noise * torch.randn(gn.shape, generator=generator).to(gn.device)).clamp_min(0)
    if cfg.drift_nu > 0 and cfg.drift_time != 1.0:
        k = cfg.drift_time ** (-cfg.drift_nu)
        gp, gn = gp * k, gn * k
    return (gp - gn) / s


def read_out(y: torch.Tensor, x: torch.Tensor, w_scale: torch.Tensor, cfg: DeviceConfig, threshold: float,
             generator: torch.Generator | None) -> torch.Tensor:
    """Apply read noise and ADC quantisation to column outputs y (real or complex)."""
    if cfg.read_noise > 0:
        # Noise magnitude is a property of the circuit, not something to learn: no gradient through it
        # (also avoids NaN gradients of |x| at x = 0)
        with torch.no_grad():
            xd = x.detach()
            p = xd.real**2 + xd.imag**2 if torch.is_complex(xd) else xd**2
            sigma = cfg.read_noise * (2.0**0.5) * p.sum(-1, keepdim=True).sqrt() / w_scale

        def noise_like(t: torch.Tensor) -> torch.Tensor:
            return torch.randn(t.shape, generator=generator).to(t.device) * sigma

        y = torch.complex(y.real + noise_like(y.real), y.imag + noise_like(y.imag)) if torch.is_complex(y) else y + noise_like(y)
    if cfg.adc_bits is not None:
        fs = cfg.adc_range * threshold
        levels = 2**cfg.adc_bits - 1

        def adc(t: torch.Tensor) -> torch.Tensor:
            u = (t.clamp(-fs, fs) + fs) / (2 * fs)
            return _ste(t, torch.round(u * levels) / levels * 2 * fs - fs)

        y = torch.complex(adc(y.real), adc(y.imag)) if torch.is_complex(y) else adc(y)
    return y
