"""Spike function H(|U| - V_th) with surrogate gradients.

The three surrogate shapes are the same formulas as ``computeSurrogateDerivative`` in
``src/simulationEngine.ts`` (relative distance to the threshold, sharpness 2.5 by default), so
the browser simulator and this library train the same way.
"""

from __future__ import annotations

import math

import torch

SURROGATES = ("von_mises", "fast_sigmoid", "atan")


def surrogate_grad(m: torch.Tensor, threshold: float, kind: str = "von_mises", sharpness: float = 2.5) -> torch.Tensor:
    """d spike / d m evaluated at magnitude ``m`` (shape preserved)."""
    diff = (m - threshold) / max(threshold, 1e-6)
    if kind == "fast_sigmoid":
        return 1.0 / (1.0 + sharpness * diff.abs()) ** 2
    if kind == "atan":
        arg = sharpness * math.pi * diff * 0.5
        return 1.0 / (1.0 + arg * arg)
    if kind == "von_mises":
        d = diff.clamp(-1.0, 1.0)
        # Same large-kappa approximation of I0 as the JS engine (keeps both implementations identical)
        i0 = math.exp(sharpness) / math.sqrt(2 * math.pi * max(0.5, sharpness))
        return torch.exp(sharpness * torch.cos(math.pi * d)) / (2 * math.pi * i0)
    raise ValueError(f"unknown surrogate {kind!r}; expected one of {SURROGATES}")


class SpikeFn(torch.autograd.Function):
    """Real-valued Heaviside H(m - threshold); backward uses the chosen surrogate."""

    @staticmethod
    def forward(ctx, m: torch.Tensor, threshold: float, kind: str, sharpness: float) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(m)
        ctx.threshold, ctx.kind, ctx.sharpness = threshold, kind, sharpness
        return (m >= threshold).to(m.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        (m,) = ctx.saved_tensors
        return grad_out * surrogate_grad(m, ctx.threshold, ctx.kind, ctx.sharpness), None, None, None


def spike(m: torch.Tensor, threshold: float, kind: str = "von_mises", sharpness: float = 2.5) -> torch.Tensor:
    if kind not in SURROGATES:
        raise ValueError(f"unknown surrogate {kind!r}; expected one of {SURROGATES}")
    return SpikeFn.apply(m, threshold, kind, sharpness)
