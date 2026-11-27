"""Step-1 upgrades: reset none / brf, heterogeneous frequency init, FFT recurrence, PRF and C-SiLIF."""

import math

import pytest
import torch

from cvsnn.hardware import estimate_cost, load_energy_table
from cvsnn.layers import CLIF, PRF, CSiLIF, linear_recurrence
from cvsnn.models import ModelConfig, SequenceClassifier


def _sequential(x, beta):
    u = torch.zeros_like(x[0])
    out = []
    for t in range(x.shape[0]):
        u = beta * u + x[t]
        out.append(u)
    return torch.stack(out)


def test_fft_recurrence_matches_sequential():
    torch.manual_seed(0)
    x = torch.randn(300, 3, 5, dtype=torch.complex128)
    beta = torch.polar(torch.tensor([0.5, 0.9, 0.99, 0.999, 0.3], dtype=torch.float64), torch.tensor([0.1, 1.0, 2.5, 0.0, -1.2], dtype=torch.float64))
    assert torch.allclose(linear_recurrence(x, beta), _sequential(x, beta), atol=1e-9)


def test_reset_none_equals_sequential_and_keeps_state():
    torch.manual_seed(1)
    n = CLIF(4, threshold=0.5, reset="none").double()
    x = 0.5 * torch.randn(50, 2, 4, dtype=torch.complex128)
    s, pot, h = n(x)
    assert torch.allclose(pot, _sequential(x, n.beta.to(x.dtype)), atol=1e-9)
    assert h.sum() > 0  # spikes happen, yet the state was never reduced by them


def test_reset_none_gradients_match_sequential_autograd():
    torch.manual_seed(2)
    n = CLIF(3, threshold=1e3, reset="none").double()
    x = (0.1 * torch.randn(20, 2, 3, dtype=torch.complex128)).requires_grad_()
    _, pot, _ = n(x)
    pot.abs().sum().backward()
    g_fft = x.grad.clone()
    x.grad = None
    _sequential(x, n.beta.to(x.dtype)).abs().sum().backward()
    assert torch.allclose(g_fft, x.grad, atol=1e-8)


def test_brf_reset_keeps_phase_raises_threshold_and_damps():
    n = CLIF(1, threshold=1.0, leak_mag=0.9, leak_phase=0.0, learn_leak=False, reset="brf", brf_gamma=0.9, brf_strength=1.0).double()
    x = torch.zeros(4, 1, 1, dtype=torch.complex128)
    x[0, 0, 0] = 1.5 * complex(math.cos(0.7), math.sin(0.7))
    s, pot, h = n(x)
    assert h[0, 0, 0] == 1 and h[1:].sum() == 0          # refractory: threshold rose to 1 + q
    assert torch.angle(pot[1, 0, 0]).item() == pytest.approx(0.7)   # phase preserved (no subtraction)
    beta = torch.sigmoid(n.leak_mag_raw)[0].item()
    assert pot[1, 0, 0].abs().item() == pytest.approx(1.5 * beta * math.exp(-1.0), rel=1e-6)  # extra damping


def test_heterogeneous_init_ranges():
    torch.manual_seed(0)
    n = CLIF(2000, leak_mag_range=(0.9, 0.999), leak_phase_range=(0.01, 1.0), phase_dist="log")
    r = torch.sigmoid(n.leak_mag_raw)
    assert r.min() >= 0.9 - 1e-6 and r.max() <= 0.999 + 1e-6 and r.std() > 0.01
    w = n.leak_phase
    assert w.min() >= 0.01 - 1e-6 and w.max() <= 1.0 + 1e-6
    assert abs(torch.log(w).mean().item() - (math.log(0.01) + math.log(1.0)) / 2) < 0.1   # log-uniform


def test_prf_and_csilif_follow_their_equations():
    torch.manual_seed(0)
    p = PRF(4, threshold=0.2, theta_max=1.0, dt_range=(0.05, 0.5)).double()
    x = torch.randn(30, 2, 4, dtype=torch.float64)
    s, u, h = p(x)
    dt = torch.exp(p.log_dt)
    ref = _sequential(torch.complex(x * dt, torch.zeros_like(x)), p.beta.to(torch.complex128))
    assert torch.allclose(u, ref, atol=1e-9) and torch.equal(h, (u.real >= 0.2).double())
    c = CSiLIF(4, threshold=0.3).double()
    s, u, h = c(x)
    ref = _sequential(torch.complex(x * c.b_re, x * c.b_im), c.beta.to(torch.complex128))
    assert torch.allclose(u, ref, atol=1e-9)
    assert torch.equal(h, ((2 * u.real - c.vth) >= 0).double())
    assert (c.beta.abs() < 1).all()


@pytest.mark.parametrize("kind", ["prf", "csilif"])
def test_new_kinds_train_and_cost(kind):
    torch.manual_seed(0)
    m = SequenceClassifier(8, 3, ModelConfig(kind=kind, hidden=[6]))
    x = torch.randn(25, 4, 8)
    y = torch.tensor([0, 1, 2, 1])
    logits, info = m(x, trace=True)
    torch.nn.functional.cross_entropy(logits, y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert info["layers"][1]["input_kind"] == "binary"
    from pathlib import Path
    table = load_energy_table(Path(__file__).resolve().parents[1] / "configs" / "energy" / "default.yaml")
    assert estimate_cost(info["layers"], table)["digital_event"]["energy_pj"] > 0


def test_cvsnn_model_with_new_options():
    torch.manual_seed(0)
    for reset in ("none", "brf"):
        m = SequenceClassifier(8, 3, ModelConfig(kind="cvsnn", hidden=[6], reset=reset, leak_mag_range=(0.9, 0.999),
                                                 leak_phase_range=(0.05, 1.0)))
        logits, _ = m(torch.randn(25, 4, 8, dtype=torch.complex64))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 1])).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_fft_recurrence_gradient_finite_when_beta_vanishes():
    beta = torch.tensor([0.0 + 0.0j, 1e-30 + 0j, 0.5 + 0.1j], dtype=torch.complex128, requires_grad=True)
    x = torch.randn(10, 1, 3, dtype=torch.complex128)
    linear_recurrence(x, beta).abs().sum().backward()
    assert torch.isfinite(torch.view_as_real(beta.grad)).all()
