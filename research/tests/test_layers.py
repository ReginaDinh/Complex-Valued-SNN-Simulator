import math

import pytest
import torch

from cvsnn.layers import CLIF, LIF, LRU, ComplexLinear


def test_karatsuba_equals_standard_and_complex_matmul():
    torch.manual_seed(0)
    a = ComplexLinear(7, 5, bias=True, karatsuba=True)
    b = ComplexLinear(7, 5, bias=True, karatsuba=False)
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        a.bias_re.normal_(); a.bias_im.normal_()
        b.load_state_dict(a.state_dict())
    x = torch.randn(4, 7, dtype=torch.complex64)
    ref = x @ a.weight.T + torch.complex(a.bias_re, a.bias_im)
    assert torch.allclose(a(x), ref, atol=1e-5)
    assert torch.allclose(b(x), ref, atol=1e-5)
    assert a.real_mults_per_sample() == 3 * 35 and b.real_mults_per_sample() == 4 * 35


def test_clif_zero_input_decays_by_beta_and_rotates():
    n = CLIF(1, threshold=100.0, leak_mag=0.9, leak_phase=0.3, learn_leak=False)
    x = torch.zeros(6, 1, 1, dtype=torch.complex128)
    x[0, 0, 0] = 1.0
    _, pot, fire = n(x)
    mag = torch.sigmoid(n.leak_mag_raw)[0].item()
    for t in range(6):
        assert pot[t, 0, 0].abs().item() == pytest.approx(mag**t, rel=1e-6)
        assert torch.angle(pot[t, 0, 0]).item() == pytest.approx((0.3 * t + math.pi) % (2 * math.pi) - math.pi, abs=1e-6)
    assert fire.sum() == 0


def test_clif_soft_reset_keeps_phase():
    n = CLIF(1, threshold=1.0, leak_mag=0.5, leak_phase=0.0, learn_leak=False)
    ang = 0.7
    x = torch.zeros(2, 1, 1, dtype=torch.complex128)
    x[0, 0, 0] = 1.5 * complex(math.cos(ang), math.sin(ang))
    spikes, pot, fire = n(x)
    assert fire[0, 0, 0] == 1
    assert spikes[0, 0, 0].abs().item() == pytest.approx(1.0)
    assert torch.angle(spikes[0, 0, 0]).item() == pytest.approx(ang)
    # step 1: U = beta * (U_pre - V_th e^{j ang}) = 0.5 * 0.5 e^{j ang}
    assert pot[1, 0, 0].abs().item() == pytest.approx(0.25, rel=1e-6)
    assert torch.angle(pot[1, 0, 0]).item() == pytest.approx(ang)


def test_clif_hard_reset_zeroes_state():
    n = CLIF(1, threshold=1.0, leak_mag=0.5, reset="hard", learn_leak=False)
    x = torch.zeros(2, 1, 1, dtype=torch.complex128)
    x[0, 0, 0] = 2.0
    _, pot, _ = n(x)
    assert pot[1, 0, 0].abs().item() == pytest.approx(0.0, abs=1e-9)


def test_clif_gradcheck_subthreshold():
    """Below threshold the cLIF is a smooth complex recurrence: autograd must match finite differences
    for the input and the learnable leak (magnitude and phase)."""
    torch.manual_seed(1)
    n = CLIF(3, threshold=1e3, leak_mag=0.8, leak_phase=0.4).double()
    x = (0.1 * torch.randn(5, 2, 3, dtype=torch.complex128)).requires_grad_()

    def f(x, raw_mag, phase):
        _, pot, _ = torch.func.functional_call(n, {"leak_mag_raw": raw_mag, "leak_phase": phase}, (x,))
        return pot.abs()

    assert torch.autograd.gradcheck(f, (x, n.leak_mag_raw.detach().clone().requires_grad_(), n.leak_phase.detach().clone().requires_grad_()))


def test_lif_and_lru_shapes():
    s, v, h = LIF(4)(torch.randn(6, 2, 4))
    assert s.shape == v.shape == (6, 2, 4)
    y, hs, _ = LRU(3, 8, 5)(torch.randn(6, 2, 3, dtype=torch.complex64))
    assert y.shape == (6, 2, 5) and hs.shape == (6, 2, 8)
    assert y.dtype == torch.float32
