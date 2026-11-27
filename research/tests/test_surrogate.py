import math

import pytest
import torch

from cvsnn.surrogate import SURROGATES, spike, surrogate_grad


def test_matches_js_engine_values(js_fixture):
    for row in js_fixture["surrogate"]:
        got = surrogate_grad(torch.tensor(row["m"], dtype=torch.float64), row["threshold"], row["kind"], 2.5)
        assert torch.allclose(got, torch.tensor(row["value"], dtype=torch.float64), rtol=1e-12, atol=1e-12), row["kind"]


@pytest.mark.parametrize("kind", SURROGATES)
def test_spike_forward_is_heaviside_and_backward_is_surrogate(kind):
    m = torch.tensor([0.2, 0.99, 1.0, 1.7], dtype=torch.float64, requires_grad=True)
    s = spike(m, 1.0, kind)
    assert s.tolist() == [0.0, 0.0, 1.0, 1.0]
    s.sum().backward()
    assert torch.allclose(m.grad, surrogate_grad(m.detach(), 1.0, kind))


@pytest.mark.parametrize("kind", SURROGATES)
def test_surrogate_peaks_at_threshold(kind):
    for th in (0.5, 1.0, 2.0):
        m = torch.tensor([0.8 * th, th, 1.2 * th])
        g = surrogate_grad(m, th, kind)
        assert g[1] > g[0] and g[1] > g[2]


def test_unknown_surrogate_rejected():
    with pytest.raises(ValueError):
        spike(torch.ones(2), 1.0, "nope")
