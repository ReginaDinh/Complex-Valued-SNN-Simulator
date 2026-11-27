"""The PyTorch cLIF must reproduce the browser engine (src/simulationEngine.ts) on identical
weights and inputs: same pre/post-reset potentials, same spikes, same read-out probabilities."""

import math

import pytest
import torch

from cvsnn.layers import CLIF, ComplexLinear


def _load(case):
    w = torch.tensor(case["weights"], dtype=torch.float64)  # (O, 2I) interleaved re/im
    x = torch.tensor(case["input"], dtype=torch.float64)    # (T, 2I)
    weight = torch.complex(w[:, 0::2], w[:, 1::2])
    xin = torch.complex(x[:, 0::2], x[:, 1::2]).unsqueeze(1)  # (T, 1, I)
    return weight, xin


@pytest.mark.parametrize("case_name", ["soft_th1", "hard_th06"])
def test_forward_matches_js(js_fixture, case_name):
    case = next(c for c in js_fixture["cases"] if c["name"] == case_name)
    weight, xin = _load(case)
    lin = ComplexLinear(weight.shape[1], weight.shape[0]).double()
    with torch.no_grad():
        lin.weight_re.copy_(weight.real)
        lin.weight_im.copy_(weight.imag)
    neuron = CLIF(weight.shape[0], threshold=case["threshold"], leak_mag=case["leakMag"], leak_phase=case["leakPhase"],
                  learn_leak=False, reset=case["reset"]).double()
    # JS uses |beta| directly; set our sigmoid-parameterised leak exactly in float64
    with torch.no_grad():
        p = case["leakMag"]
        neuron.leak_mag_raw.fill_(math.log(p / (1 - p)))
    assert torch.sigmoid(neuron.leak_mag_raw)[0].item() == pytest.approx(case["leakMag"], abs=1e-12)

    with torch.no_grad():
        _, pot, fire = neuron(lin(xin.to(torch.complex128)))
    pre = torch.tensor(case["preReset"], dtype=torch.float64)
    pre_c = torch.complex(pre[:, 0::2], pre[:, 1::2])
    js_spikes = torch.tensor(case["spikes"], dtype=torch.float64)

    assert torch.equal(fire[:, 0, :], js_spikes)
    # JS stores potentials as float32: compare at float32 precision
    assert torch.allclose(pot[:, 0, :], pre_c, atol=2e-6, rtol=1e-5)

    # Read-out: mean_t(|U_pre| / V_th + 0.5 * spike), softmax with temperature 2.5
    rate = (pot[:, 0, :].abs() / case["threshold"] + 0.5 * fire[:, 0, :]).mean(0)
    assert torch.allclose(rate, torch.tensor(case["surrogateRates"], dtype=torch.float64), atol=1e-5)
    probs = torch.softmax(2.5 * rate, -1)
    assert torch.allclose(probs, torch.tensor(case["probs"], dtype=torch.float64), atol=1e-5)


def test_fixture_exercises_spikes(js_fixture):
    total = {c["name"]: sum(sum(r) for r in c["spikes"]) for c in js_fixture["cases"]}
    assert total["soft_th1"] == 0 and total["hard_th06"] > 10
