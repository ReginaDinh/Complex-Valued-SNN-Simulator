import pytest
import torch

from cvsnn.data.synthetic import input_transform, spectral_batch, transformed_features
from cvsnn.models import KINDS, ModelConfig, SequenceClassifier

INPUTS = {"cvsnn": "complex", "lif": "reim", "cvnn": "complex", "lru": "complex", "prf": "reim", "csilif": "reim"}


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("hidden", [[], [8]])
def test_forward_backward_all_kinds(kind, hidden):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 2, (4,), generator=g)
    x = input_transform(spectral_batch(y, generator=g), INPUTS[kind])
    m = SequenceClassifier(transformed_features(16, INPUTS[kind]), 2, ModelConfig(kind=kind, hidden=hidden))
    logits, info = m(x)
    assert logits.shape == (4, 2) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, y).backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(gr is not None and torch.isfinite(gr).all() for gr in grads)
    assert m.real_mults_per_step() > 0 and 0.0 <= info["spike_rate"] <= 1.0


def test_lif_rejects_complex_input():
    m = SequenceClassifier(16, 2, ModelConfig(kind="lif"))
    with pytest.raises(ValueError):
        m(torch.zeros(3, 1, 16, dtype=torch.complex64))


def test_data_is_reproducible_and_phase_blind_removes_phase():
    a = spectral_batch(torch.tensor([0, 1]), generator=torch.Generator().manual_seed(5))
    b = spectral_batch(torch.tensor([0, 1]), generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and a.shape == (30, 2, 16)
    mag = input_transform(a, "magnitude")
    assert not torch.is_complex(mag) and torch.allclose(mag, a.abs())


def test_delayed_order_task_labels_and_silence():
    from cvsnn.data.synthetic import delayed_order_batch
    x, y = delayed_order_batch(256, symbols=4, channels=8, window=10, delay=20, tail=5, generator=torch.Generator().manual_seed(0))
    assert x.shape == (35, 256, 8) and y.min() >= 0 and y.max() <= 11
    assert x[10:].abs().sum() == 0            # nothing after the presentation window
    assert (x.sum(dim=(0, 2)) == 4).all()     # two pulses x 2 channels each
    assert len(torch.unique(y)) == 12


def test_readout_last_ignores_early_steps():
    torch.manual_seed(0)
    m = SequenceClassifier(4, 3, ModelConfig(kind="cvnn", readout_last=2))
    x = torch.zeros(10, 1, 4, dtype=torch.complex64)
    x2 = x.clone()
    x2[0, 0, 0] = 0.0  # identical
    a, _ = m(x)
    assert torch.allclose(a, m(x2)[0])
