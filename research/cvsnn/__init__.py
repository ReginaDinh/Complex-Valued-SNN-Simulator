"""cvsnn: complex-valued spiking neural networks (cLIF) for research."""

from .layers import CLIF, LIF, LRU, ComplexLeakyRNN, ComplexLinear, complex_magnitude
from .models import KINDS, ModelConfig, SequenceClassifier
from .surrogate import SURROGATES, spike, surrogate_grad

__all__ = [
    "CLIF", "LIF", "LRU", "ComplexLeakyRNN", "ComplexLinear", "complex_magnitude",
    "KINDS", "ModelConfig", "SequenceClassifier", "SURROGATES", "spike", "surrogate_grad",
]
__version__ = "0.1.0"
