import json
from pathlib import Path

import pytest
import torch

torch.set_default_dtype(torch.float32)


@pytest.fixture(scope="session")
def js_fixture():
    return json.loads(Path(__file__).parent / "fixtures" / "js_engine.json").read_text())
