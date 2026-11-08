import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def question_set():
    return json.loads((FIXTURES / "question_set_sample.json").read_text())


@pytest.fixture
def resolution_set():
    return json.loads((FIXTURES / "resolution_set_sample.json").read_text())
