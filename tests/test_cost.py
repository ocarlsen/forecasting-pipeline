import pytest

from fbe.cost import CostCapExceededError, enforce_cap, estimate_cost
from fbe.data import normalize_question_set
from fbe.models import get_model


def test_estimate_cost_positive(question_set):
    rows, _ = normalize_question_set(question_set)
    model = get_model("claude-haiku-4-5")
    est = estimate_cost(rows, model, use_batch=True)
    assert est.n_calls == 4  # 2 market + 2 dataset questions (combo dropped)
    assert est.ceiling_usd > 0
    assert est.ceiling_input_tokens > 0
    assert est.ceiling_output_tokens > 0


def test_cost_cap_refuses_when_ceiling_exceeds_max_usd(question_set):
    rows, _ = normalize_question_set(question_set)
    model = get_model("claude-opus-5-5")
    est = estimate_cost(rows, model, use_batch=False)
    with pytest.raises(CostCapExceededError):
        enforce_cap(est, max_usd=est.ceiling_usd / 2)


def test_cost_cap_allows_when_under_budget(question_set):
    rows, _ = normalize_question_set(question_set)
    model = get_model("claude-haiku-4-5")
    est = estimate_cost(rows, model, use_batch=True)
    enforce_cap(est, max_usd=est.ceiling_usd * 2)  # should not raise


def test_cost_cap_none_never_refuses(question_set):
    rows, _ = normalize_question_set(question_set)
    model = get_model("claude-opus-5-5")
    est = estimate_cost(rows, model, use_batch=False)
    enforce_cap(est, max_usd=None)
