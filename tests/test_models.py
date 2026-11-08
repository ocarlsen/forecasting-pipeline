import datetime as dt

import pytest

from fbe.models import CutoffUnknownError, ModelSpec, get_model


def test_candidate_models_have_sourced_cutoffs():
    for model_id in ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5"):
        model = get_model(model_id)
        assert model.cutoff_date is not None
        assert model.cutoff_url is not None and model.cutoff_url.startswith("https://")


def test_allowed_for_backtest_accepts_due_date_after_cutoff():
    model = get_model("claude-haiku-4-5")
    model.allowed_for_backtest(dt.date(2025, 8, 1))  # after cutoff 2025-07-31 -- should not raise


def test_allowed_for_backtest_refuses_due_date_before_cutoff():
    model = get_model("claude-sonnet-5")
    with pytest.raises(CutoffUnknownError):
        model.allowed_for_backtest(dt.date(2025, 1, 1))  # before cutoff 2026-01-31


def test_allowed_for_backtest_refuses_unknown_cutoff_loudly():
    model = ModelSpec(
        model_id="fake-model",
        display_name="Fake",
        price_in_per_mtok=1.0,
        price_out_per_mtok=1.0,
        cutoff_date=None,
        cutoff_url=None,
        cutoff_note="never sourced",
        supports_temperature=True,
        thinking_mode="none",
        default_effort=None,
    )
    with pytest.raises(CutoffUnknownError):
        model.allowed_for_backtest(dt.date(2030, 1, 1))


def test_unknown_model_id_raises():
    with pytest.raises(ValueError):
        get_model("not-a-real-model")
