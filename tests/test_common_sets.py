"""Backtest's (and plan's) set selection either refuses loudly when the user
names a set some requested model can't use, or defaults to the intersection of
cutoff-allowed AND actually-resolved sets.
"""
import datetime as dt

import pytest

import fbe.cli as cli


def test_explicit_sets_refuses_when_a_model_cant_use_one(monkeypatch):
    # claude-sonnet-5's cutoff is 2026-01-31 -- a set due before that is unusable for it.
    with pytest.raises(SystemExit) as exc:
        cli._resolve_common_sets(["claude-haiku-4-5", "claude-sonnet-5"], "2025-09-01")
    assert "claude-sonnet-5" in str(exc.value)
    assert "2025-09-01" in str(exc.value)


def test_explicit_sets_accepted_when_every_model_can_use_them():
    dates = cli._resolve_common_sets(["claude-haiku-4-5", "claude-sonnet-5"], "2026-10-11,2026-09-27")
    assert dates == ["2026-09-27", "2026-10-11"]


def test_default_sets_intersects_cutoff_and_resolved(monkeypatch):
    fake_dates = ["2024-01-01", "2025-08-01", "2026-02-01", "2026-09-01"]
    monkeypatch.setattr(cli, "list_question_set_dates", lambda **kw: fake_dates)

    def fake_load_resolution_set(date_str, **kw):
        # 2024-01-01: before any candidate model's cutoff -- would be excluded anyway.
        # 2025-08-01: after Haiku's cutoff (2025-07-31) but before Sonnet's (2026-01-31).
        # 2026-02-01: after both cutoffs, but has ZERO resolved rows (too recent).
        # 2026-09-01: after both cutoffs AND has resolved rows -- the one that should survive.
        if date_str == "2026-02-01":
            return {"resolutions": [{"id": "x", "resolved": False}]}
        if date_str == "2026-09-01":
            return {"resolutions": [{"id": "x", "resolved": True, "resolved_to": 1.0}]}
        return {"resolutions": [{"id": "x", "resolved": True, "resolved_to": 1.0}]}

    monkeypatch.setattr(cli, "load_resolution_set", fake_load_resolution_set)

    dates = cli._resolve_common_sets(["claude-haiku-4-5", "claude-sonnet-5"], None)
    assert dates == ["2026-09-01"]


def test_cutoff_ok_helper_matches_model_spec():
    from fbe.models import get_model

    haiku = get_model("claude-haiku-4-5")
    assert cli._cutoff_ok(haiku, dt.date(2025, 8, 1))
    assert not cli._cutoff_ok(haiku, dt.date(2025, 1, 1))
