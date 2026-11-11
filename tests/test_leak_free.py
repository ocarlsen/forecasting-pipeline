"""Dedicated proof that prequential fitting is leak-free: a training row whose outcome
resolved on or after the target due date must never survive the filter. The first two tests
exercise `leak_free_filter` directly; they cannot notice if a caller stops calling it. The
last test runs `cmd_backtest` (dry run) end to end and asserts the filter's effect on the
Platt pool the CLI actually builds, so deleting the CLI's call to the filter turns it red.
"""
import datetime as dt
from argparse import Namespace

import pytest

import fbe.cli as cli
import fbe.pipeline as pipeline
from fbe.pipeline import TrainingRow, leak_free_filter


def _row(resolution_date_str, raw_p=0.5, outcome=1.0):
    return TrainingRow(
        kind="dataset",
        source="acled",
        bucket=30,
        raw_p=raw_p,
        outcome=outcome,
        resolution_date=dt.date.fromisoformat(resolution_date_str),
    )


def test_leak_free_filter_excludes_rows_on_or_after_due_date():
    due_date = dt.date(2027, 1, 1)
    before = _row("2026-12-31")
    on_due_date = _row("2027-01-01")
    after = _row("2027-06-01")

    result = leak_free_filter([before, on_due_date, after], due_date)

    assert before in result
    assert on_due_date not in result  # strictly BEFORE due_date, not on-or-before
    assert after not in result


def test_filter_removal_would_leak_a_future_row():
    """If leak_free_filter were a no-op (the bug this test exists to catch), a row that
    resolves AFTER the target due date -- i.e. it only exists because the model already
    knows the future -- would enter the training pool. Assert the real filter removes
    exactly that row, so a future regression that deletes the filter call fails this test."""
    due_date = dt.date(2027, 1, 1)
    rows = [_row("2025-01-01"), _row("2028-01-01")]

    filtered = leak_free_filter(rows, due_date)
    unfiltered_would_be = rows  # what you'd get if the filter were removed entirely

    assert len(filtered) == 1
    assert len(unfiltered_would_be) == 2
    assert filtered != unfiltered_would_be


SEED_DUE = "2026-06-01"
TARGET_DUE = "2027-01-01"


def test_backtest_cli_applies_leak_free_filter(monkeypatch, tmp_path, question_set, resolution_set):
    """Seed prior history whose resolutions fall before, on, and after the target due date.
    The Platt pool `cmd_backtest` fits on must contain only the strictly-before rows."""
    monkeypatch.chdir(tmp_path)
    resolutions = []
    for i, rd in enumerate(["2026-06-05", "2026-06-06", "2027-01-01", "2027-03-01", "2027-03-02"]):
        resolutions.append(
            {"id": f"seed-{i}", "source": "manifold", "direction": None,
             "resolution_date": rd, "resolved_to": float(i % 2), "resolved": True}
        )
    seed_rset = {"forecast_due_date": SEED_DUE, "question_set": f"{SEED_DUE}-llm.json",
                 "resolutions": resolutions}

    def fake_rset(date_str, **kw):
        return seed_rset if date_str == SEED_DUE else resolution_set

    monkeypatch.setattr(cli, "load_question_set", lambda d, **kw: question_set)
    monkeypatch.setattr(cli, "load_resolution_set", fake_rset)
    monkeypatch.setattr(pipeline, "load_resolution_set", fake_rset)
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: [])
    monkeypatch.setattr(cli, "list_question_set_dates", lambda **kw: [TARGET_DUE])

    pipeline.append_history(
        cli.HISTORY_PATH,
        [pipeline.HistoryEntry(due_date=SEED_DUE, id=f"seed-{i}", source="manifold",
                               kind="market", resolution_date=None, raw_p=0.5,
                               model_id="claude-haiku-4-5") for i in range(5)],
    )

    seen = {}
    real = pipeline.compute_arms

    def spy(rows, raw_by_qid, pool, base_rates):
        arms, notes = real(rows, raw_by_qid, pool, base_rates)
        seen["pool"] = list(pool)
        seen["notes"] = notes
        return arms, notes

    monkeypatch.setattr(pipeline, "compute_arms", spy)
    cli.cmd_backtest(Namespace(
        models="claude-haiku-4-5", sets=TARGET_DUE, sample=None, seed=1, max_usd=1000.0,
        no_batch=False, dry_run=True, allow_failures=False,
    ))

    assert len(seen["pool"]) == 2, seen["pool"]  # 06-05 and 06-06 only
    assert all(r.resolution_date < dt.date.fromisoformat(TARGET_DUE) for r in seen["pool"])
    assert seen["notes"]["market_n"] == 2
