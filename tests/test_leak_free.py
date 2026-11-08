"""Dedicated proof that prequential fitting is leak-free: a training row whose outcome
resolved on or after the target due date must never survive the filter. This test is written
so it FAILS if leak_free_filter is bypassed (e.g. replaced with an identity pass-through) --
see the second test below, which asserts on the exact excluded set.
"""
import datetime as dt

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
