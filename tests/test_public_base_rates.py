"""SHOULD-FIX 7: the control arm's dataset anchor must come from the PUBLIC resolution
record, leak-free the same way Platt's training pool is -- a resolution row that resolves on
or after the target due date must never affect the base rate.
"""
import datetime as dt

import fbe.pipeline as pipeline


def _rset(due_date_str, rows):
    return {
        "forecast_due_date": due_date_str,
        "question_set": f"{due_date_str}-llm.json",
        "resolutions": rows,
    }


def _row(qid, resolution_date, resolved_to, resolved=True, source="acled"):
    return {
        "id": qid,
        "source": source,
        "direction": None,
        "resolution_date": resolution_date,
        "resolved_to": resolved_to,
        "resolved": resolved,
    }


def test_public_base_rates_excludes_rows_on_or_after_due_date(monkeypatch):
    # Set A (due 2026-11-01): one row resolves well BEFORE the target D -- counts.
    # Set B (due 2026-12-01): one row resolves ON/AFTER D -- must be excluded (the leak row).
    set_a = _rset("2026-11-01", [_row("a1", "2026-11-08", 1.0)])
    set_b = _rset("2026-12-01", [_row("b1", "2027-06-01", 0.0)])  # resolves AFTER D

    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: ["2026-11-01", "2026-12-01"])

    def fake_load(date_str, **kw):
        return {"2026-11-01": set_a, "2026-12-01": set_b}.get(date_str)

    monkeypatch.setattr(pipeline, "load_resolution_set", fake_load)

    due_date = dt.date(2027, 1, 1)
    rates = pipeline.public_base_rates(due_date)

    # only set_a's row should have entered the pool -- acled/bucket(7) base rate is exactly 1.0
    bucket = pipeline.nearest_horizon_bucket(dt.date(2026, 11, 1), dt.date(2026, 11, 8))
    assert rates[("acled", bucket)] == 1.0

    # If the leak row (b1, resolved_to=0.0, resolution_date AFTER D) had been included, the
    # bucket rate would be diluted toward 0.5 (or a second bucket key would appear). Either
    # way the asserted rate above (1.0, exactly a1's value) is the proof it was excluded.
    b_bucket = pipeline.nearest_horizon_bucket(dt.date(2026, 12, 1), dt.date(2027, 6, 1))
    if b_bucket != bucket:
        assert ("acled", b_bucket) not in rates


def test_public_base_rates_only_uses_sets_due_before_target(monkeypatch):
    # A set due ON/AFTER D must be skipped entirely, even if one of its rows superficially
    # looks resolved before D (shouldn't happen in real data, but the function must not rely
    # on that -- it already skips by `set_due < due_date` before even looking at rows).
    set_future = _rset("2027-02-01", [_row("f1", "2026-01-01", 1.0)])
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: ["2027-02-01"])
    monkeypatch.setattr(pipeline, "load_resolution_set", lambda date_str, **kw: set_future)

    due_date = dt.date(2027, 1, 1)
    rates = pipeline.public_base_rates(due_date)
    assert rates == {}


def test_public_base_rates_skips_unresolved_and_combo_rows(monkeypatch):
    rset = _rset(
        "2026-11-01",
        [
            _row("r1", "2026-11-08", 1.0, resolved=False),  # unresolved -- skip
            {"id": ["c1", "c2"], "source": "acled", "resolution_date": "2026-11-08",
             "resolved_to": 1.0, "resolved": True},  # combo -- skip
            _row("r2", "2026-11-08", 0.0),  # the only real row
        ],
    )
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: ["2026-11-01"])
    monkeypatch.setattr(pipeline, "load_resolution_set", lambda date_str, **kw: rset)

    rates = pipeline.public_base_rates(dt.date(2027, 1, 1))
    bucket = pipeline.nearest_horizon_bucket(dt.date(2026, 11, 1), dt.date(2026, 11, 8))
    assert rates[("acled", bucket)] == 0.0  # only r2 counted
