"""`fbe live`'s Platt pool must be built from THIS model's own backtest + prior
live history (runs/history.jsonl), filtered by model_id, then the usual leak-free filter.
With >=50 resolved rows before the target due date, the fit must be non-identity (not just
reported as "fit", but a real Platt a/b away from the untrained 1.0/0.0 identity).
"""
import datetime as dt
import json
from argparse import Namespace

import pytest

import fbe.cli as cli
import fbe.pipeline as pipeline


SEED_DUE_DATE = "2026-06-01"
LIVE_DUE_DATE = "2027-01-01"
N_SEED_ROWS = 60


def _seed_resolution_set():
    # Half resolve YES, half NO -- raw_p values are spread so a real logistic fit has
    # something to do (not a degenerate single-point fit).
    rows = []
    for i in range(N_SEED_ROWS):
        rows.append(
            {
                "id": f"seed-{i}",
                "source": "manifold",
                "direction": None,
                "resolution_date": "2026-06-05",
                "resolved_to": 1.0 if i % 2 == 0 else 0.0,
                "resolved": True,
            }
        )
    return {
        "forecast_due_date": SEED_DUE_DATE,
        "question_set": f"{SEED_DUE_DATE}-llm.json",
        "resolutions": rows,
    }


@pytest.fixture(autouse=True)
def _offline(monkeypatch, question_set, resolution_set, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_question_set", lambda date_str, **kw: question_set)

    seed_rset = _seed_resolution_set()

    def fake_load_resolution_set(date_str, **kw):
        if date_str == SEED_DUE_DATE:
            return seed_rset
        if date_str == LIVE_DUE_DATE:
            return resolution_set
        return None

    monkeypatch.setattr(cli, "load_resolution_set", fake_load_resolution_set)
    monkeypatch.setattr(pipeline, "load_resolution_set", fake_load_resolution_set)
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: [SEED_DUE_DATE])
    yield


def _seed_history(model_id: str):
    """Simulates "a backtest already ran (non-dry) and appended to runs/history.jsonl" --
    the exact mechanism `fbe backtest` uses (pipeline.append_history to HISTORY_PATH)."""
    entries = [
        pipeline.HistoryEntry(
            due_date=SEED_DUE_DATE,
            id=f"seed-{i}",
            source="manifold",
            kind="market",
            resolution_date=None,
            raw_p=0.2 + 0.6 * (i / N_SEED_ROWS),  # spread across (0.2, 0.8)
            model_id=model_id,
        )
        for i in range(N_SEED_ROWS)
    ]
    pipeline.append_history(cli.HISTORY_PATH, entries)


def test_live_platt_is_non_identity_given_enough_prior_history(capsys):
    _seed_history("claude-haiku-4-5")

    args = Namespace(
        model="claude-haiku-4-5",
        date=LIVE_DUE_DATE,
        org="ocarlsen",
        max_usd=1000.0,
        seed=1,
        no_batch=False,
        dry_run=True,
        allow_failures=False,
        resume_batch=None,
    )
    cli.cmd_live(args)
    out = capsys.readouterr().out
    assert "Platt fit on 60 market / 0 dataset rows" in out
    market_line = next(l for l in out.splitlines() if l.strip().startswith("market:"))
    assert "identity" not in market_line
    assert "fit on 60" in market_line


def test_live_platt_ignores_a_different_models_history(capsys):
    # Seed history under a DIFFERENT model id -- the live run (claude-haiku-4-5) must not
    # use it, and should fall back to identity (0 rows available for ITS id).
    _seed_history("claude-sonnet-5")

    args = Namespace(
        model="claude-haiku-4-5",
        date=LIVE_DUE_DATE,
        org="ocarlsen",
        max_usd=1000.0,
        seed=1,
        no_batch=False,
        dry_run=True,
        allow_failures=False,
        resume_batch=None,
    )
    cli.cmd_live(args)
    out = capsys.readouterr().out
    assert "Platt fit on 0 market / 0 dataset rows" in out
    market_line = next(l for l in out.splitlines() if l.strip().startswith("market:"))
    assert "identity" in market_line
