"""End-to-end --dry-run of `fbe backtest` and `fbe live`: fully offline (question/resolution
set fetches are monkeypatched to the small fixtures), $0 (stub model), producing real files
on disk that must pass `fbe validate`.
"""
import json
from argparse import Namespace

import pytest

import fbe.cli as cli
import fbe.pipeline as pipeline
from fbe.writer import validate_file


@pytest.fixture(autouse=True)
def _offline_dataset_fetches(monkeypatch, question_set, resolution_set, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_question_set", lambda date_str, **kw: question_set)
    monkeypatch.setattr(cli, "load_resolution_set", lambda date_str, **kw: resolution_set)
    monkeypatch.setattr(pipeline, "load_resolution_set", lambda date_str, **kw: resolution_set)
    # public_base_rates() and the default---sets path both call list_question_set_dates();
    # keep these tests offline and scoped to just the one fixture due date.
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: [])
    monkeypatch.setattr(cli, "list_question_set_dates", lambda **kw: ["2027-01-01"])
    yield


def test_backtest_dry_run_end_to_end(capsys):
    args = Namespace(
        models="claude-haiku-4-5",
        sets="2027-01-01",
        sample=None,
        seed=1,
        max_usd=1000.0,
        no_batch=False,
        dry_run=True,
        allow_failures=False,
    )
    cli.cmd_backtest(args)
    run_dirs = list(cli.RUNS_DIR.glob("2*"))  # exclude history.dryrun.jsonl etc
    assert len(run_dirs) == 1
    report = (run_dirs[0] / "report.md").read_text()
    assert "claude-haiku-4-5" in report
    assert "overall" in report
    assert (run_dirs[0] / "raw_responses.json").exists()
    assert (run_dirs[0] / "usage.json").exists()
    scores = json.loads((run_dirs[0] / "scores.json").read_text())
    assert scores["models"] == ["claude-haiku-4-5"]
    assert "2027-01-01" in scores["scores"]["claude-haiku-4-5"]

    # backtest --dry-run must NOT pollute the real history file
    assert not cli.HISTORY_PATH.exists()
    assert cli.DRYRUN_HISTORY_PATH.exists()


def test_live_dry_run_end_to_end_writes_three_valid_files():
    args = Namespace(
        model="claude-haiku-4-5",
        date="2027-01-01",
        org="ocarlsen",
        anon_org="Anonymous 999",
        max_usd=1000.0,
        seed=1,
        no_batch=False,
        dry_run=True,
        allow_failures=False,
        resume_batch=None,
    )
    cli.cmd_live(args)

    expected_market = {"mkt-1", "mkt-2"}
    expected_dataset = {"ds-1": {"2027-01-08", "2027-02-01"}, "ds-2": {"2027-01-08"}}

    for n, arm in ((1, "raw"), (2, "control"), (3, "platt")):
        slug = "ocarlsen" if arm == "raw" else "Anonymous999"
        path = cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"2027-01-01.{slug}.{n}.json"
        assert path.exists()
        data = validate_file(path, expected_market, expected_dataset)
        assert data["model"] == f"claude-haiku-4-5 ({arm})"
        assert len(data["forecasts"]) == 5  # 2 market + 2 ds-1 horizons + 1 ds-2 horizon

    # live --dry-run must NOT pollute the real history file
    assert not cli.HISTORY_PATH.exists()
    assert cli.DRYRUN_HISTORY_PATH.exists()
    lines = cli.DRYRUN_HISTORY_PATH.read_text().splitlines()
    assert len(lines) == 5
    entry = json.loads(lines[0])
    assert entry["model_id"] == "claude-haiku-4-5"


def test_live_refuses_on_mass_failure(monkeypatch):
    """A run where every reply is garbage must refuse to write files, not ship
    all-0.5-imputed ones silently."""
    import fbe.batch as batch_mod

    monkeypatch.setattr(
        batch_mod, "run_stub", lambda specs, seed=0: batch_mod.BatchOutcome(texts={}, errors={})
    )
    args = Namespace(
        model="claude-haiku-4-5",
        date="2027-01-01",
        org="ocarlsen",
        anon_org="Anonymous 999",
        max_usd=1000.0,
        seed=1,
        no_batch=False,
        dry_run=True,
        allow_failures=False,
        resume_batch=None,
    )
    with pytest.raises(SystemExit):
        cli.cmd_live(args)
    assert not (cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / "2027-01-01.ocarlsen.1.json").exists()
    run_dirs = [d for d in cli.RUNS_DIR.iterdir() if d.is_dir()]
    assert any((d / "failures.json").exists() for d in run_dirs)


def test_backtest_budget_stops_before_a_round_that_would_cross_the_cap(capsys):
    # A cap below one round's ceiling: the run must start (warning only), refuse the first
    # round, and still write its report rather than spend anything.
    args = Namespace(
        models="claude-haiku-4-5", sets="2027-01-01", sample=None, seed=1,
        max_usd=0.0001, no_batch=False, dry_run=True, allow_failures=False,
    )
    cli.cmd_backtest(args)
    run_dir = sorted(cli.RUNS_DIR.glob("2*"))[-1]
    report = (run_dir / "report.md").read_text()
    assert "REFUSED (budget" in report
    assert "WARNING: full-run ceiling" in capsys.readouterr().err


def test_load_history_dedupes_repeated_rows(tmp_path):
    e = pipeline.HistoryEntry(
        due_date="2027-01-01", id="X", source="fred", kind="dataset",
        resolution_date="2027-01-08", raw_p=0.3, model_id="m",
    )
    path = tmp_path / "h.jsonl"
    pipeline.append_history(path, [e])
    pipeline.append_history(path, [e])
    assert len(pipeline.load_history(path)) == 1
