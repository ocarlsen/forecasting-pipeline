"""CLI hardening: dry-run output isolation, paid-reply persistence and --from-raw, history
dedupe on re-run, and clear refusals for a bad date or a model whose cutoff forbids the round.
Everything is offline; any model call is stubbed or asserted absent.
"""
import json
from argparse import Namespace

import pytest

import fbe.batch as batch_mod
import fbe.cli as cli
import fbe.pipeline as pipeline

DATE = "2027-01-01"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, question_set, resolution_set, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_question_set", lambda d, **kw: question_set)
    monkeypatch.setattr(cli, "load_resolution_set", lambda d, **kw: resolution_set)
    monkeypatch.setattr(pipeline, "load_resolution_set", lambda d, **kw: resolution_set)
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: [])


def _args(**over):
    base = dict(
        model="claude-haiku-4-5", date=DATE, org="exampleorg", max_usd=1000.0, seed=1,
        no_batch=False, dry_run=True, allow_failures=False, resume_batch=None, from_raw=None,
    )
    base.update(over)
    return Namespace(**base)


# A. dry-run output isolation

def test_dry_run_never_touches_a_real_runs_output():
    cli.OUT_DIR.mkdir()
    real = cli.OUT_DIR / f"{DATE}.exampleorg.1.json"
    real.write_text("REAL")
    cli.cmd_live(_args())
    assert real.read_text() == "REAL"
    for n in (1, 2, 3):
        assert (cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"{DATE}.exampleorg.{n}.json").exists()
    assert not (cli.OUT_DIR / f"{DATE}.exampleorg.2.json").exists()


def test_dry_run_guard_refuses_any_path_outside_the_dryrun_dir():
    with pytest.raises(SystemExit, match="REFUSED"):
        cli._guard_dry_run_path(cli.OUT_DIR / f"{DATE}.exampleorg.1.json", dry_run=True)
    cli._guard_dry_run_path(cli.OUT_DIR / f"{DATE}.exampleorg.1.json", dry_run=False)  # real: ok
    cli._guard_dry_run_path(cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / "x.json", dry_run=True)


# B. persistence of paid replies and --from-raw

def test_failure_after_forecast_still_leaves_raw_responses(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("base-rate fetch failed")

    monkeypatch.setattr(pipeline, "public_base_rates", boom)
    with pytest.raises(RuntimeError):
        cli.cmd_live(_args())
    (run_dir,) = list(cli.RUNS_DIR.glob("2*-live-*"))
    saved = json.loads((run_dir / "raw_responses.json").read_text())
    assert saved["texts"], "paid replies were not persisted"
    assert {"texts", "errors", "stop_reasons"} <= set(saved)


def test_from_raw_rescoring_does_not_call_the_model(monkeypatch, capsys):
    cli.cmd_live(_args())
    (first,) = list(cli.RUNS_DIR.glob("2*-live-*"))
    first_texts = json.loads((first / "raw_responses.json").read_text())["texts"]

    def no_model(*a, **kw):
        raise AssertionError("model call attempted during --from-raw")

    monkeypatch.setattr(cli, "_run_forecast", no_model)
    monkeypatch.setattr(batch_mod, "run_stub", no_model)
    monkeypatch.setattr(cli, "_get_client", no_model)
    import time
    time.sleep(1.1)  # run dirs are timestamped to the second
    cli.cmd_live(_args(from_raw=str(first)))
    runs = sorted(cli.RUNS_DIR.glob("2*-live-*"))
    assert len(runs) == 2
    assert json.loads((runs[1] / "raw_responses.json").read_text())["texts"] == first_texts
    for n in (1, 2, 3):
        assert (cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"{DATE}.exampleorg.{n}.json").exists()


def test_from_raw_missing_file_refuses(tmp_path):
    with pytest.raises(SystemExit, match="REFUSED"):
        cli.cmd_live(_args(from_raw=str(tmp_path / "nope")))


# C. history dedupe

def _entry(i, p=0.5):
    return pipeline.HistoryEntry(
        due_date=DATE, id=f"q{i}", source="manifold", kind="market",
        resolution_date=None, raw_p=p, model_id="m",
    )


def test_merge_history_overwrites_instead_of_duplicating():
    first = [_entry(i) for i in range(4)]
    pool = pipeline.merge_history([], first)
    assert len(pool) == 4
    pool = pipeline.merge_history(pool, [_entry(i, p=0.9) for i in range(4)])
    assert len(pool) == 4
    assert all(e.raw_p == 0.9 for e in pool)


def test_append_history_rerun_does_not_grow_file():
    path = cli.RUNS_DIR / "h.jsonl"
    pipeline.append_history(path, [_entry(i) for i in range(4)])
    pipeline.append_history(path, [_entry(i, p=0.7) for i in range(4)])
    lines = path.read_text().splitlines()
    assert len(lines) == 4
    assert all(json.loads(l)["raw_p"] == 0.7 for l in lines)


def test_rerunning_a_dry_live_round_keeps_history_size():
    cli.cmd_live(_args())
    n1 = len(cli.DRYRUN_HISTORY_PATH.read_text().splitlines())
    cli.cmd_live(_args())
    assert len(cli.DRYRUN_HISTORY_PATH.read_text().splitlines()) == n1


# E, F. refusals

def test_cutoff_violation_is_a_refusal_not_a_traceback(capsys):
    with pytest.raises(SystemExit) as ei:
        cli.cmd_live(_args(date="2025-01-01"))  # before Haiku 4.5's cutoff
    assert ei.value.code not in (0, None)
    assert "REFUSED" in capsys.readouterr().err


def test_bad_date_refuses_before_any_fetch(monkeypatch, capsys):
    def fetch(*a, **kw):
        raise AssertionError("fetch attempted")

    monkeypatch.setattr(cli, "load_question_set", fetch)
    with pytest.raises(SystemExit) as ei:
        cli.cmd_live(_args(date="2027-13-45"))
    assert ei.value.code not in (0, None)
    assert "REFUSED" in capsys.readouterr().err
