"""The --no-batch flag must actually select the synchronous path, and
--resume-batch must resume an existing batch rather than creating a new one."""
import os
from unittest.mock import MagicMock

import pytest

import fbe.batch as batch_mod
import fbe.cli as cli
from fbe.data import normalize_question_set
from fbe.forecast import build_call_specs
from fbe.models import get_model


def test_dry_run_never_touches_batch_or_sync(question_set, monkeypatch):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    model = get_model("claude-haiku-4-5")

    def _boom(*a, **kw):
        raise AssertionError("dry-run must not call run_batch/run_sync/the Anthropic client")

    monkeypatch.setattr(batch_mod, "run_batch", _boom)
    monkeypatch.setattr(batch_mod, "run_sync", _boom)
    monkeypatch.setattr(cli, "_get_client", _boom)

    outcome = cli._run_forecast(model, specs, dry_run=True, seed=1, no_batch=True, run_dir=None, resume_batch=None)
    assert outcome.texts  # stub produced something


def test_no_batch_flag_selects_run_sync(question_set, monkeypatch):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    model = get_model("claude-haiku-4-5")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-never-sent")
    fake_client = object()
    monkeypatch.setattr(cli, "_get_client", lambda: fake_client)

    called = {}

    def fake_run_sync(client, specs_, model_):
        called["sync"] = (client, specs_, model_)
        return batch_mod.BatchOutcome()

    def fake_run_batch(*a, **kw):
        raise AssertionError("no_batch=True must not call run_batch")

    monkeypatch.setattr(batch_mod, "run_sync", fake_run_sync)
    monkeypatch.setattr(batch_mod, "run_batch", fake_run_batch)

    cli._run_forecast(model, specs, dry_run=False, seed=1, no_batch=True, run_dir=None, resume_batch=None)
    assert called["sync"][0] is fake_client


def test_without_no_batch_flag_selects_run_batch(question_set, monkeypatch):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    model = get_model("claude-haiku-4-5")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-never-sent")
    monkeypatch.setattr(cli, "_get_client", lambda: object())

    called = {}

    def fake_run_batch(client, specs_, model_, run_dir=None):
        called["batch"] = True
        return batch_mod.BatchOutcome()

    def fake_run_sync(*a, **kw):
        raise AssertionError("no_batch=False must not call run_sync")

    monkeypatch.setattr(batch_mod, "run_batch", fake_run_batch)
    monkeypatch.setattr(batch_mod, "run_sync", fake_run_sync)

    cli._run_forecast(model, specs, dry_run=False, seed=1, no_batch=False, run_dir=None, resume_batch=None)
    assert called.get("batch")


def test_resume_batch_calls_run_batch_resume_not_run_batch(question_set, monkeypatch):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    model = get_model("claude-haiku-4-5")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-never-sent")
    monkeypatch.setattr(cli, "_get_client", lambda: object())

    called = {}

    def fake_resume(client, batch_id, specs_, run_dir=None):
        called["resume_id"] = batch_id
        return batch_mod.BatchOutcome()

    def fake_run_batch(*a, **kw):
        raise AssertionError("--resume-batch must not create a NEW batch")

    monkeypatch.setattr(batch_mod, "run_batch_resume", fake_resume)
    monkeypatch.setattr(batch_mod, "run_batch", fake_run_batch)

    cli._run_forecast(
        model, specs, dry_run=False, seed=1, no_batch=False, run_dir=None, resume_batch="batch_abc123"
    )
    assert called["resume_id"] == "batch_abc123"
