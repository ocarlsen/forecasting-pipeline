"""`fbe pick` reads a prior backtest run's scores.json and applies the rule to
already-computed numbers -- it never calls a model (no stub, no client, no network)."""
import json
from argparse import Namespace

import pytest

import fbe.cli as cli


def _scores_payload():
    return {
        "models": ["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5"],
        "sets": ["2026-09-27", "2026-10-11"],
        "scores": {
            "claude-haiku-4-5": {
                "2026-09-27": {"raw": {"overall_brier": 0.30, "n_market": 10, "n_dataset": 10}},
                "2026-10-11": {"raw": {"overall_brier": 0.28, "n_market": 10, "n_dataset": 10}},
            },
            "claude-sonnet-5": {
                "2026-09-27": {"raw": {"overall_brier": 0.20, "n_market": 10, "n_dataset": 10}},
                "2026-10-11": {"raw": {"overall_brier": 0.22, "n_market": 10, "n_dataset": 10}},
            },
            "claude-opus-5-5": {
                # SKIPPED on 2026-09-27 (e.g. cutoff) -- not a common set for opus
                "2026-10-11": {"raw": {"overall_brier": 0.19, "n_market": 10, "n_dataset": 10}},
            },
        },
    }


def test_pick_never_calls_a_model(tmp_path, monkeypatch):
    import fbe.batch as batch_mod

    def _boom(*a, **kw):
        raise AssertionError("fbe pick must never call a model")

    monkeypatch.setattr(batch_mod, "run_stub", _boom)
    monkeypatch.setattr(batch_mod, "run_batch", _boom)
    monkeypatch.setattr(batch_mod, "run_sync", _boom)

    run_dir = tmp_path / "runs" / "20261028T000000Z"
    run_dir.mkdir(parents=True)
    (run_dir / "scores.json").write_text(json.dumps(_scores_payload()))

    args = Namespace(run=str(run_dir), seed=7)
    cli.cmd_pick(args)

    mapping_path = run_dir / "pick_mapping.json"
    assert mapping_path.exists()
    result = json.loads(mapping_path.read_text())
    # common sets exclude 2026-10-11-only opus -- opus has no score on 2026-09-27, so the
    # common set across ALL THREE models is just the intersection: none, since opus is
    # missing 2026-09-27. Only sets present for every model count.
    assert result["common_sets"] == ["2026-10-11"]


def test_pick_picks_lowest_overall_raw_brier(tmp_path):
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    payload = _scores_payload()
    # make opus available on both sets so the common-set intersection is non-trivial
    payload["scores"]["claude-opus-5-5"]["2026-09-27"] = {
        "raw": {"overall_brier": 0.50, "n_market": 10, "n_dataset": 10}
    }
    (run_dir / "scores.json").write_text(json.dumps(payload))

    args = Namespace(run=str(run_dir), seed=7)
    cli.cmd_pick(args)

    result = json.loads((run_dir / "pick_mapping.json").read_text())
    assert sorted(result["common_sets"]) == ["2026-09-27", "2026-10-11"]
    # sonnet-5 has the lowest average raw Brier ((0.20+0.22)/2=0.21 vs haiku 0.29 vs opus 0.345)
    assert result["chosen_model"] == "claude-sonnet-5"
    assert result["rule"] == "lowest overall raw Brier on common sets, ties to cheaper"


def test_pick_refuses_when_scores_file_missing(tmp_path):
    run_dir = tmp_path / "runs" / "none"
    args = Namespace(run=str(run_dir), seed=7)
    with pytest.raises(SystemExit):
        cli.cmd_pick(args)
