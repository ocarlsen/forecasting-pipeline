"""`fbe` command-line entry point."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

from fbe import batch as batch_mod
from fbe import pipeline, writer
from fbe.calibrate import CALIBRATOR_VERSION
from fbe.cost import CostCapExceededError, actual_cost_usd, enforce_cap, estimate_cost
from fbe.data import (
    DatasetFetchError,
    find_resolution,
    list_question_set_dates,
    load_question_set,
    load_resolution_set,
    normalize_question_set,
    resolution_lookup,
)
from fbe.forecast import build_call_specs
from fbe.models import CutoffUnknownError, ModelSpec, get_model
from fbe.score import ScoredRow, score

RUNS_DIR = Path("runs")
OUT_DIR = Path("out")
HISTORY_PATH = RUNS_DIR / "history.jsonl"
DRYRUN_HISTORY_PATH = RUNS_DIR / "history.dryrun.jsonl"

DRYRUN_OUT_SUBDIR = "dryrun"  # a dry run's stub forecasts never share a path with a real run's

MAX_FAILURE_RATE = 0.02  # refuse to write or score a round whose failure rate is above this


def _get_client():
    if os.environ.get("ANTHROPIC_API_KEY") is None:
        raise SystemExit(
            "ANTHROPIC_API_KEY is not set. fbe reads the key only from that env var and "
            "never looks at .env files -- set it in your shell to run a live (non --dry-run) call."
        )
    import anthropic

    return anthropic.Anthropic()


def _cutoff_ok(model: ModelSpec, due_date: dt.date) -> bool:
    try:
        model.allowed_for_backtest(due_date)
        return True
    except CutoffUnknownError:
        return False


def _resolve_common_sets(
    model_ids: list[str], explicit_sets: str | None, require_resolved: bool = True
) -> list[str]:
    """Shared by `plan`/`backtest`: the set of due dates every requested model can legally be
    used on. With an explicit --sets, refuses loudly (not silent skip) if ANY requested model
    fails the cutoff check on ANY requested set. Without one, defaults to the intersection of
    (cutoff-allowed for every model) AND (a resolution set exists with >0 resolved rows)."""
    models = [get_model(m) for m in model_ids]
    if explicit_sets:
        dates = sorted(set(explicit_sets.split(",")))
        for date_str in dates:
            due = dt.date.fromisoformat(date_str)
            for model in models:
                if not _cutoff_ok(model, due):
                    raise SystemExit(
                        f"REFUSED: set {date_str} is not usable by {model.model_id} "
                        f"(due_date must be after its training cutoff {model.cutoff_date}, "
                        f"{model.cutoff_url}). Remove it from --sets or drop that model."
                    )
        return dates

    all_dates = list_question_set_dates()
    chosen = []
    for date_str in all_dates:
        due = dt.date.fromisoformat(date_str)
        if not all(_cutoff_ok(model, due) for model in models):
            continue
        if require_resolved:
            rset = load_resolution_set(date_str)
            if rset is None:
                continue
            if not any(r.get("resolved") for r in rset.get("resolutions", [])):
                continue
        chosen.append(date_str)
    return chosen


def cmd_sets(args: argparse.Namespace) -> None:
    dates = list_question_set_dates(refresh=args.refresh)
    for d in dates:
        rset = load_resolution_set(d, refresh=args.refresh)
        if rset is None:
            print(f"{d}  (not yet resolved)")
            continue
        n = len(rset.get("resolutions", []))
        n_resolved = sum(1 for r in rset["resolutions"] if r.get("resolved"))
        print(f"{d}  {n_resolved}/{n} resolved")


def _load_normalized(date_str: str):
    qset = load_question_set(date_str)
    rows, stats = normalize_question_set(qset)
    return qset, rows, stats


def cmd_plan(args: argparse.Namespace) -> None:
    model_ids = args.model.split(",")
    set_dates = _resolve_common_sets(model_ids, args.sets)
    print(f"comparing on {len(set_dates)} common sets: {set_dates}")
    grand_total = 0.0
    for date_str in set_dates:
        _, rows, stats = _load_normalized(date_str)
        sampled = pipeline.sample_rows(rows, args.sample, seed=args.seed)
        print(
            f"\n=== {date_str} ({stats.n_market_questions} market Q / "
            f"{stats.n_dataset_questions} dataset Q / {stats.n_combo_dropped} combo dropped) "
            f"-> sampled {args.sample or 'ALL'} ==="
        )
        for model_id in model_ids:
            model = get_model(model_id)
            est = estimate_cost(sampled, model, use_batch=not args.no_batch)
            grand_total += est.ceiling_usd
            print(
                f"  {model_id:22s} {est.n_calls:4d} calls  "
                f"~{est.ceiling_input_tokens:>8,} in / {est.ceiling_output_tokens:>8,} out tok  "
                f"ceiling ${est.ceiling_usd:.4f}"
                + ("  (batch price)" if est.use_batch else "  (sync price)")
            )
    print(f"\nTOTAL ceiling across all models/sets shown: ${grand_total:.4f}")


def _write_report(run_dir: Path, lines: list[str]) -> None:
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")


def _run_forecast(
    model: ModelSpec,
    specs,
    dry_run: bool,
    seed: int,
    no_batch: bool,
    run_dir: Path | None,
    resume_batch: str | None,
):
    """Dispatch one round's calls: the stub under --dry-run, an existing batch under
    --resume-batch, the synchronous path under --no-batch, else a new batch."""
    if dry_run:
        return batch_mod.run_stub(specs, seed=seed)
    client = _get_client()
    if resume_batch:
        return batch_mod.run_batch_resume(client, resume_batch, specs, run_dir=run_dir)
    if no_batch:
        return batch_mod.run_sync(client, specs, model)
    return batch_mod.run_batch(client, specs, model, run_dir=run_dir)


def _guard_dry_run_path(path: Path, dry_run: bool) -> None:
    """A dry run may only write under out/dryrun/. Refuse any other target, so a stub-model
    file can never overwrite a real run's output, whatever path logic changes later."""
    if not dry_run:
        return
    dry_dir = (OUT_DIR / DRYRUN_OUT_SUBDIR).resolve()
    if dry_dir not in path.resolve().parents:
        raise SystemExit(
            f"REFUSED: a --dry-run must not write {path}; dry-run output goes under {dry_dir}"
        )


def _write_raw_responses(path: Path, outcome) -> None:
    path.write_text(
        json.dumps(
            {
                "texts": outcome.texts,
                "errors": outcome.errors,
                "stop_reasons": outcome.stop_reasons,
                "usage": outcome.usage,
            },
            indent=2,
        )
    )


def _load_raw_outcome(run_dir: Path) -> batch_mod.BatchOutcome:
    path = run_dir / "raw_responses.json"
    if not path.exists():
        raise SystemExit(f"REFUSED: {path} not found; --from-raw needs a prior live run's directory")
    data = json.loads(path.read_text())
    return batch_mod.BatchOutcome(
        texts=data.get("texts", {}),
        errors=data.get("errors", {}),
        stop_reasons=data.get("stop_reasons", {}),
        usage=data.get("usage", {}),
    )


def _failure_rate(n_failures: int, n_specs: int) -> float:
    return (n_failures / n_specs) if n_specs else 0.0


def _write_failures_json(path: Path, specs, failure_reasons: dict[str, str], outcome) -> None:
    by_qid = {s.qid: s for s in specs}
    rows = [
        {
            "id": qid,
            "source": by_qid[qid].source if qid in by_qid else None,
            "reason": reason,
            "stop_reason": outcome.stop_reasons.get(qid),
        }
        for qid, reason in failure_reasons.items()
    ]
    path.write_text(json.dumps(rows, indent=2))


def _round_for_model(
    model: ModelSpec,
    rows,
    dry_run: bool,
    seed: int,
    no_batch: bool,
    platt_training_pool,
    dataset_base_rates,
    run_dir: Path | None = None,
):
    specs = build_call_specs(rows)
    outcome = _run_forecast(model, specs, dry_run, seed, no_batch, run_dir, resume_batch=None)
    raw_by_qid, n_failures, failure_reasons = pipeline.forecast_specs(
        specs, outcome.texts, outcome.errors
    )
    arm_results, notes = pipeline.compute_arms(
        rows, raw_by_qid, platt_training_pool, dataset_base_rates
    )
    return specs, outcome, raw_by_qid, arm_results, notes, n_failures, failure_reasons


def _outcomes_for_rows(rows, date_str: str) -> dict:
    rset = load_resolution_set(date_str)
    lookup = resolution_lookup(rset)
    outcomes = {}
    for row in rows:
        r = find_resolution(lookup, row.id, row.resolution_date)
        if r is not None:
            outcomes[(row.id, row.resolution_date)] = float(r["resolved_to"])
    return outcomes


def _score_report_lines(rows, arm_results, outcomes, prefix="  ") -> tuple[list[str], dict]:
    """Per-arm market/dataset/overall Brier (with n), plus always-0.5 and market-copier on the
    SAME rows. Returns (report lines, {arm: ScoreResult-as-dict}) for scores.json."""
    lines = []
    summary = {}
    for arm in ("raw", "control", "platt"):
        s, _scored = pipeline.score_rows_for_arm(rows, arm_results, outcomes, arm)
        fallback = ""
        if s.overall_brier is not None and (s.market_brier is None or s.dataset_brier is None):
            only = "market" if s.dataset_brier is None else "dataset"
            fallback = f" (overall falls back to {only} only -- the other type had 0 resolved rows)"
        lines.append(
            f"{prefix}{arm:7s} market Brier={s.market_brier} (n={s.n_market})  "
            f"dataset Brier={s.dataset_brier} (n={s.n_dataset})  overall={s.overall_brier}{fallback}"
        )
        summary[arm] = {
            "market_brier": s.market_brier,
            "dataset_brier": s.dataset_brier,
            "overall_brier": s.overall_brier,
            "n_market": s.n_market,
            "n_dataset": s.n_dataset,
        }

    half_scored = [
        ScoredRow(kind=row.kind, outcome=outcomes[(row.id, row.resolution_date)], p=0.5)
        for row in rows
        if (row.id, row.resolution_date) in outcomes
    ]
    half = score(half_scored) if half_scored else None
    if half:
        lines.append(f"{prefix}always-0.5 overall={half.overall_brier}")
        summary["always_half"] = {
            "market_brier": half.market_brier,
            "dataset_brier": half.dataset_brier,
            "overall_brier": half.overall_brier,
            "n_market": half.n_market,
            "n_dataset": half.n_dataset,
        }

    copier_scored = [
        ScoredRow(
            kind="market",
            outcome=outcomes[(row.id, row.resolution_date)],
            p=pipeline.market_anchor(row.freeze_datetime_value),
        )
        for row in rows
        if row.kind == "market" and (row.id, row.resolution_date) in outcomes
    ]
    copier = score(copier_scored) if copier_scored else None
    if copier:
        lines.append(
            f"{prefix}market-copier market Brier={copier.market_brier} (n={copier.n_market}) "
            "(dataset: not applicable -- freeze value isn't a probability for dataset questions)"
        )
        summary["market_copier"] = {
            "market_brier": copier.market_brier,
            "n_market": copier.n_market,
        }
    return lines, summary


def cmd_backtest(args: argparse.Namespace) -> None:
    model_ids = args.models.split(",")
    set_dates = _resolve_common_sets(model_ids, args.sets)
    print(f"comparing on {len(set_dates)} common sets: {set_dates}")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    total_ceiling = 0.0
    per_combo_rows = {}
    for date_str in set_dates:
        _, rows, _stats = _load_normalized(date_str)
        sampled = pipeline.sample_rows(rows, args.sample, seed=args.seed)
        per_combo_rows[date_str] = sampled
        for model_id in model_ids:
            model = get_model(model_id)
            est = estimate_cost(sampled, model, use_batch=not args.no_batch)
            total_ceiling += est.ceiling_usd
    # The cap is cumulative on ACTUAL spend: before each (model, set) round, refuse if
    # spent-so-far + that round's ceiling would exceed --max-usd. The full-run ceiling is
    # only a warning, since thinking-model ceilings run ~10x actual spend.
    if total_ceiling > args.max_usd:
        print(
            f"WARNING: full-run ceiling ${total_ceiling:.2f} exceeds --max-usd ${args.max_usd:.2f}; "
            "the run stops at the first round whose ceiling would cross the cap",
            file=sys.stderr,
        )
    spent_usd = 0.0
    budget_stopped = False

    report = [f"# Backtest run {ts}", "", f"Models: {model_ids}", f"Sets: {set_dates}", ""]
    raw_log = {}
    usage_log = {"ceiling_usd_total": total_ceiling, "dry_run": args.dry_run, "rounds": {}}
    scores_json = {"models": model_ids, "sets": set_dates, "scores": {m: {} for m in model_ids}}
    history_path = DRYRUN_HISTORY_PATH if args.dry_run else HISTORY_PATH

    for model_id in model_ids:
        if budget_stopped:
            break
        model = get_model(model_id)
        # prequential within THIS run, seeded with whatever this model has already earned in
        # runs/history.jsonl from a PRIOR (non-dry) backtest/live -- same pool `live` reads.
        history: list[pipeline.HistoryEntry] = [
            e for e in pipeline.load_history(HISTORY_PATH) if e.model_id == model_id
        ]
        report.append(f"\n## {model_id}\n")
        for date_str in set_dates:
            due_date = dt.date.fromisoformat(date_str)
            if not _cutoff_ok(model, due_date):
                report.append(f"- {date_str}: SKIPPED (cutoff)")
                continue
            rows = per_combo_rows[date_str]
            round_ceiling = estimate_cost(rows, model, use_batch=not args.no_batch).ceiling_usd
            if spent_usd + round_ceiling > args.max_usd:
                report.append(
                    f"- {date_str}: REFUSED (budget: spent ${spent_usd:.2f} + next ceiling "
                    f"${round_ceiling:.2f} > cap ${args.max_usd:.2f}); run stopped here"
                )
                budget_stopped = True
                break
            resolved_history = pipeline.resolve_history_entries(history)
            platt_pool = pipeline.leak_free_filter(resolved_history, due_date)
            dataset_base_rates = pipeline.public_base_rates(due_date)

            round_dir = run_dir / model_id / date_str
            (
                specs, outcome, raw_by_qid, arm_results, notes, n_failures, failure_reasons,
            ) = _round_for_model(
                model, rows, args.dry_run, args.seed, args.no_batch,
                platt_pool, dataset_base_rates, run_dir=round_dir,
            )

            in_tok, out_tok, actual_usd = actual_cost_usd(
                outcome.usage, model, use_batch=not args.no_batch
            )
            spent_usd += actual_usd
            usage_log["rounds"][f"{model_id}_{date_str}"] = {
                "input_tokens": in_tok, "output_tokens": out_tok, "actual_usd": actual_usd,
            }

            rate = _failure_rate(n_failures, len(specs))
            if rate > MAX_FAILURE_RATE and not args.allow_failures:
                round_dir.mkdir(parents=True, exist_ok=True)
                _write_failures_json(round_dir / "failures.json", specs, failure_reasons, outcome)
                report.append(
                    f"- {date_str}: REFUSED (mass failure: {n_failures}/{len(specs)} = "
                    f"{rate:.1%} > {MAX_FAILURE_RATE:.0%}; see {round_dir}/failures.json; "
                    "pass --allow-failures to score anyway)"
                )
                continue

            outcomes = _outcomes_for_rows(rows, date_str)
            lines, summary = _score_report_lines(rows, arm_results, outcomes)
            report.append(
                f"- {date_str}: {len(rows)} rows, {len(outcomes)} resolved, "
                f"{n_failures} parse failures -> 0.5 ({rate:.1%})"
            )
            report.append(f"  platt fit: market -> {notes['market_platt']}; dataset -> {notes['dataset_platt']}")
            report.extend(lines)
            scores_json["scores"][model_id][date_str] = summary

            raw_log[f"{model_id}_{date_str}"] = {"texts": outcome.texts, "errors": outcome.errors}

            new_entries = [
                pipeline.HistoryEntry(
                    due_date=date_str, id=row.id, source=row.source, kind=row.kind,
                    resolution_date=row.resolution_date,
                    raw_p=raw_by_qid[row.id][row.resolution_date], model_id=model_id,
                )
                for row in rows
            ]
            history = pipeline.merge_history(history, new_entries)
            pipeline.append_history(history_path, new_entries)

        # per-model aggregate across every set actually scored (not skipped/refused).
        model_scores = scores_json["scores"][model_id]
        if model_scores:
            report.append(f"\n  AGGREGATE across {len(model_scores)} scored set(s):")
            for arm in ("raw", "control", "platt"):
                total_brier = 0.0
                total_n = 0
                for date_str, per_arm in model_scores.items():
                    s = per_arm[arm]
                    if s["overall_brier"] is None:
                        continue
                    n = s["n_market"] + s["n_dataset"]
                    total_brier += s["overall_brier"] * n
                    total_n += n
                avg = (total_brier / total_n) if total_n else None
                report.append(f"    {arm:7s} overall Brier (n-weighted avg) = {avg}  (n={total_n})")

    (run_dir / "raw_responses.json").write_text(json.dumps(raw_log, indent=2))
    (run_dir / "usage.json").write_text(json.dumps(usage_log, indent=2))
    (run_dir / "scores.json").write_text(json.dumps(scores_json, indent=2))
    _write_report(run_dir, report)
    print(f"Wrote {run_dir}/report.md and {run_dir}/scores.json")
    print("\n".join(report))


def cmd_pick(args: argparse.Namespace) -> None:
    """pick never calls a model: it reads a prior `fbe backtest` run's scores.json
    (--run runs/<ts>) and applies the rule to already-computed numbers."""
    run_dir = Path(args.run)
    scores_path = run_dir / "scores.json"
    if not scores_path.exists():
        raise SystemExit(
            f"REFUSED: {scores_path} not found. Run `fbe backtest --models ... --sets ...` "
            "first (with all candidate models you want to pick among), then `fbe pick --run "
            f"{run_dir}`."
        )
    scores_json = json.loads(scores_path.read_text())
    model_ids = scores_json["models"]

    # "common sets" = sets where EVERY model has a recorded score (i.e. wasn't cutoff-skipped).
    common_sets = [
        date_str
        for date_str in scores_json["sets"]
        if all(date_str in scores_json["scores"].get(m, {}) for m in model_ids)
    ]
    print(f"comparing on {len(common_sets)} common sets: {common_sets}")
    if not common_sets:
        raise SystemExit(
            "REFUSED: no set in this run has a recorded score for every candidate model -- "
            "nothing to compare blind."
        )

    rng = random.Random(args.seed)
    labels = ["A", "B", "C"][: len(model_ids)]
    shuffled = model_ids[:]
    rng.shuffle(shuffled)
    mapping = dict(zip(labels, shuffled))

    agg = {}
    for label, model_id in mapping.items():
        total_brier = 0.0
        total_n = 0
        for date_str in common_sets:
            raw = scores_json["scores"][model_id][date_str]["raw"]
            if raw["overall_brier"] is None:
                continue
            n = raw["n_market"] + raw["n_dataset"]
            total_brier += raw["overall_brier"] * n
            total_n += n
        avg = total_brier / total_n if total_n else float("inf")
        model = get_model(model_id)
        agg[label] = (avg, model.price_in_per_mtok + model.price_out_per_mtok, total_n)
        print(f"Model {label}: overall raw Brier = {avg:.4f} (n={total_n})")

    print("\nRule: lowest overall raw Brier on the common sets wins; ties broken by the cheaper model.")
    best_label = min(agg, key=lambda l: (agg[l][0], agg[l][1]))
    print(f"Picked: {best_label} -> revealing mapping...")
    print(f"Mapping: {mapping}")
    print(f"Chosen model: {mapping[best_label]}")

    (run_dir / "pick_mapping.json").write_text(
        json.dumps(
            {
                "mapping": mapping,
                "scores": {l: {"avg_overall_raw_brier": v[0], "n": v[2]} for l, v in agg.items()},
                "common_sets": common_sets,
                "chosen_label": best_label,
                "chosen_model": mapping[best_label],
                "rule": "lowest overall raw Brier on common sets, ties to cheaper",
            },
            indent=2,
        )
    )


def cmd_live(args: argparse.Namespace) -> None:
    try:
        due_date = dt.date.fromisoformat(args.date)
    except ValueError:
        print(
            f"REFUSED: --date {args.date!r} is not a valid ISO date (expected YYYY-MM-DD)",
            file=sys.stderr,
        )
        sys.exit(2)
    model = get_model(args.model)
    try:
        model.allowed_for_backtest(due_date)  # same rule: model must already "exist" for this date
    except CutoffUnknownError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

    qset = load_question_set(args.date)
    rows, stats = normalize_question_set(qset)
    print(
        f"{args.date}: {stats.n_market_questions} market Q, {stats.n_dataset_questions} dataset Q, "
        f"{stats.n_combo_dropped} combo dropped"
    )

    est = estimate_cost(rows, model, use_batch=not args.no_batch)
    print(f"Ceiling: {est.n_calls} calls, ${est.ceiling_usd:.4f}")
    try:
        enforce_cap(est, args.max_usd)
    except CostCapExceededError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)

    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / f"{ts}-live-{args.date}"
    run_dir.mkdir(parents=True, exist_ok=True)

    specs = build_call_specs(rows)
    from_raw = getattr(args, "from_raw", None)
    if from_raw:
        outcome = _load_raw_outcome(Path(from_raw))
        print(f"Re-scoring from {from_raw}/raw_responses.json (no model call)")
    else:
        outcome = _run_forecast(
            model, specs, args.dry_run, args.seed, args.no_batch, run_dir, args.resume_batch
        )
    # Persist the paid replies before anything else can fail (base-rate fetch, coverage
    # check, validation): a later error must never lose a round that was already paid for.
    _write_raw_responses(run_dir / "raw_responses.json", outcome)
    raw_by_qid, n_failures, failure_reasons = pipeline.forecast_specs(
        specs, outcome.texts, outcome.errors
    )
    rate = _failure_rate(n_failures, len(specs))
    if rate > MAX_FAILURE_RATE and not args.allow_failures:
        _write_failures_json(run_dir / "failures.json", specs, failure_reasons, outcome)
        print(
            f"REFUSED: mass failure ({n_failures}/{len(specs)} = {rate:.1%} > "
            f"{MAX_FAILURE_RATE:.0%}); see {run_dir}/failures.json; pass --allow-failures to "
            "write all-0.5-imputed files anyway",
            file=sys.stderr,
        )
        sys.exit(2)

    # The Platt pool is THIS model's own backtest + prior live rounds (never a dry-run's --
    # those live in history.dryrun.jsonl, never HISTORY_PATH). Resolution-set fetches are
    # refreshed so a just-resolved outcome is seen, not a stale cache hit.
    history = [e for e in pipeline.load_history(HISTORY_PATH) if e.model_id == args.model]
    resolved_history = pipeline.resolve_history_entries(history, refresh=True)
    platt_pool = pipeline.leak_free_filter(resolved_history, due_date)
    dataset_base_rates = pipeline.public_base_rates(due_date, refresh=True)
    arm_results, notes = pipeline.compute_arms(rows, raw_by_qid, platt_pool, dataset_base_rates)
    print(f"Platt fit on {notes['market_n']} market / {notes['dataset_n']} dataset rows")
    print(f"  market: {notes['market_platt']}")
    print(f"  dataset: {notes['dataset_platt']}")
    print(f"Parse/stop-reason failures imputed to 0.5: {n_failures} ({rate:.1%})")

    in_tok, out_tok, actual_usd = actual_cost_usd(outcome.usage, model, use_batch=not args.no_batch)
    (run_dir / "usage.json").write_text(
        json.dumps(
            {
                "ceiling_usd": est.ceiling_usd,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "actual_usd": actual_usd,
                "dry_run": args.dry_run,
            },
            indent=2,
        )
    )

    expected_market_ids = {r.id for r in rows if r.kind == "market"}
    expected_dataset_question_dates: dict[str, set[str]] = {}
    for r in rows:
        if r.kind == "dataset":
            expected_dataset_question_dates.setdefault(r.id, set()).add(r.resolution_date)

    out_dir = OUT_DIR / DRYRUN_OUT_SUBDIR if args.dry_run else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for arm in ("raw", "control", "platt"):
        frows = [
            writer.ForecastRow(
                id=row.id,
                source=row.source,
                forecast=getattr(arm_results[(row.id, row.resolution_date)], f"{arm}_p"),
                resolution_date=row.resolution_date,
                reasoning=f"{CALIBRATOR_VERSION if arm == 'platt' else arm} arm",
            )
            for row in rows
        ]
        path = out_dir / writer.filename_for(args.date, args.org, arm)
        _guard_dry_run_path(path, args.dry_run)
        writer.write_forecast_file(
            path, args.org, args.model, arm,
            qset.get("question_set", f"{args.date}-llm.json"),
            frows, expected_market_ids, expected_dataset_question_dates,
        )
        writer.validate_file(path, expected_market_ids, expected_dataset_question_dates)
        written[arm] = str(path)
        print(f"Wrote + validated {path}")

    # Never pollute the real history with a dry run; every entry carries its model_id.
    history_path = DRYRUN_HISTORY_PATH if args.dry_run else HISTORY_PATH
    new_entries = [
        pipeline.HistoryEntry(
            due_date=args.date, id=row.id, source=row.source, kind=row.kind,
            resolution_date=row.resolution_date,
            raw_p=raw_by_qid[row.id][row.resolution_date], model_id=args.model,
        )
        for row in rows
    ]
    pipeline.append_history(history_path, new_entries)
    print(json.dumps(written, indent=2))


def cmd_validate(args: argparse.Namespace) -> None:
    path = Path(args.file)
    data = json.loads(path.read_text())
    expected_market_ids = None
    expected_dataset_question_dates = None
    qset_name = data.get("question_set")
    if qset_name:
        date_str = qset_name.split("-llm.json")[0]
        try:
            qset = load_question_set(date_str)
            rows, _stats = normalize_question_set(qset)
            expected_market_ids = {r.id for r in rows if r.kind == "market"}
            expected_dataset_question_dates = {}
            for r in rows:
                if r.kind == "dataset":
                    expected_dataset_question_dates.setdefault(r.id, set()).add(r.resolution_date)
        except DatasetFetchError:
            print("(could not fetch referenced question set; skipping coverage check)")
    writer.validate_file(path, expected_market_ids, expected_dataset_question_dates)
    print(f"OK: {path} is schema-valid" + (" and >=95% covered" if expected_market_ids else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fbe")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sets")
    sp.add_argument("--refresh", action="store_true")
    sp.set_defaults(func=cmd_sets)

    sp = sub.add_parser("plan")
    sp.add_argument("--model", required=True, help="comma-separated model ids")
    sp.add_argument("--sets", default=None, help="comma-separated due dates (default: common sets)")
    sp.add_argument("--sample", type=int, default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-batch", action="store_true")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("backtest")
    sp.add_argument("--models", required=True)
    sp.add_argument("--sets", default=None, help="comma-separated due dates (default: common sets)")
    sp.add_argument("--sample", type=int, default=None)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-usd", type=float, required=True)
    sp.add_argument("--no-batch", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--allow-failures", action="store_true")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("pick")
    sp.add_argument("--run", required=True, help="a prior `fbe backtest` run dir, e.g. runs/20261028T000000Z")
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_pick)

    sp = sub.add_parser("live")
    sp.add_argument("--model", required=True)
    sp.add_argument("--date", required=True)
    sp.add_argument("--org", required=True)
    sp.add_argument("--max-usd", type=float, required=True)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--no-batch", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument("--allow-failures", action="store_true")
    sp.add_argument("--resume-batch", default=None, help="an existing batch id to resume/poll instead of creating a new one")
    sp.add_argument("--from-raw", default=None, metavar="RUN_DIR", help="re-score from RUN_DIR/raw_responses.json (a prior live run) instead of calling the model")
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser("validate")
    sp.add_argument("file")
    sp.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
