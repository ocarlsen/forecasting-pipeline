"""Orchestration shared by `fbe backtest` and `fbe live`: sampling, leak-free training-pool
construction, and running one round (forecast -> raw/control/platt -> score) for one
(model, question set) pair.
"""
from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from fbe.control import (
    CONTROL_SHRINK,
    HistoricalResolution,
    build_dataset_base_rates,
    control_probability,
    dataset_anchor,
    market_anchor,
    nearest_horizon_bucket,
)
from fbe.calibrate import PlattFit, apply_platt_fit, fit_platt_for_type
from fbe.data import (
    NormalizedRow,
    find_resolution,
    list_question_set_dates,
    load_resolution_set,
    resolution_lookup,
)
from fbe.forecast import CallSpec, ParseError, build_call_specs, parse_response_text
from fbe.score import ScoredRow, score, score_always_half


def sample_rows(
    rows: list[NormalizedRow], sample_n: int | None, seed: int
) -> list[NormalizedRow]:
    """Deterministic, stratified-by-type sample of questions (not rows): samples from the
    distinct question ids of each kind proportionally, then returns all rows for the sampled
    ids (so a sampled dataset question still brings along every one of its horizons)."""
    if sample_n is None:
        return rows
    by_id: dict[str, list[NormalizedRow]] = {}
    order: list[str] = []
    for r in rows:
        if r.id not in by_id:
            by_id[r.id] = []
            order.append(r.id)
        by_id[r.id].append(r)
    market_ids = [qid for qid in order if by_id[qid][0].kind == "market"]
    dataset_ids = [qid for qid in order if by_id[qid][0].kind == "dataset"]
    total_q = len(market_ids) + len(dataset_ids)
    if total_q == 0 or sample_n >= total_q:
        return rows
    n_market = max(1, round(sample_n * len(market_ids) / total_q)) if market_ids else 0
    n_dataset = max(1, sample_n - n_market) if dataset_ids else 0
    n_market = min(n_market, len(market_ids))
    n_dataset = min(n_dataset, len(dataset_ids))
    rng = random.Random(seed)
    chosen = sorted(rng.sample(market_ids, n_market)) + sorted(rng.sample(dataset_ids, n_dataset))
    out: list[NormalizedRow] = []
    for qid in chosen:
        out.extend(by_id[qid])
    return out


@dataclass(frozen=True)
class HistoryEntry:
    """One past round's raw (uncalibrated) forecast for one (question, resolution_date) --
    the unit of data Platt's training pool is built from. Persisted as JSONL so `fbe backtest`
    (non-dry-run) and `fbe live` can accumulate calibration history across separate
    invocations. `model_id` is required: history is per-model --
    Platt for model X must never be fit on model Y's raw probabilities."""

    due_date: str
    id: str
    source: str
    kind: str
    resolution_date: str | None  # None for market rows (matches the forecast-file convention)
    raw_p: float
    model_id: str


def history_key(e: HistoryEntry) -> tuple:
    return (e.model_id, e.due_date, e.source, e.id, e.resolution_date)


def merge_history(
    existing: list[HistoryEntry], new: list[HistoryEntry]
) -> list[HistoryEntry]:
    """Merge by (model, set, source, question, horizon); a re-run overwrites its earlier rows
    instead of adding duplicates, so the same round is never double-weighted in a Platt fit."""
    by_key = {history_key(e): e for e in existing}
    for e in new:
        by_key[history_key(e)] = e
    return list(by_key.values())


def append_history(path: Path, entries: list[HistoryEntry]) -> None:
    """Add entries to the history file, overwriting any earlier row with the same key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged = merge_history(load_history(path), entries)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for e in merged:
            f.write(json.dumps(asdict(e)) + "\n")
    tmp.replace(path)


def load_history(path: Path) -> list[HistoryEntry]:
    if not path.exists():
        return []
    # Dedupe on (model, set, question, horizon), last write wins, so re-running a backtest
    # over an overlapping (model, set) never double-weights those rows in the Platt fit.
    by_key: dict[tuple, HistoryEntry] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        e = HistoryEntry(**json.loads(line))
        by_key[(e.model_id, e.due_date, e.source, e.id, e.resolution_date)] = e
    return list(by_key.values())


@dataclass(frozen=True)
class TrainingRow:
    kind: str
    source: str
    bucket: int | None
    raw_p: float
    outcome: float
    resolution_date: dt.date  # the REAL resolution date, for the leak-free check


def leak_free_filter(rows: list[TrainingRow], due_date: dt.date) -> list[TrainingRow]:
    """The one place that enforces prequential training: a row only enters a fit if its
    outcome was knowable strictly before `due_date`. The filter itself is unit-tested in
    tests/test_leak_free.py; that `cmd_backtest` actually calls it is covered by the dry-run
    CLI test in the same file."""
    return [r for r in rows if r.resolution_date < due_date]


def resolve_history_entries(
    entries: list[HistoryEntry], cache_base: Path | None = None, refresh: bool = False
) -> list[TrainingRow]:
    """Join past raw forecasts against their now-known resolution, dropping anything still
    unresolved. Groups by the entry's own due_date to minimize resolution-set fetches. Callers
    that want a per-model pool must filter `entries` by `model_id` themselves before calling
    this (kept general here -- it doesn't need to know about models)."""
    by_due: dict[str, list[HistoryEntry]] = {}
    for e in entries:
        by_due.setdefault(e.due_date, []).append(e)

    out: list[TrainingRow] = []
    for due_date_str, group in by_due.items():
        rset = load_resolution_set(due_date_str, cache_base=cache_base, refresh=refresh)
        lookup = resolution_lookup(rset)
        origin_due = dt.date.fromisoformat(due_date_str)
        for e in group:
            row = find_resolution(lookup, e.id, e.resolution_date)
            if row is None:
                continue  # not resolved yet
            resolution_date_val = row.get("resolution_date")
            if resolution_date_val in (None, "N/A"):
                continue
            real_resolution_date = dt.date.fromisoformat(resolution_date_val)
            outcome = float(row["resolved_to"])
            bucket = (
                nearest_horizon_bucket(origin_due, real_resolution_date)
                if e.kind == "dataset"
                else None
            )
            out.append(
                TrainingRow(
                    kind=e.kind,
                    source=e.source,
                    bucket=bucket,
                    raw_p=e.raw_p,
                    outcome=outcome,
                    resolution_date=real_resolution_date,
                )
            )
    return out


def public_base_rates(
    due_date: dt.date, cache_base: Path | None = None, refresh: bool = False
) -> dict[tuple[str, int], float]:
    """The control arm's dataset anchor: a (source, horizon-bucket) base rate built from the
    PUBLIC resolution record -- every earlier published set's resolution rows -- never from
    this model's own forecast history (those are two different
    concerns; Platt trains on our own model's raw-p/outcome pairs, but the control blend's
    anchor is supposed to be a model-independent reference point).

    Leak-free the same way as Platt's training pool: only resolution rows with
    `resolution_date < due_date` (strict) are used, regardless of which set they came from."""
    history: list[HistoricalResolution] = []
    for set_date_str in list_question_set_dates(cache_base=cache_base, refresh=refresh):
        set_due = dt.date.fromisoformat(set_date_str)
        if not (set_due < due_date):
            continue
        rset = load_resolution_set(set_date_str, cache_base=cache_base, refresh=refresh)
        if rset is None:
            continue
        origin_due = dt.date.fromisoformat(rset.get("forecast_due_date", set_date_str))
        for r in rset.get("resolutions", []):
            if isinstance(r.get("id"), list) or not r.get("resolved"):
                continue
            rd = r.get("resolution_date")
            if rd in (None, "N/A"):
                continue
            resolution_date = dt.date.fromisoformat(rd)
            if not (resolution_date < due_date):
                continue  # leak-free: the same strict check as leak_free_filter()
            bucket = nearest_horizon_bucket(origin_due, resolution_date)
            history.append(
                HistoricalResolution(
                    source=r["source"], bucket=bucket, resolved_to=float(r["resolved_to"])
                )
            )
    return build_dataset_base_rates(history)


@dataclass
class ArmResult:
    raw_p: float
    control_p: float
    platt_p: float


def forecast_specs(
    specs: list[CallSpec], texts: dict[str, str], errors: dict[str, str]
) -> tuple[dict[str, dict[str | None, float]], int, dict[str, str]]:
    """Parse every call's reply text into {qid: {resolution_date: raw_p}}. A truncated or
    unparseable reply (or one batch/sync already flagged as errored) is imputed to 0.5 for
    every horizon of that question and counted as a failure -- never silently. Returns
    (raw_by_qid, n_failures, failure_reasons) -- failure_reasons is used to write failures.json."""
    out: dict[str, dict[str | None, float]] = {}
    failure_reasons: dict[str, str] = {}
    for spec in specs:
        if spec.qid in errors:
            out[spec.qid] = {h.resolution_date: 0.5 for h in spec.horizons}
            failure_reasons[spec.qid] = errors[spec.qid]
            continue
        text = texts.get(spec.qid)
        if text is None:
            out[spec.qid] = {h.resolution_date: 0.5 for h in spec.horizons}
            failure_reasons[spec.qid] = "no reply text and no recorded error"
            continue
        try:
            parsed = parse_response_text(spec, text)
        except ParseError as e:
            out[spec.qid] = {h.resolution_date: 0.5 for h in spec.horizons}
            failure_reasons[spec.qid] = f"parse_error: {e}"
            continue
        out[spec.qid] = {rd: p for rd, (p, _reasoning) in parsed.items()}
    return out, len(failure_reasons), failure_reasons


def compute_arms(
    rows: list[NormalizedRow],
    raw_by_qid: dict[str, dict[str | None, float]],
    platt_training_pool: list[TrainingRow],
    dataset_base_rates: dict[tuple[str, int], float],
) -> tuple[dict[tuple[str, str | None], ArmResult], dict[str, object]]:
    """Fits Platt (per type) from `platt_training_pool` (this model's own past raw-p/outcome
    pairs), then computes raw/control/platt for every row using `dataset_base_rates` (the
    PUBLIC, model-independent base rates from `public_base_rates()`) as the control arm's
    dataset anchor. Returns (results keyed by (qid, resolution_date), notes)."""
    market_pairs = [(r.raw_p, r.outcome) for r in platt_training_pool if r.kind == "market"]
    dataset_pairs = [(r.raw_p, r.outcome) for r in platt_training_pool if r.kind == "dataset"]
    market_fit, market_note = fit_platt_for_type(market_pairs)
    dataset_fit, dataset_note = fit_platt_for_type(dataset_pairs)

    due_date = dt.date.fromisoformat(rows[0].forecast_due_date) if rows else None
    results: dict[tuple[str, str | None], ArmResult] = {}
    for row in rows:
        raw_p = raw_by_qid[row.id][row.resolution_date]
        if row.kind == "market":
            anchor = market_anchor(row.freeze_datetime_value)
            platt_p = apply_platt_fit(market_fit, raw_p)
        else:
            assert due_date is not None
            resolution_date = dt.date.fromisoformat(row.resolution_date)
            bucket = nearest_horizon_bucket(due_date, resolution_date)
            anchor = dataset_anchor(dataset_base_rates, row.source, bucket)
            platt_p = apply_platt_fit(dataset_fit, raw_p)
        control_p = control_probability(raw_p, anchor)
        results[(row.id, row.resolution_date)] = ArmResult(
            raw_p=raw_p, control_p=control_p, platt_p=platt_p
        )
    notes = {
        "market_platt": market_note,
        "dataset_platt": dataset_note,
        "market_n": len(market_pairs),
        "dataset_n": len(dataset_pairs),
    }
    return results, notes


def score_rows_for_arm(
    rows: list[NormalizedRow],
    arm_results: dict[tuple[str, str | None], ArmResult],
    outcomes: dict[tuple[str, str | None], float],
    arm: str,
):
    scored = []
    for row in rows:
        key = (row.id, row.resolution_date)
        if key not in outcomes:
            continue
        p = getattr(arm_results[key], f"{arm}_p")
        scored.append(ScoredRow(kind=row.kind, outcome=outcomes[key], p=p))
    return score(scored), scored
