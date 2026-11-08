"""Brier scoring: per question type and overall, plus reference baselines.

Overall = mean(market Brier, dataset Brier), equal weight regardless of how many rows each
type has. This is plain Brier, not FRI's difficulty-adjusted Brier Index -- see README.md for
why those two numbers differ and shouldn't be compared directly to the public leaderboard.
"""
from __future__ import annotations

from dataclasses import dataclass


def brier(p: float, outcome: float) -> float:
    return (p - outcome) ** 2


@dataclass(frozen=True)
class ScoredRow:
    kind: str  # "market" | "dataset"
    outcome: float  # 0.0 or 1.0 (resolved_to)
    p: float


@dataclass(frozen=True)
class ScoreResult:
    market_brier: float | None
    dataset_brier: float | None
    overall_brier: float | None
    n_market: int
    n_dataset: int


def _mean_brier(rows: list[ScoredRow]) -> float | None:
    if not rows:
        return None
    return sum(brier(r.p, r.outcome) for r in rows) / len(rows)


def score(rows: list[ScoredRow]) -> ScoreResult:
    market = [r for r in rows if r.kind == "market"]
    dataset = [r for r in rows if r.kind == "dataset"]
    mb = _mean_brier(market)
    db = _mean_brier(dataset)
    if mb is not None and db is not None:
        overall = (mb + db) / 2
    else:
        overall = mb if mb is not None else db
    return ScoreResult(mb, db, overall, len(market), len(dataset))


def score_always_half(rows: list[ScoredRow]) -> ScoreResult:
    return score([ScoredRow(r.kind, r.outcome, 0.5) for r in rows])


def score_market_copier(rows: list[ScoredRow], anchors: dict[int, float]) -> ScoreResult:
    """`anchors` maps id(row)->freeze-derived anchor; only market rows have a meaningful
    market-price anchor, so dataset rows are dropped from this baseline (see README.md)."""
    market_only = [r for r in rows if r.kind == "market" and id(r) in anchors]
    return score([ScoredRow(r.kind, r.outcome, anchors[id(r)]) for r in market_only])
