"""The non-learning control arm.

The CONTROL_SHRINK blend:
`p_control = 0.8*p_raw + 0.2*anchor`.
For market questions, anchor = the market's own freeze-time price (fallback 0.5 if "N/A"). For
dataset questions, anchor = the past-only base rate of `resolved_to` for that (source,
horizon-bucket), estimated only from resolution rows that resolved strictly before the target
set's due date -- never a row that could leak the future.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

CONTROL_SHRINK = 0.8

# Dataset horizons are nominally +7/30/90/180/365 days (older sets had up to 8); bucket to the
# nearest of these so the base rate has enough rows per bucket to mean anything.
CANONICAL_HORIZON_DAYS = (7, 30, 90, 180, 365)


def nearest_horizon_bucket(due_date: dt.date, resolution_date: dt.date) -> int:
    days = (resolution_date - due_date).days
    return min(CANONICAL_HORIZON_DAYS, key=lambda h: abs(h - days))


def market_anchor(freeze_value: str | None) -> float:
    if freeze_value is None or freeze_value == "N/A":
        return 0.5
    try:
        v = float(freeze_value)
    except (TypeError, ValueError):
        return 0.5
    if not (0.0 <= v <= 1.0):
        return 0.5
    return v


@dataclass(frozen=True)
class HistoricalResolution:
    source: str
    bucket: int
    resolved_to: float


def build_dataset_base_rates(
    history: list[HistoricalResolution],
) -> dict[tuple[str, int], float]:
    sums: dict[tuple[str, int], float] = {}
    counts: dict[tuple[str, int], int] = {}
    for h in history:
        key = (h.source, h.bucket)
        sums[key] = sums.get(key, 0.0) + h.resolved_to
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def dataset_anchor(
    base_rates: dict[tuple[str, int], float], source: str, bucket: int
) -> float:
    return base_rates.get((source, bucket), 0.5)


def control_probability(raw_p: float, anchor: float) -> float:
    return CONTROL_SHRINK * raw_p + (1 - CONTROL_SHRINK) * anchor
