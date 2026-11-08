"""Pre-run ceiling cost estimates and the hard --max-usd cap.

A ceiling, not a prediction: it assumes every call uses its FULL max_tokens budget on output
and a generous chars-per-token ratio on input, so the real run should always cost less. The
cap refuses to start a run whose ceiling exceeds --max-usd -- never discovered mid-run.
"""
from __future__ import annotations

from dataclasses import dataclass

from fbe.data import NormalizedRow
from fbe.forecast import CallSpec, build_call_specs, estimate_max_tokens
from fbe.models import ModelSpec

# Conservative chars-per-token ratio for English prose (real tokenizers run ~4 chars/token;
# using a slightly lower number keeps this a true ceiling, never an underestimate).
CHARS_PER_TOKEN = 3.5

# Fixed system-prompt / instructions overhead, in characters, added to every call's input
# estimate (the per-model forecasting instructions in fbe/forecast.py).
SYSTEM_OVERHEAD_CHARS = 1400


class CostCapExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class CostEstimate:
    n_calls: int
    n_rows: int
    ceiling_input_tokens: int
    ceiling_output_tokens: int
    ceiling_usd: float
    use_batch: bool


def _input_chars(spec: CallSpec) -> int:
    return SYSTEM_OVERHEAD_CHARS + len(spec.question) + len(spec.background) + len(
        spec.resolution_criteria
    )


def estimate_cost(
    rows: list[NormalizedRow], model: ModelSpec, use_batch: bool = True
) -> CostEstimate:
    specs = build_call_specs(rows)
    input_tokens = 0
    output_tokens = 0
    for spec in specs:
        input_tokens += int(_input_chars(spec) / CHARS_PER_TOKEN) + 1
        output_tokens += estimate_max_tokens(spec)
    price_in = model.batch_price_in() if use_batch else model.price_in_per_mtok
    price_out = model.batch_price_out() if use_batch else model.price_out_per_mtok
    usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return CostEstimate(
        n_calls=len(specs),
        n_rows=len(rows),
        ceiling_input_tokens=input_tokens,
        ceiling_output_tokens=output_tokens,
        ceiling_usd=usd,
        use_batch=use_batch,
    )


def actual_cost_usd(
    usage: dict[str, dict[str, int]], model: ModelSpec, use_batch: bool
) -> tuple[int, int, float]:
    """Sum real input/output tokens from a BatchOutcome.usage dict and price them at whatever
    was actually used (batch vs. sync pricing) -- the REAL cost, not the pre-run ceiling."""
    input_tokens = sum(u.get("input_tokens", 0) for u in usage.values())
    output_tokens = sum(u.get("output_tokens", 0) for u in usage.values())
    price_in = model.batch_price_in() if use_batch else model.price_in_per_mtok
    price_out = model.batch_price_out() if use_batch else model.price_out_per_mtok
    usd = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return input_tokens, output_tokens, usd


def enforce_cap(estimate: CostEstimate, max_usd: float | None) -> None:
    if max_usd is None:
        return
    if estimate.ceiling_usd > max_usd:
        raise CostCapExceededError(
            f"ceiling cost ${estimate.ceiling_usd:.2f} exceeds --max-usd ${max_usd:.2f} "
            f"({estimate.n_calls} calls, {estimate.ceiling_input_tokens} input tok / "
            f"{estimate.ceiling_output_tokens} output tok ceiling) -- refusing to start"
        )
