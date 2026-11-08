"""Prompt construction and the Anthropic call shape -- one call per QUESTION.

A market question is one call, one probability. A dataset question is one call that returns
a probability for every one of its resolution_dates at once (BUILD_SPEC.md: "dataset questions
return all horizons in one call"). Structured outputs (`output_config.format`) pin the reply to
a JSON schema so parsing never has to guess at prose.

No web/search tools are ever attached here -- a backtest with live search would leak outcomes.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from fbe.data import NormalizedRow
from fbe.models import ModelSpec

SYSTEM_INSTRUCTIONS = (
    "You are a careful, calibrated forecaster for the ForecastBench benchmark. You will be "
    "given one question (or, for a dataset question, several resolution dates for the same "
    "underlying question) with its background, resolution criteria, and the most recently "
    "observed/market value as of the freeze date. Respond ONLY with your best-calibrated "
    "probability that the question resolves YES (or, for a numeric/threshold dataset question, "
    "that the stated condition holds), for each resolution date asked. Use the full [0,1] range; "
    "do not default to 0.5 out of caution. You have no access to any tool, search, or "
    "information published after the freeze date -- reason only from what is given."
)


@dataclass(frozen=True)
class HorizonItem:
    resolution_date: str | None
    question: str


@dataclass(frozen=True)
class CallSpec:
    qid: str
    source: str
    kind: str  # "market" | "dataset"
    background: str
    resolution_criteria: str
    freeze_datetime: str
    freeze_datetime_value: str
    freeze_datetime_value_explanation: str
    forecast_due_date: str
    horizons: tuple[HorizonItem, ...] = field(default_factory=tuple)

    @property
    def question(self) -> str:
        """Concatenated question text across horizons -- used only for cost estimation."""
        return "\n\n".join(h.question for h in self.horizons)


def build_call_specs(rows: list[NormalizedRow]) -> list[CallSpec]:
    """Group normalized rows (one per market question, or one per dataset-question-horizon)
    back into one CallSpec per underlying question."""
    by_id: dict[str, list[NormalizedRow]] = {}
    order: list[str] = []
    for r in rows:
        if r.id not in by_id:
            by_id[r.id] = []
            order.append(r.id)
        by_id[r.id].append(r)
    specs = []
    for qid in order:
        group = by_id[qid]
        first = group[0]
        specs.append(
            CallSpec(
                qid=qid,
                source=first.source,
                kind=first.kind,
                background=first.background,
                resolution_criteria=first.resolution_criteria,
                freeze_datetime=first.freeze_datetime,
                freeze_datetime_value=first.freeze_datetime_value,
                freeze_datetime_value_explanation=first.freeze_datetime_value_explanation,
                forecast_due_date=first.forecast_due_date,
                horizons=tuple(
                    HorizonItem(resolution_date=r.resolution_date, question=r.question)
                    for r in group
                ),
            )
        )
    return specs


MARKET_MAX_TOKENS = 8000
DATASET_MAX_TOKENS = 12000


def estimate_max_tokens(spec: CallSpec) -> int:
    """Ceiling on output tokens. Thinking models (Sonnet 5 adaptive, Opus 5.5 adaptive-forced)
    can spend thousands of tokens on thinking before ever emitting the structured JSON answer;
    a small max_tokens starves the answer entirely (stop_reason=max_tokens, no JSON). Floors
    here are deliberately generous -- the cost ceiling
    in fbe/cost.py rises accordingly, which is the honest tradeoff."""
    if spec.kind == "market":
        return MARKET_MAX_TOKENS
    # a handful of older sets have up to 8 horizons per dataset question; give extra room.
    extra = 300 * max(0, len(spec.horizons) - 5)
    return DATASET_MAX_TOKENS + extra


def build_prompt(spec: CallSpec) -> str:
    parts = [
        f"Source: {spec.source}",
        f"Forecast due date: {spec.forecast_due_date}",
        f"Freeze datetime: {spec.freeze_datetime}",
        f"Freeze value: {spec.freeze_datetime_value}",
        f"Freeze value explanation: {spec.freeze_datetime_value_explanation}",
        f"Background: {spec.background}",
        f"Resolution criteria: {spec.resolution_criteria}",
    ]
    if spec.kind == "market":
        parts.append(f"Question: {spec.horizons[0].question}")
        parts.append(
            "Respond with your calibrated probability (0 to 1) that this resolves YES."
        )
    else:
        parts.append("This dataset question has multiple resolution dates. For each one:")
        for h in spec.horizons:
            parts.append(f"  - resolution_date {h.resolution_date}: {h.question}")
        parts.append(
            "Respond with one calibrated probability (0 to 1) per resolution_date listed above."
        )
    return "\n\n".join(parts)


def output_schema(spec: CallSpec) -> dict:
    """JSON schema for structured outputs.

    The Claude structured-outputs schema validator rejects numeric constraints
    (`minimum`/`maximum`/...) and `minItems`/`maxItems` outside {0,1} with a 400, and the
    raw `output_config.format.schema` dict is NOT transformed by the SDK before being sent,
    so this schema carries none of those keywords. [0,1]-range and exact-date-set
    enforcement both happen client-side in `parse_response_text()`, so nothing is lost.
    """
    if spec.kind == "market":
        return {
            "type": "object",
            "properties": {
                "probability": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["probability", "reasoning"],
            "additionalProperties": False,
        }
    dates = [h.resolution_date for h in spec.horizons]
    return {
        "type": "object",
        "properties": {
            "forecasts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "resolution_date": {"type": "string", "enum": dates},
                        "probability": {"type": "number"},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["resolution_date", "probability", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["forecasts"],
        "additionalProperties": False,
    }


def build_request_params(spec: CallSpec, model: ModelSpec, max_tokens: int | None = None) -> dict:
    """Build the exact `client.messages.create(**params)` kwargs for this call.

    Per-model quirks (BUILD_SPEC.md): no `temperature` on Sonnet 5 / Opus 5.5 (400); Opus 5.5
    thinking cannot be disabled (effort low); Sonnet 5 thinking set explicitly (adaptive or
    disabled, with effort set); Haiku 4.5 gets no `thinking` param at all.
    """
    params: dict = {
        "model": model.model_id,
        "max_tokens": max_tokens or estimate_max_tokens(spec),
        "system": SYSTEM_INSTRUCTIONS,
        "messages": [{"role": "user", "content": build_prompt(spec)}],
        "output_config": {"format": {"type": "json_schema", "schema": output_schema(spec)}},
    }
    if model.thinking_mode == "adaptive_forced":
        params["thinking"] = {"type": "adaptive"}
        params["output_config"]["effort"] = model.default_effort
    elif model.thinking_mode == "adaptive_or_disabled":
        params["thinking"] = {"type": "disabled"}
        params["output_config"]["effort"] = model.default_effort
    # thinking_mode == "none" (Haiku 4.5): omit `thinking` entirely.
    return params


class ParseError(RuntimeError):
    pass


def parse_response_text(spec: CallSpec, text: str) -> dict[str | None, tuple[float, str]]:
    """Parse a (supposedly) schema-conformant JSON reply into {resolution_date: (p, reasoning)}.

    Raises ParseError on anything truncated or unparseable -- callers must treat that as a
    recorded failure (impute 0.5, count it), never silently.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(f"{spec.qid}: invalid JSON ({e})") from e

    out: dict[str | None, tuple[float, str]] = {}
    if spec.kind == "market":
        try:
            p = float(data["probability"])
            reasoning = str(data.get("reasoning", ""))
        except (KeyError, TypeError, ValueError) as e:
            raise ParseError(f"{spec.qid}: missing/invalid probability ({e})") from e
        if not (0.0 <= p <= 1.0) or p != p:  # p != p catches NaN
            raise ParseError(f"{spec.qid}: probability {p} out of [0,1]")
        out[None] = (p, reasoning)
        return out

    try:
        forecasts = data["forecasts"]
    except (KeyError, TypeError) as e:
        raise ParseError(f"{spec.qid}: missing 'forecasts' ({e})") from e
    expected = {h.resolution_date for h in spec.horizons}
    for item in forecasts:
        try:
            rd = item["resolution_date"]
            p = float(item["probability"])
            reasoning = str(item.get("reasoning", ""))
        except (KeyError, TypeError, ValueError) as e:
            raise ParseError(f"{spec.qid}: malformed forecast item ({e})") from e
        if not (0.0 <= p <= 1.0) or p != p:
            raise ParseError(f"{spec.qid}: probability {p} out of [0,1] for {rd}")
        out[rd] = (p, reasoning)
    missing = expected - set(out)
    if missing:
        raise ParseError(f"{spec.qid}: missing resolution_dates {sorted(missing)}")
    return out


def stub_response_text(spec: CallSpec, seed: int = 0) -> str:
    """Deterministic fake model for --dry-run: a stable pseudo-probability per (qid, date)
    derived from a hash, so the whole pipeline runs for $0 and reproducibly."""

    def fake_p(key: str) -> float:
        h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        return 0.01 + (int(h[:8], 16) % 9800) / 10000.0  # in [0.01, 0.99)

    if spec.kind == "market":
        p = fake_p(spec.qid)
        return json.dumps({"probability": p, "reasoning": "stub (--dry-run)"})
    forecasts = [
        {
            "resolution_date": h.resolution_date,
            "probability": fake_p(f"{spec.qid}:{h.resolution_date}"),
            "reasoning": "stub (--dry-run)",
        }
        for h in spec.horizons
    ]
    return json.dumps({"forecasts": forecasts})
