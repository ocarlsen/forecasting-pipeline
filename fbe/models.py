"""Model registry: ids, prices, and sourced training-data cutoffs.

Cutoffs are read from Anthropic's models overview docs (never invented). If a model's cutoff
can't be sourced from a URL, `cutoff_date` is None and `allowed_for_backtest()` refuses it --
loudly, not silently -- rather than guess.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


class CutoffUnknownError(RuntimeError):
    """Raised when a model with no sourced training-data cutoff is asked to backtest/forecast
    a set whose due date can't be checked against that cutoff."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    price_in_per_mtok: float
    price_out_per_mtok: float
    # Training-data cutoff and the exact doc URL it was read from on 2026-10-27. None means
    # "could not be sourced" -- the model is refused for backtesting rather than guessed at.
    cutoff_date: dt.date | None
    cutoff_url: str | None
    cutoff_note: str
    # Per-model API quirks (BUILD_SPEC.md "Per-model API rules" section).
    supports_temperature: bool
    thinking_mode: str  # "none" | "adaptive_or_disabled" | "adaptive_forced"
    default_effort: str | None  # effort to send when thinking_mode requires it explicitly

    def allowed_for_backtest(self, forecast_due_date: dt.date) -> None:
        """Raise if this model cannot be backtested on a set with this due date.

        A model can only forecast sets strictly after its own training cutoff would have
        already happened -- i.e. normal use. For BACKTESTING we need the inverse: the set's
        due date must be far enough in the PAST relative to when we're told the model's
        training data ends that using it wouldn't leak the future into the model's own
        training. BUILD_SPEC.md's rule, read literally: "a model is only allowed on sets with
        forecast_due_date > model cutoff" -- i.e. the live-forecast-looking-forward case
        (model must already exist / know about the world up to the set). We enforce exactly
        that literal rule here; a missing cutoff always refuses.
        """
        if self.cutoff_date is None:
            raise CutoffUnknownError(
                f"{self.model_id}: no sourced training-data cutoff -- refusing to backtest "
                f"({self.cutoff_note})"
            )
        if not (forecast_due_date > self.cutoff_date):
            raise CutoffUnknownError(
                f"{self.model_id}: forecast_due_date {forecast_due_date} is not after the "
                f"model's training-data cutoff {self.cutoff_date} ({self.cutoff_url}) -- refusing"
            )

    def batch_price_in(self) -> float:
        return self.price_in_per_mtok * 0.5

    def batch_price_out(self) -> float:
        return self.price_out_per_mtok * 0.5


# Sourced 2026-10-27 from platform.claude.com model pages (WebFetch'd live, URLs below).
MODELS: dict[str, ModelSpec] = {
    "claude-haiku-4-5": ModelSpec(
        model_id="claude-haiku-4-5",
        display_name="Claude Haiku 4.5",
        price_in_per_mtok=1.0,
        price_out_per_mtok=5.0,
        cutoff_date=dt.date(2025, 7, 31),  # "Training data cutoff: Jul 2025"
        cutoff_url="https://platform.claude.com/docs/en/models/haiku-4-5/overview",
        cutoff_note="training data cutoff Jul 2025 (reliable knowledge cutoff Feb 2025)",
        supports_temperature=True,
        thinking_mode="none",  # BUILD_SPEC.md: "Haiku 4.5 no thinking"
        default_effort=None,
    ),
    "claude-sonnet-5": ModelSpec(
        model_id="claude-sonnet-5",
        display_name="Claude Sonnet 5",
        price_in_per_mtok=2.0,
        price_out_per_mtok=10.0,
        cutoff_date=dt.date(2026, 1, 31),  # "Training data cutoff: Jan 2026"
        cutoff_url="https://platform.claude.com/docs/en/models/sonnet-5/overview",
        cutoff_note="training data cutoff Jan 2026 (reliable knowledge cutoff Jan 2026)",
        supports_temperature=False,  # BUILD_SPEC.md: temperature -> 400 on Sonnet 5
        thinking_mode="adaptive_or_disabled",
        default_effort="medium",
    ),
    "claude-opus-5-5": ModelSpec(
        model_id="claude-opus-5-5",
        display_name="Claude Opus 5.5",
        price_in_per_mtok=4.0,
        price_out_per_mtok=20.0,
        cutoff_date=dt.date(2026, 6, 30),  # "Training data cutoff: Jun 2026"
        cutoff_url="https://platform.claude.com/docs/en/about-claude/models/overview.md",
        cutoff_note="training data cutoff Jun 2026, from the current-lineup table "
        "(reliable knowledge cutoff also Jun 2026)",
        supports_temperature=False,  # BUILD_SPEC.md: temperature -> 400 on Opus 5.5
        thinking_mode="adaptive_forced",  # cannot be disabled; use effort instead
        default_effort="low",
    ),
}

CANDIDATE_MODEL_IDS = ("claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5-5")


def get_model(model_id: str) -> ModelSpec:
    try:
        return MODELS[model_id]
    except KeyError as e:
        raise ValueError(
            f"unknown model id {model_id!r}; candidates are {list(MODELS)}"
        ) from e
