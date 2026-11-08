# Method

The three arms, how they are scored, and the rules that keep the calibration leak-free.

## The three arms

- **raw**: the model's own probability, straight from one Anthropic API call per question
  (`fbe/forecast.py`). No calibration, no blending.
- **control**: a non-learning blend, `p_control = 0.8*p_raw + 0.2*anchor` (`fbe/control.py`).
  `anchor` is the market's own freeze-time price for market questions (0.5 if unavailable), or
  the past-only base rate of outcomes for that (source, horizon-bucket) for dataset questions. It
  exists so a Brier improvement from the `platt` arm is never credited to "calibration" when a
  fixed blend would have done as well.
- **platt**: Platt scaling (`fbe/calibrate.py`), `p = sigmoid(a*logit(p_raw) + b)`, with `(a, b)`
  fit by L2-penalized logistic regression on this model's own past raw forecasts against their
  resolved outcomes, fit separately for market and dataset questions. The penalty pulls `(a, b)`
  toward the identity (1, 0), not toward (0, 0), so a starved fit decays toward "trust the raw
  model" rather than collapsing to a constant. Raw p is clipped to [0.01, 0.99] before the logit.
  Below 50 training rows for a type, Platt is the identity for that type and the report says so.

## Scoring

`fbe/score.py` reports plain Brier per type and overall (mean of market Brier and dataset Brier,
equal weight), alongside always-0.5, market-copier and the control on the same rows.

This is not FRI's difficulty-adjusted Brier Index. That index weights questions by how hard the
field as a whole found them; this repo never tries to approximate it. Compare this repo's plain
Brier against its own baselines on its own rows, not against the public leaderboard's numbers.

## Leak-free calibration

Fitting Platt, and building the control's dataset base rates, on past rounds is prequential: for
a target set with due date `D`, only resolutions with `resolution_date < D` from earlier rounds
enter the training pool (`fbe/pipeline.py:leak_free_filter`, tested directly in
`tests/test_leak_free.py`). A dataset question's resolution can be up to a year after its own
round's due date, so even a past round can still contain rows that have not resolved by `D`.
Those rows are excluded too, not just whole earlier rounds.

`fbe live`'s Platt pool is that model's own rows from `runs/history.jsonl` (backtest plus prior
live rounds), never another model's. A brand-new entry's first live round has no history, so its
Platt fit is the identity; there is nothing to calibrate against yet.

The control arm's dataset anchor is a separate, model-independent pool: `public_base_rates()`
scans every earlier published resolution set's own rows, not this model's history, under the
same `resolution_date < D` rule, bucketed by (source, horizon-bucket). Platt calibrates what this
model's raw probabilities mean; the control's anchor is a reference point no model-specific
history can influence.

## Model cutoffs

A model is refused for any round whose due date is not after that model's own training-data
cutoff (`fbe/models.py`). Using a model on a date before it finished training risks the model
having memorized the answer. If a cutoff cannot be sourced from Anthropic's own docs, the model
is refused rather than guessed at.

| Model | Training data cutoff | Source |
|---|---|---|
| `claude-haiku-4-5` | Jul 2025 (reliable knowledge: Feb 2025) | <https://platform.claude.com/docs/en/models/haiku-4-5/overview> |
| `claude-sonnet-5` | Jan 2026 | <https://platform.claude.com/docs/en/models/sonnet-5/overview> |
| `claude-opus-5-5` | Jun 2026 | <https://platform.claude.com/docs/en/about-claude/models/overview.md> (current-lineup table) |

Cutoffs were read from those pages on 2026-10-27.

## No web access

The forecast prompt gives the model the question, background, resolution criteria, the freeze
value with its explanation and date, and the due date. It has no web tools. A backtest with live
search would leak outcomes.
