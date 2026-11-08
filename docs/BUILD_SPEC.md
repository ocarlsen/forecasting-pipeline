# forecasting-pipeline: build spec (v1)

Owen Carlsen's entry to ForecastBench (forecastbench.org, Forecasting Research Institute). Named entry.
Three forecast files per round: **raw**, **control**, **platt**. FRI scores all three, so the leaderboard
itself shows what each step was worth.

Hard date: question set **2026-11-08** (released 00:00 UTC Nov 8, due 23:59:59 UTC Nov 8).

## Verified facts (2026-10-27, from the wiki + datasets repo; re-check before relying)

- Question sets: `https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main/datasets/question_sets/<D>-llm.json`
  (43 sets, biweekly Sundays since 2024-07-21; `latest-llm.json` too). Top keys: `forecast_due_date`,
  `question_set`, `questions` (500).
- Resolution sets: `.../datasets/resolution_sets/<D>_resolution_set.json`, keys `forecast_due_date`,
  `question_set`, `resolutions`: rows `{id, source, direction, resolution_date, resolved_to, resolved}`.
  Market questions are often `resolved: false`; only use `resolved: true` rows.
- Sets before 2025-10-26 contain combination questions (`id` is a list): DROP them.
- Market sources: kalshi, manifold, metaculus, polymarket (also historic `infer`). Dataset sources: acled,
  dbnomics, fred, wikipedia, yfinance, serpapi (new). Treat source lists as data: a question is a dataset
  question iff `resolution_dates` is a list; market iff it is the string "N/A". Don't hard-code source names
  for that split.
- Market question: 1 forecast, `resolution_date: null`. Dataset question: 1 forecast per date in
  `resolution_dates` (currently 5: +7/30/90/180/365 d; older sets up to 8).
- Dataset `question` is an f-string template with `{forecast_due_date}` and `{resolution_date}`.
- `freeze_datetime_value` (string, may be "N/A") = market price for market questions, last observed value
  for dataset questions; frozen 10-16 days before the due date (`freeze_datetime`). Explanation in
  `freeze_datetime_value_explanation`.
- Forecast file: name `<due_date>.<organization>.<N>.json`; JSON keys exactly `organization`, `model`,
  `model_organization`, `question_set`, `forecasts` (rows `{id, source, forecast, resolution_date, reasoning}`).
  `forecast` float in [0,1]. Need >=95% of market AND >=95% of dataset questions covered; missing = 0.5.
  Max 3 files per round. `model` cannot be changed after posting.
- Leaderboard reference rows (tournament): Naive Forecaster market 71.2 / dataset 48.1; superforecaster
  median 68.5 overall; top 69.7. Bare Claude models ~59-61 (baseline board).

## Shape

Python >=3.11 package `fbe`, venv `.venv` on python3.12 (system python3 is 3.9). Deps: `anthropic`, `pytest`
(dev). No pandas. CLI entry point `fbe`. MIT license, author Owen Carlsen.

Modules:
- `fbe/data.py` - fetch + cache (under `data/cache/`, gitignored) question and resolution sets; list
  available sets via the GitHub contents API or by date arithmetic; normalize questions (drop combos,
  classify market/dataset, render the dataset template for each resolution date).
- `fbe/models.py` - model registry: id, prices ($/MTok in/out), training cutoff date **with the URL it was
  read from** (fetch from Anthropic's models overview docs; if a cutoff can't be sourced, the model is
  refused for backtesting, loudly). Candidates: `claude-haiku-4-5`, `claude-sonnet-5`, `claude-opus-5-5`.
  Prices: Haiku 4.5 $1/$5, Sonnet 5 $2/$10, Opus 5.5 $4/$20; Batch API is 50% off.
- `fbe/forecast.py` - prompt + Anthropic call. One call per question; dataset questions return all horizons
  in one call. Use structured outputs (`output_config: {format: ...}`) for a JSON schema of probabilities.
  Read the claude-api reference before writing SDK code (don't guess). Per-model API rules learned from
  earlier live runs: no `temperature` on Sonnet 5 / Opus 5.5 (400); Opus 5.5 thinking cannot be
  disabled (use effort `low`); Sonnet 5 thinking adaptive or disabled, set effort explicitly; Haiku 4.5 no
  thinking; size `max_tokens` so thinking can't starve the answer, and treat a truncated/unparseable reply
  as a recorded failure (impute 0.5, count it), never silently. Prompt gives the question, background,
  resolution criteria, the freeze value + its explanation + its date, and the due date. No web tools (a
  backtest with live search would leak outcomes).
- `fbe/batch.py` - Message Batches API path (default) with a synchronous fallback; results keyed by
  `custom_id`, never by position.
- `fbe/cost.py` - pre-run CEILING estimate (max_tokens-based) and a hard `--max-usd` cap that refuses to
  start if the ceiling exceeds it; record actual usage to the run dir.
- `fbe/calibrate.py` - Platt scaling: fit a, b on logit(raw p) -> outcome by logistic regression.
  Fit separately for market and dataset questions. Clip p to [0.01, 0.99] before logit. If too few
  training rows (< 50) for a type, Platt = identity for that type and say so in the report.
- `fbe/control.py` - the non-learning control (`CONTROL_SHRINK`): `p_control = 0.8 * p_raw + 0.2 *
  anchor`, where anchor = market freeze value for market questions (fallback 0.5 if N/A), and for dataset questions the
  past-only base rate of `resolved_to` for that (source, horizon-bucket) from resolution rows resolved before
  the target due date (fallback 0.5).
- `fbe/score.py` - Brier per question type and overall (overall = mean of market Brier and dataset Brier,
  equal weight), plus reference baselines on the same rows: always-0.5, market-copier (freeze value), and
  the control. Don't invent or approximate FRI's Brier Index: report plain Brier and the difference vs
  always-0.5 and vs copier, and note in the README that FRI's difficulty-adjusted index differs.
- `fbe/writer.py` - write + validate the three forecast files (`.1` raw, `.2` control, `.3` platt; `model`
  fields like `"<model-id> (raw)"`, `"<model-id> (control)"`, `"<model-id> (platt)"`), schema-checked,
  coverage-checked (refuse < 95% per type), probabilities finite and in [0,1].
- `fbe/cli.py`:
  - `fbe sets` - list sets with resolution counts.
  - `fbe plan --model M --sets ... --sample N` - $0, prints the ceiling cost.
  - `fbe backtest --models ... --sets ... --sample N --max-usd X` - forecasts past sets, scores raw/control/platt
    with LEAK-FREE fitting: for target set D, calibration rows are only resolutions with
    `resolution_date < D` from earlier sets (prequential); and a model is only allowed on sets with
    `forecast_due_date > model cutoff`. Sampling is deterministic (seeded), stratified by type.
    Writes `runs/<timestamp>/` (gitignored) with raw responses, usage, and a `report.md`.
  - `fbe pick` - blind model pick: scores shown under labels A/B/C; mapping saved to a file and revealed
    only after the rule (lowest overall raw Brier, ties to cheaper) has chosen. Print the rule.
  - `fbe live --model M --date D --org NAME --max-usd X` - fetch set D, forecast, fit Platt on everything
    resolved before D, write the three files to `out/`, validate. Must work unattended in one command.
  - `fbe validate FILE` - standalone validator.
  - `--dry-run` everywhere metered: a stub model (deterministic fake probabilities) so the whole path runs at $0.
- API key: read only from the `ANTHROPIC_API_KEY` env var. Never print it, never write it, never read .env files.

## Tests (pytest, all offline, $0)

Fixtures = small slices of real sets saved under `tests/fixtures/`. Must cover: combo drop; market/dataset
split; template render; writer schema + filename + coverage refusal; Platt recovers known a,b on synthetic
data and is identity below 50 rows; leak-free fitting (a row with resolution_date >= D must never enter the
fit: prove it with a test that fails if the filter is removed); cutoff refusal; cost-cap refusal; batch
results matched by custom_id when out of order; truncated reply -> 0.5 + counted; end-to-end `--dry-run` of
backtest and live producing three valid files.

## Done means

`pytest` green; `fbe backtest --dry-run` and `fbe live --dry-run --date 2026-10-25` run end to end and the
three files pass `fbe validate`; `fbe plan` prints ceilings for the three candidates on the clean sets with
`--sample 150`; README explains the method, the three arms, the leak-free rule and how to run live. Report
the plan ceilings and anything unverified.
