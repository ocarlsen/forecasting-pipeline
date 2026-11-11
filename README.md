# forecasting-pipeline

A leak-free LLM forecasting pipeline, built by Owen Carlsen and first entered in
[ForecastBench](https://www.forecastbench.org/) (Forecasting Research Institute). Every round
submits three forecast files from the same model, `raw`, `control` and `platt`, so the
leaderboard itself shows what each step was worth. The model is picked blind, has no web access,
and is calibrated only on outcomes whose `resolution_date` is before the round's due date `D`.

## How it works

| step | command | what it does |
|---|---|---|
| 1 | `fbe sets` | Lists the published question sets and how many rows of each have resolved. |
| 2 | `fbe plan` | Prints the cost ceiling for a run. Calls no model. |
| 3 | `fbe backtest` | Forecasts past sets with each candidate, scores all three arms leak-free, writes `runs/<ts>/`. Refuses any round that would cross `--max-usd`. |
| 4 | `fbe pick` | Reads a backtest's `scores.json`, shows the models as A/B/C, applies the rule, then reveals the mapping. Calls no model. |
| 5 | `fbe live` | Forecasts one round, fits Platt on this model's resolved history, writes and validates the three files under `out/`. |
| 6 | `fbe validate` | Checks one forecast file's schema and coverage. |

## The idea

- **Three arms, one model.** `raw` is the model's own probability from one API call per
  question. `control` is `0.8*raw + 0.2*anchor`, where the anchor is the market price for market
  questions or the public past base rate for dataset questions. `platt` is Platt scaling fit on
  the model's own past forecasts. If `platt` beats `raw`, the control shows whether a fixed
  blend would have done as well.
- **Picked blind.** The three candidates ran on eight past rounds, every one dated after every
  candidate's training cutoff. `fbe pick` chose by the lowest raw Brier under shuffled labels and
  only then revealed which model that was.
- **No leaks.** No web tools. For a round due on date `D`, the calibration fit uses only
  outcomes with `resolution_date < D`, row by row, not round by round. A model is refused for any
  round not after its sourced training cutoff.

## Quickstart

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...   # only read from the env; never from a .env file
fbe plan --model claude-haiku-4-5,claude-sonnet-5,claude-opus-5-5 --sample 150
fbe backtest --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5-5 --sample 150 --max-usd 20
fbe pick --run runs/<timestamp>
fbe live --model claude-opus-5-5 --date 2026-11-08 --org ocarlsen --max-usd 60
fbe validate out/2026-11-08.ocarlsen.1.json
```

`--dry-run` on `backtest` and `live` runs the whole pipeline against a stub model for $0.
`--max-usd` is required on both. Every flag is in [Commands](docs/commands.md).

`fbe live` saves the model's replies to `runs/<ts>-live-<date>/raw_responses.json` right after the
call, so a later failure never loses a paid round. `fbe live --from-raw runs/<ts>-live-<date>`
re-scores those saved replies without calling the model. A `--dry-run` writes its files under
`out/dryrun/` and never touches a real run's output.

## Backtest

Eight rounds, 2026-07-05 through 2026-10-11, 150 questions each, three candidates. Total spend
$12.92.

The blind pick chose Claude Opus 5.5: raw Brier 0.137 against 0.160 for Sonnet 5 and 0.173 for
Haiku 4.5, 1335 scored rows each.

Opus 5.5 by question type (lower is better):

| rows | raw | control | platt | copy the market |
|---|---|---|---|---|
| market, n=196 | 0.113 | 0.114 | 0.115 | 0.117 |
| dataset, n=1139 | 0.158 | 0.161 | 0.158 | n/a |

A paired bootstrap ([`scripts/bootstrap.py`](scripts/bootstrap.py)) puts Platt minus raw on
dataset questions at +0.0005 (95% interval -0.0011 to +0.0021) and raw minus copy-the-market on
market questions at -0.0035 (-0.0074 to +0.0003). Calibration was worth about zero because Opus
5.5 was already calibrated, and on market questions it ties copying the price. The control is
slightly but measurably worse on dataset questions.

Caveats: the resolved market questions are the early resolvers; the 180-day and 365-day
horizons have not resolved yet; this is plain Brier, not FRI's difficulty-adjusted index. The
full tables, including the other two models, are in [Backtest](docs/backtest.md).
`runs/` is gitignored, so `scripts/bootstrap.py` needs the output of a local backtest run to
reproduce the intervals. The model comparison also partly measures recency: Opus 5.5's training
cutoff (2026-06-30) is about 5 days before the first backtest set (2026-07-05), while Haiku
4.5's (2025-07-31) is about 11 months before it, so the newer model had seen far more of the
period leading up to the questions.

## Nov 8 round: pending

The first live round is the 2026-11-08 question set. FRI posts preliminary scores about 10 days
after the round and full scores about 50 days after. This section will carry the leaderboard
result when it exists.

## Documentation

- [Commands](docs/commands.md): every subcommand and flag, the cost cap, batch and dry-run
  behavior, the mass-failure refusal, running live.
- [Method](docs/method.md): the three arms, scoring, the leak-free rule, model cutoffs.
- [Backtest](docs/backtest.md): the full result tables and caveats.
- [Design notes](docs/design.md): module layout and the implementation choices.
- [Build spec](docs/BUILD_SPEC.md): the spec the code was built to.

## Development

```bash
.venv/bin/pytest -q
```

Runs offline for $0 against fixtures under `tests/fixtures/`.

## License

MIT. See `LICENSE`.
