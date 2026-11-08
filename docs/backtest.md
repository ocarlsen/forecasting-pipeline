# Backtest

Run `runs/20261028T003636Z` (run dirs are gitignored): three candidate models on the eight
published sets dated 2026-07-05 through 2026-10-11, 150 sampled questions per set (seed 42),
every set after every candidate's training cutoff. Actual spend $12.92 against a $222.61
ceiling. Numbers below are n-weighted means over that run's `scores.json`.

## The blind pick

`fbe pick` shuffled the three models under labels A, B, C, applied the rule (lowest overall raw
Brier on the common sets, ties to the cheaper model), and only then revealed the mapping.

| label | model | overall raw Brier | n |
|---|---|---|---|
| A | `claude-opus-5-5` | 0.137 | 1335 |
| C | `claude-sonnet-5` | 0.160 | 1335 |
| B | `claude-haiku-4-5` | 0.173 | 1335 |

"Overall" here is each set's mean of market Brier and dataset Brier, weighted across sets by that
set's scored rows. Picked: A, which was Opus 5.5.

## Opus 5.5 by question type

| rows | raw | control | platt | copy the market | always 0.5 |
|---|---|---|---|---|---|
| market, n=196 | 0.113 | 0.114 | 0.115 | 0.117 | 0.250 |
| dataset, n=1139 | 0.158 | 0.161 | 0.158 | n/a | 0.250 |

Copy-the-market has no dataset column: a dataset question's freeze value is a last observed
value, not a probability.

The same table for the other two candidates:

| model | rows | raw | control | platt |
|---|---|---|---|---|
| `claude-sonnet-5` | market, n=196 | 0.150 | 0.139 | 0.151 |
| `claude-sonnet-5` | dataset, n=1139 | 0.166 | 0.168 | 0.167 |
| `claude-haiku-4-5` | market, n=196 | 0.159 | 0.145 | 0.159 |
| `claude-haiku-4-5` | dataset, n=1139 | 0.185 | 0.184 | 0.182 |

## What the differences are worth

`scripts/bootstrap.py` rebuilds every scored row from `runs/history.jsonl`, checks that the rows
reproduce `scores.json` exactly, then resamples whole questions (a dataset question's horizons
move together, 4,000 draws). Difference in Brier, negative means the first arm is better:

| rows | comparison | difference | 95% interval |
|---|---|---|---|
| market, n=196 | raw minus copy the market | -0.0035 | -0.0074 to +0.0003 |
| market, n=196 | platt minus raw | +0.0017 | -0.0020 to +0.0057 |
| market, n=196 | control minus raw | +0.0005 | -0.0003 to +0.0012 |
| dataset, n=1139 | platt minus raw | +0.0005 | -0.0011 to +0.0021 |
| dataset, n=1139 | control minus raw | +0.0028 | +0.0008 to +0.0047 |

So calibration was worth about zero, because Opus 5.5 was already calibrated on these rows, and
on market questions the raw arm ties copying the price. The control is measurably worse on
dataset questions: the public base rates it blends toward did not hold in these rounds.

The blind pick holds on the same rows: Opus 5.5 beats Sonnet 5 by 0.037 (market) and 0.008
(dataset), and Haiku 4.5 by 0.046 and 0.027, every interval excluding zero.

The control arm helped the two weaker models on market questions, because it pulls toward the
price. A gain the fixed blend also gets is not a calibration gain, which is what the control is
there to show.

Reproduce: `.venv/bin/python scripts/bootstrap.py --run runs/20261028T003636Z` (offline, $0).

## Caveats

- The market questions that have resolved are the early resolvers. The 2026-10-11 set had no
  resolved market rows at all, so its market columns are empty and the 196 market rows come from
  the other seven sets.
- The 180-day and 365-day dataset horizons have not resolved yet, so they are not in these
  scores.
- This is plain Brier, not FRI's difficulty-adjusted Brier Index. See [Method](method.md).
- 150 questions per set is a sample, not the full 500.

## Reproducing

```bash
fbe backtest --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5-5 --sample 150 --max-usd 20
fbe pick --run runs/<timestamp>
```

The run writes `report.md`, `scores.json`, `usage.json` and `raw_responses.json` to
`runs/<timestamp>/`. A second run will not reproduce these numbers exactly: the models are not
deterministic, and more rows will have resolved.
