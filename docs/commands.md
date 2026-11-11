# Commands

Every subcommand and flag of the `fbe` command, as defined in `fbe/cli.py`.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...   # only read from the env; never from a .env file
```

## The commands

```bash
fbe sets                                                    # list published sets + resolution counts
fbe plan --model claude-haiku-4-5,claude-sonnet-5,claude-opus-5-5 \
         --sample 150                                       # $0 ceiling estimate; --sets optional
fbe backtest --models claude-haiku-4-5,claude-sonnet-5,claude-opus-5-5 \
             --sets 2026-09-27,2026-10-11 --sample 50 --max-usd 5 --dry-run
fbe pick --run runs/20261028T002210Z                         # reads that backtest's scores.json
fbe live --model claude-haiku-4-5 --date 2026-11-08 \
         --org ocarlsen --max-usd 60                         # writes out/<date>.<org>.{1,2,3}.json
fbe validate out/2026-11-08.ocarlsen.1.json
```

| command | required | optional |
|---|---|---|
| `fbe sets` | | `--refresh` re-fetches the set list and resolution sets instead of using the cache |
| `fbe plan` | `--model` (comma-separated ids) | `--sets`, `--sample N`, `--seed` (default 42), `--no-batch` |
| `fbe backtest` | `--models` (comma-separated ids), `--max-usd` | `--sets`, `--sample N`, `--seed` (default 42), `--no-batch`, `--dry-run`, `--allow-failures` |
| `fbe pick` | `--run` (a backtest run dir) | `--seed` (default 7; shuffles the A/B/C labels) |
| `fbe live` | `--model`, `--date`, `--org`, `--max-usd` | `--seed` (default 42), `--no-batch`, `--dry-run`, `--allow-failures`, `--resume-batch ID`, `--from-raw RUN_DIR` |
| `fbe validate` | `FILE` | |

`plan` takes `--model`; `backtest` takes `--models`. Both accept a comma-separated list.

## Cost cap

`--max-usd` is required on `backtest` and `live` (no default, even under `--dry-run`), so there is
no accidental unbounded metered run.

On `backtest` the cap is cumulative on actual spend: before each (model, set) round the run refuses
if spent-so-far plus that round's ceiling would cross the cap, and stops there. The full-run
ceiling is only a warning, because thinking-model ceilings run far above actual spend.

On `live` the cap applies to the ceiling of the one round, so it refuses to start if the ceiling
exceeds `--max-usd`. A 500-question Opus 5.5 round ceilings at about $51, so pass `--max-usd` of at
least that.

## Which sets

`--sets` is optional on `plan` and `backtest`. When omitted, it defaults to the intersection of
(every requested model's cutoff allows that set) and (that set has at least one resolved row).
When you pass `--sets` and some requested model cannot use one of them because its cutoff forbids
it, the command refuses with a named model and set rather than skipping it. Both cases print
`comparing on N common sets: [...]`.

`pick` compares on the sets where every model in the run has a recorded score; a set one model
skipped or refused does not count.

## Batch, sync, dry run

Every metered path defaults to the Message Batches API (50% off); `--no-batch` forces the
synchronous fallback. `--dry-run` replaces the Anthropic call with a deterministic stub model
everywhere, so the entire pipeline, including the cost-cap check, runs for $0.

A batch's id is written to the run dir immediately after creation, before polling starts, so a
crash mid-poll never loses the only pointer to an already-billed batch. `--resume-batch ID` on
`live` resumes polling and fetching an existing batch instead of creating a new one.

## Mass-failure refusal

If more than 2% of a round's calls fail (truncated or unparseable reply, a `stop_reason` of
`max_tokens` or `refusal`, or a missing batch result), `fbe live` refuses to write the three
submission files at all, and `fbe backtest` refuses to score that one (model, set) round. Without
this, a double-digit share of "forecasts" could be the 0.5 imputation wearing a validated schema.
Every failure's id, reason and `stop_reason` (when known) are written to `failures.json` in the
run dir. `--allow-failures` overrides the refusal when you have already seen why and want the
imputed file anyway.

## History files

`fbe backtest` and `fbe live` (both without `--dry-run`) record every round's raw forecasts, tagged
with `model_id`, in `runs/history.jsonl`. Rows are keyed by model, set, source, question and
horizon, so re-running a round replaces its earlier rows instead of duplicating them. `--dry-run`
writes to `runs/history.dryrun.jsonl` the same way, so a rehearsal never pollutes the real calibration history. `fbe live` prints
`Platt fit on N market / M dataset rows` plus either the fitted (a, b) or the identity reason.

## Running live for real

1. `fbe plan ...` first. It is free and prints the ceiling before anything else runs.
2. `export ANTHROPIC_API_KEY=...` (never committed, never read from a file).
3. `fbe live --model <id> --date <due-date> --org <name> --max-usd <cap>`. It refuses to start if
   the ceiling exceeds `--max-usd`, refuses to write if more than 2% of calls failed, otherwise
   writes and validates the three files under `out/` and appends the round's raw forecasts to
   `runs/history.jsonl` for the next round's Platt fit.
4. Submit the three files. `model` cannot change after posting, so pick it first with
   `fbe backtest` (all candidates, same sets) and then `fbe pick --run <that run dir>`. `pick`
   only reads recorded scores; it never calls a model.

## Output files

`fbe live` writes `out/<date>.<org>.1.json` (raw), `.2.json` (control) and `.3.json` (platt). Each
file's `model` field is `"<model-id> (raw|control|platt)"`. `fbe validate` checks the schema and,
when it can fetch the referenced question set, that at least 95% of market and of dataset
questions are covered.
