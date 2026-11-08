# Design notes

## Layout

- `fbe/data.py`: fetch, cache and normalize question and resolution sets (combo drop,
  market/dataset split, dataset template rendering).
- `fbe/models.py`: model registry: price, sourced cutoff, per-model API quirks.
- `fbe/forecast.py`: prompt, structured-output schema and response parsing. One call per
  question; a dataset question's call covers every one of its resolution dates at once.
- `fbe/batch.py`: Message Batches API path (default) with a synchronous fallback.
- `fbe/cost.py`: pre-run ceiling estimate and the `--max-usd` hard cap.
- `fbe/calibrate.py`: Platt scaling.
- `fbe/control.py`: the non-learning control blend.
- `fbe/score.py`: Brier scoring and baselines.
- `fbe/writer.py`: builds, schema-checks, coverage-checks and writes the three forecast files.
- `fbe/pipeline.py`: sampling, the leak-free training pool, and running one round.
- `fbe/cli.py`: the `fbe` command.

## Choices

- **Coverage is measured per question, not per row.** A dataset question only counts as covered
  once every one of its resolution-date rows is present. The 95% threshold is read as 95% of
  questions, and the writer enforces that literally.
- **Platt's training pool is this model's own prior rounds, filtered by `model_id`.** Never
  another organization's or another candidate's submissions. `fbe backtest` and `fbe live` read
  and append to the same `runs/history.jsonl`, so calibration compounds across invocations;
  `--dry-run` writes to `runs/history.dryrun.jsonl` instead.
- **Batch `custom_id` is synthetic.** The Batches API requires `custom_id` to match
  `^[a-zA-Z0-9_-]{1,64}$`, and real question ids contain dots and run past 64 characters. Every
  request gets a `q00000`-style id, the id-to-question map is written to the run dir, and results
  are matched back through that map, never by position.
- **A "succeeded" batch result can still be a failure.** A result whose `stop_reason` is
  `max_tokens` or `refusal` has no usable JSON and is recorded as a failure, not scored.
- **Generous `max_tokens`.** Thinking models can spend thousands of tokens before emitting the
  structured answer; a small `max_tokens` starves the answer entirely. The floors are generous
  and the cost ceiling rises accordingly, which is the honest tradeoff.
- **The output schema carries no numeric constraints.** The structured-outputs validator rejects
  `minimum`/`maximum` and `minItems`/`maxItems` outside {0,1}. Range and exact-date-set checks
  happen client-side in `parse_response_text()` instead.
- **Common sets.** `fbe pick`'s common sets and `fbe backtest`/`fbe plan`'s default `--sets` both
  scope to sets every requested model is usable on; `pick` additionally requires a recorded score
  for every candidate before a set counts.
- **`--resume-batch` is wired on `live`, not `backtest`.** A live round is one submission-scale
  batch where losing the id matters. A backtest spans many smaller per-round batches, where
  re-running the round is simpler than resuming one. A scope choice, not an oversight.
- **Dataset horizons are bucketed to the nearest of {7, 30, 90, 180, 365} days** for the control
  arm's base-rate anchor. Tracked per exact day offset, most buckets would never hold enough rows
  to mean anything.
- **`market-copier` is scored only on market rows.** A dataset question's freeze value is a last
  observed value, not a probability, so there is no principled copier for that type.
- **The API key is read only from `ANTHROPIC_API_KEY`.** Never printed, never written, never read
  from a `.env` file.
- The code was built and tested offline (`--dry-run`, mocked fetches). The first metered runs came
  after that: a 4-question smoke test ($0.04) and the full backtest (8 sets x 3 models x 150
  questions, $12.92 actual against a $222.61 ceiling).
