"""Message Batches API path (default, 50% off), with a synchronous fallback.

The Batches API requires `custom_id` to match
`^[a-zA-Z0-9_-]{1,64}$`. Real ForecastBench question ids are NOT safe to use directly -- a
spot check across a real question set found ids containing dots (`polymarket`/`dbnomics`-style)
and ids over 64 characters (`kalshi` ids), both of which the pattern rejects. `custom_id` is
therefore always a synthetic `f"q{i:05d}"`, with an explicit `{custom_id: qid}` map built from
the same (deterministic) spec order and persisted to the run dir for audit. Results are then
matched back to the real qid via that map, keyed by `custom_id` -- NEVER by position.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from fbe.forecast import CallSpec, build_request_params, stub_response_text
from fbe.models import ModelSpec

# stop_reason values on an otherwise "succeeded" result that mean the reply is NOT usable --
# the model either ran out of room (thinking starved the JSON answer) or refused outright.
FAILURE_STOP_REASONS = {"max_tokens", "refusal"}


@dataclass
class BatchOutcome:
    texts: dict[str, str] = field(default_factory=dict)  # qid -> raw reply text
    errors: dict[str, str] = field(default_factory=dict)  # qid -> error description
    stop_reasons: dict[str, str | None] = field(default_factory=dict)  # qid -> stop_reason seen
    usage: dict[str, dict[str, int]] = field(
        default_factory=dict
    )  # qid -> {"input_tokens":.., "output_tokens":..}


def build_custom_id_map(specs: list[CallSpec]) -> dict[str, str]:
    """Deterministic custom_id -> qid map, built purely from spec order (so it can always be
    rebuilt identically from the same question set, e.g. to resume a batch in a new process)."""
    return {f"q{i:05d}": spec.qid for i, spec in enumerate(specs)}


def extract_text(message) -> str | None:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def _usage_dict(message) -> dict[str, int]:
    usage = getattr(message, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def results_to_outcome(
    results, expected_ids: set[str], custom_id_to_qid: dict[str, str]
) -> BatchOutcome:
    """Build a BatchOutcome from a batch `results()` iterable, keyed by the REAL qid
    (recovered via `custom_id_to_qid`), regardless of the order results arrive in.

    A "succeeded" result whose message.stop_reason is max_tokens or refusal is recorded as a
    failure even though the Batches API itself calls it "succeeded" -- the JSON
    answer in that case is truncated/absent and must never be scored as if it were real.
    """
    outcome = BatchOutcome()
    seen: set[str] = set()
    for r in results:
        cid = r.custom_id
        qid = custom_id_to_qid.get(cid, cid)  # fall back to cid itself if map is somehow stale
        seen.add(qid)
        rtype = r.result.type
        if rtype == "succeeded":
            message = r.result.message
            stop_reason = getattr(message, "stop_reason", None)
            outcome.stop_reasons[qid] = stop_reason
            outcome.usage[qid] = _usage_dict(message)
            if stop_reason in FAILURE_STOP_REASONS:
                outcome.errors[qid] = f"stop_reason={stop_reason}"
                continue
            text = extract_text(message)
            if text is None:
                outcome.errors[qid] = "succeeded but no text block"
            else:
                outcome.texts[qid] = text
        else:
            outcome.stop_reasons[qid] = None
            outcome.errors[qid] = f"batch result type={rtype}"
    for missing in expected_ids - seen:
        outcome.stop_reasons[missing] = None
        outcome.errors[missing] = "no batch result returned"
    return outcome


def run_stub(specs: list[CallSpec], seed: int = 0) -> BatchOutcome:
    """--dry-run path: deterministic fake text, no network, no client required."""
    texts = {spec.qid: stub_response_text(spec, seed) for spec in specs}
    return BatchOutcome(texts=texts)


def create_batch_requests(specs: list[CallSpec], model: ModelSpec):
    """Returns (requests, custom_id_to_qid). `custom_id` is always the synthetic `q00000`
    form -- never the raw question id (see module docstring)."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    id_map = build_custom_id_map(specs)
    spec_by_qid = {s.qid: s for s in specs}
    reqs = []
    for custom_id, qid in id_map.items():
        spec = spec_by_qid[qid]
        params = build_request_params(spec, model)
        reqs.append(Request(custom_id=custom_id, params=MessageCreateParamsNonStreaming(**params)))
    return reqs, id_map


def _persist_batch_metadata(run_dir: Path | None, batch_id: str, id_map: dict[str, str]) -> None:
    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "batch_id.txt").write_text(batch_id)
    (run_dir / "batch_id_map.json").write_text(json.dumps(id_map, indent=2))


def run_batch(
    client,
    specs: list[CallSpec],
    model: ModelSpec,
    poll_interval_s: float = 15.0,
    max_wait_s: float = 24 * 3600,
    run_dir: Path | None = None,
) -> BatchOutcome:
    reqs, id_map = create_batch_requests(specs, model)
    batch = client.messages.batches.create(requests=reqs)
    # Persist the batch id BEFORE polling -- a crash/kill mid-poll must never lose the only
    # record of a batch that's already been billed and is running on Anthropic's servers.
    _persist_batch_metadata(run_dir, batch.id, id_map)
    print(f"[batch] created {batch.id} ({len(reqs)} requests) -- id persisted to {run_dir}")
    waited = 0.0
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        if waited > max_wait_s:
            raise TimeoutError(
                f"batch {batch.id} did not end within {max_wait_s}s -- resume later with "
                f"--resume-batch {batch.id}"
            )
    results = client.messages.batches.results(batch.id)
    return results_to_outcome(results, {s.qid for s in specs}, id_map)


def run_batch_resume(
    client,
    batch_id: str,
    specs: list[CallSpec],
    poll_interval_s: float = 15.0,
    max_wait_s: float = 24 * 3600,
    run_dir: Path | None = None,
) -> BatchOutcome:
    """Resume polling/fetching an already-created batch (`--resume-batch ID`). `specs` must be
    rebuilt from the SAME question set/order as the original run -- custom_id assignment is
    purely positional (q00000, q00001, ...), so an identical rebuild reproduces the same map."""
    id_map = build_custom_id_map(specs)
    _persist_batch_metadata(run_dir, batch_id, id_map)
    waited = 0.0
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        time.sleep(poll_interval_s)
        waited += poll_interval_s
        if waited > max_wait_s:
            raise TimeoutError(f"batch {batch_id} did not end within {max_wait_s}s")
    results = client.messages.batches.results(batch_id)
    return results_to_outcome(results, {s.qid for s in specs}, id_map)


def run_sync(client, specs: list[CallSpec], model: ModelSpec) -> BatchOutcome:
    """Synchronous fallback -- one messages.create() per question, no batch discount."""
    outcome = BatchOutcome()
    for spec in specs:
        params = build_request_params(spec, model)
        try:
            resp = client.messages.create(**params)
        except Exception as e:  # noqa: BLE001 -- recorded as a per-question failure, not raised
            outcome.errors[spec.qid] = f"{type(e).__name__}: {e}"
            continue
        stop_reason = getattr(resp, "stop_reason", None)
        outcome.stop_reasons[spec.qid] = stop_reason
        outcome.usage[spec.qid] = _usage_dict(resp)
        if stop_reason in FAILURE_STOP_REASONS:
            outcome.errors[spec.qid] = f"stop_reason={stop_reason}"
            continue
        text = extract_text(resp)
        if text is None:
            outcome.errors[spec.qid] = "no text block in response"
        else:
            outcome.texts[spec.qid] = text
    return outcome
