import json
from types import SimpleNamespace

from fbe.batch import (
    build_custom_id_map,
    create_batch_requests,
    extract_text,
    results_to_outcome,
)
from fbe.data import normalize_question_set
from fbe.forecast import build_call_specs
from fbe.models import get_model


def _succeeded(custom_id, text, stop_reason="end_turn", input_tokens=10, output_tokens=20):
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="succeeded", message=message))


def _errored(custom_id):
    return SimpleNamespace(custom_id=custom_id, result=SimpleNamespace(type="errored"))


def test_extract_text_finds_first_text_block():
    message = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="..."), SimpleNamespace(type="text", text="hi")]
    )
    assert extract_text(message) == "hi"


def test_results_matched_by_custom_id_out_of_order():
    id_map = {"q00000": "q1", "q00001": "q2", "q00002": "q3"}
    # Results arrive in an order that does NOT match request order -- q3, q1, q2.
    results = [
        _succeeded("q00002", "text-3"),
        _succeeded("q00000", "text-1"),
        _succeeded("q00001", "text-2"),
    ]
    outcome = results_to_outcome(results, expected_ids={"q1", "q2", "q3"}, custom_id_to_qid=id_map)
    assert outcome.texts == {"q1": "text-1", "q2": "text-2", "q3": "text-3"}
    assert outcome.errors == {}
    assert outcome.usage["q1"] == {"input_tokens": 10, "output_tokens": 20}


def test_results_missing_and_errored_are_recorded_by_id():
    id_map = {"q00000": "q1", "q00001": "q2", "q00002": "q3"}
    results = [_succeeded("q00000", "text-1"), _errored("q00001")]
    outcome = results_to_outcome(results, expected_ids={"q1", "q2", "q3"}, custom_id_to_qid=id_map)
    assert outcome.texts == {"q1": "text-1"}
    assert outcome.errors["q2"] == "batch result type=errored"
    assert outcome.errors["q3"] == "no batch result returned"


def test_max_tokens_stop_reason_is_a_failure_despite_succeeded_type():
    id_map = {"q00000": "q1"}
    results = [_succeeded("q00000", '{"probability": 0.4', stop_reason="max_tokens")]
    outcome = results_to_outcome(results, expected_ids={"q1"}, custom_id_to_qid=id_map)
    assert "q1" not in outcome.texts
    assert outcome.errors["q1"] == "stop_reason=max_tokens"
    assert outcome.stop_reasons["q1"] == "max_tokens"


def test_refusal_stop_reason_is_a_failure():
    id_map = {"q00000": "q1"}
    results = [_succeeded("q00000", "", stop_reason="refusal")]
    outcome = results_to_outcome(results, expected_ids={"q1"}, custom_id_to_qid=id_map)
    assert outcome.errors["q1"] == "stop_reason=refusal"


# --- real ids with dots / >64 chars must still produce a valid custom_id ---

REAL_IDS_THAT_BREAK_THE_BATCH_PATTERN = [
    "NQ26.1Q.DBN",  # dbnomics-style: contains dots
    "KXUSAGDPYOY-26.1Q-T0.5",  # kalshi-style: dots AND over 64 chars once source-qualified
    "a" * 70,  # a real polymarket id observed over 64 chars
    "poly.market/question-with.dots.in.it.and.more.than.sixty.four.characters.total",
]


def test_custom_ids_always_match_the_batch_pattern(question_set):
    import re

    pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    # graft in ids that are known to break the pattern, as a real batch run would see them
    broken_specs = []
    for i, spec in enumerate(specs):
        bad_id = REAL_IDS_THAT_BREAK_THE_BATCH_PATTERN[i % len(REAL_IDS_THAT_BREAK_THE_BATCH_PATTERN)]
        broken_specs.append(spec.__class__(**{**spec.__dict__, "qid": bad_id}))

    id_map = build_custom_id_map(broken_specs)
    assert len(id_map) == len(broken_specs)
    for custom_id in id_map:
        assert pattern.match(custom_id), f"{custom_id!r} violates the Batches API pattern"
    # and the map still recovers the real (pattern-violating) qid
    assert set(id_map.values()) == {s.qid for s in broken_specs}


def test_custom_id_round_trips_out_of_order_for_real_ids(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    broken_specs = [
        spec.__class__(**{**spec.__dict__, "qid": REAL_IDS_THAT_BREAK_THE_BATCH_PATTERN[i % 4]})
        for i, spec in enumerate(specs)
    ]
    id_map = build_custom_id_map(broken_specs)
    # simulate results arriving in reverse order
    results = [
        _succeeded(custom_id, json.dumps({"ok": True}))
        for custom_id in reversed(list(id_map))
    ]
    outcome = results_to_outcome(
        results, expected_ids={s.qid for s in broken_specs}, custom_id_to_qid=id_map
    )
    for spec in broken_specs:
        assert spec.qid in outcome.texts


def test_create_batch_requests_uses_valid_custom_ids_for_real_ids(question_set):
    import re

    pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    broken_specs = [
        spec.__class__(**{**spec.__dict__, "qid": REAL_IDS_THAT_BREAK_THE_BATCH_PATTERN[i % 4]})
        for i, spec in enumerate(specs)
    ]
    reqs, id_map = create_batch_requests(broken_specs, get_model("claude-haiku-4-5"))
    assert len(reqs) == len(broken_specs)
    for req in reqs:
        assert pattern.match(req["custom_id"])
    assert set(id_map.values()) == {s.qid for s in broken_specs}
