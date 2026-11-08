import json

import pytest

from fbe.data import normalize_question_set
from fbe.forecast import (
    ParseError,
    build_call_specs,
    parse_response_text,
    stub_response_text,
)
from fbe.pipeline import forecast_specs


def test_stub_response_parses_cleanly(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    for spec in specs:
        text = stub_response_text(spec, seed=1)
        parsed = parse_response_text(spec, text)
        for h in spec.horizons:
            p, _reasoning = parsed[h.resolution_date]
            assert 0.0 <= p <= 1.0


def test_truncated_reply_raises_parse_error(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    market_spec = next(s for s in specs if s.kind == "market")
    truncated = '{"probability": 0.7, "reasoning": "cut off mid-sent'  # not valid JSON
    with pytest.raises(ParseError):
        parse_response_text(market_spec, truncated)


def test_dataset_reply_missing_a_horizon_raises_parse_error(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    ds_spec = next(s for s in specs if s.kind == "dataset" and len(s.horizons) == 2)
    only_one = json.dumps(
        {"forecasts": [{"resolution_date": ds_spec.horizons[0].resolution_date, "probability": 0.4, "reasoning": "x"}]}
    )
    with pytest.raises(ParseError):
        parse_response_text(ds_spec, only_one)


def test_forecast_specs_imputes_half_and_counts_failures(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    market_spec = next(s for s in specs if s.kind == "market")
    good_spec = next(s for s in specs if s.qid != market_spec.qid)

    texts = {good_spec.qid: stub_response_text(good_spec, seed=2)}
    texts[market_spec.qid] = "{not valid json"  # truncated/garbled reply
    errors = {}

    raw_by_qid, n_failures, failure_reasons = forecast_specs(specs, texts, errors)

    # the truncated one is imputed to 0.5 for every one of its horizons, and counted
    assert n_failures >= 1
    for h in market_spec.horizons:
        assert raw_by_qid[market_spec.qid][h.resolution_date] == 0.5
    assert market_spec.qid in failure_reasons
    assert "parse_error" in failure_reasons[market_spec.qid]

    # any spec never given a text at all is also imputed + counted (covers batch 'missing')
    missing_specs = [s for s in specs if s.qid not in texts]
    assert len(missing_specs) >= 1
    for s in missing_specs:
        for h in s.horizons:
            assert raw_by_qid[s.qid][h.resolution_date] == 0.5
        assert s.qid in failure_reasons
