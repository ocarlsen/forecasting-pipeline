"""The structured-outputs schema validator rejects numeric constraints and
minItems/maxItems outside {0,1} with a 400. The raw output_config.format.schema dict is sent
to the API as-is -- nothing in the SDK strips these keywords -- so they must never appear.
"""
from fbe.data import normalize_question_set
from fbe.forecast import build_call_specs, output_schema

FORBIDDEN_KEYS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
}


def _walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _assert_schema_clean(schema: dict):
    for key, value in _walk(schema):
        assert key not in FORBIDDEN_KEYS, f"forbidden keyword {key!r} found with value {value!r}"
        if key == "minItems":
            assert value in (0, 1), f"minItems={value} not in {{0,1}}"
        if key == "maxItems":
            raise AssertionError(f"maxItems must not appear at all, found {value!r}")


def test_market_schema_has_no_forbidden_keywords(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    market_spec = next(s for s in specs if s.kind == "market")
    _assert_schema_clean(output_schema(market_spec))


def test_dataset_schema_has_no_forbidden_keywords(question_set):
    rows, _ = normalize_question_set(question_set)
    specs = build_call_specs(rows)
    dataset_spec = next(s for s in specs if s.kind == "dataset" and len(s.horizons) > 1)
    schema = output_schema(dataset_spec)
    _assert_schema_clean(schema)
    # range/exact-count enforcement moved entirely to parse_response_text(); the schema still
    # pins the exact set of valid dates via `enum`, which is not a numeric/length constraint.
    forecasts_items = schema["properties"]["forecasts"]["items"]
    assert "enum" in forecasts_items["properties"]["resolution_date"]
    assert "minItems" not in schema["properties"]["forecasts"]
    assert "maxItems" not in schema["properties"]["forecasts"]
