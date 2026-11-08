import datetime as dt

from fbe.data import find_resolution, normalize_question_set, render_template, resolution_lookup


def test_combo_dropped(question_set):
    rows, stats = normalize_question_set(question_set)
    assert stats.n_combo_dropped == 1
    assert all(not isinstance(r.id, list) for r in rows)
    assert "combo-a" not in {r.id for r in rows}


def test_market_dataset_split(question_set):
    rows, stats = normalize_question_set(question_set)
    assert stats.n_market_questions == 2
    assert stats.n_dataset_questions == 2
    market_rows = [r for r in rows if r.kind == "market"]
    dataset_rows = [r for r in rows if r.kind == "dataset"]
    assert {r.id for r in market_rows} == {"mkt-1", "mkt-2"}
    assert all(r.resolution_date is None for r in market_rows)
    # ds-1 has 2 resolution_dates -> 2 rows; ds-2 has 1 -> 1 row
    assert len([r for r in dataset_rows if r.id == "ds-1"]) == 2
    assert len([r for r in dataset_rows if r.id == "ds-2"]) == 1
    assert all(r.resolution_date is not None for r in dataset_rows)


def test_template_render(question_set):
    rows, _ = normalize_question_set(question_set)
    ds1_rows = sorted(
        (r for r in rows if r.id == "ds-1"), key=lambda r: r.resolution_date
    )
    assert ds1_rows[0].question == (
        "Will there be a fixture event in 2027-01-01 to 2027-01-08?"
    )
    assert ds1_rows[1].question == (
        "Will there be a fixture event in 2027-01-01 to 2027-02-01?"
    )
    # the raw template still contains the placeholders
    raw_template = next(
        q["question"] for q in question_set["questions"] if q.get("id") == "ds-1"
    )
    assert "{forecast_due_date}" in raw_template and "{resolution_date}" in raw_template
    assert render_template(raw_template, "2030-01-01", "2030-02-01") == (
        "Will there be a fixture event in 2030-01-01 to 2030-02-01?"
    )


def test_resolution_lookup_drops_combo_and_unresolved(resolution_set):
    lookup = resolution_lookup(resolution_set)
    assert find_resolution(lookup, "mkt-2", None) is None  # resolved: false
    mkt1 = find_resolution(lookup, "mkt-1", None)
    assert mkt1 is not None and mkt1["resolved_to"] == 1.0
    assert find_resolution(lookup, "ds-1", "2027-01-08") is not None
    assert find_resolution(lookup, "ds-1", "2027-02-01") is not None
    assert not any(isinstance(k, list) for k in lookup)
