import json
from pathlib import Path

import pytest

from fbe.writer import (
    CoverageError,
    ForecastRow,
    ValidationError,
    filename_for,
    validate_file,
    validate_filename,
    write_forecast_file,
)


def _rows(include_mkt2=True, include_ds2=True):
    rows = [
        ForecastRow("mkt-1", "manifold", 0.6, None, "r"),
        ForecastRow("ds-1", "acled", 0.5, "2027-01-08", "r"),
        ForecastRow("ds-1", "acled", 0.4, "2027-02-01", "r"),
    ]
    if include_mkt2:
        rows.append(ForecastRow("mkt-2", "kalshi", 0.3, None, "r"))
    if include_ds2:
        rows.append(ForecastRow("ds-2", "acled", 0.2, "2027-01-08", "r"))
    return rows


EXPECTED_MARKET = {"mkt-1", "mkt-2"}
EXPECTED_DATASET = {"ds-1": {"2027-01-08", "2027-02-01"}, "ds-2": {"2027-01-08"}}


def test_filename_format():
    assert filename_for("2027-01-01", "ocarlsen", "raw") == "2027-01-01.ocarlsen.1.json"
    assert filename_for("2027-01-01", "ocarlsen", "control") == "2027-01-01.ocarlsen.2.json"
    assert filename_for("2027-01-01", "ocarlsen", "platt") == "2027-01-01.ocarlsen.3.json"


def test_filename_drops_spaces_but_file_keeps_exact_org(tmp_path):
    name = filename_for("2026-11-08", "Owen Carlsen", "raw")
    assert name == "2026-11-08.OwenCarlsen.1.json"
    data = write_forecast_file(
        tmp_path / name, "Owen Carlsen", "claude-opus-5-5", "raw", "2026-11-08-llm.json",
        _rows(), EXPECTED_MARKET, EXPECTED_DATASET,
    )
    assert data["organization"] == "Owen Carlsen"


def test_filename_with_space_rejected():
    with pytest.raises(ValidationError):
        validate_filename(Path("2026-11-08.Owen Carlsen.1.json"))


def test_write_and_validate_full_coverage(tmp_path):
    path = tmp_path / filename_for("2027-01-01", "ocarlsen", "raw")
    data = write_forecast_file(
        path, "ocarlsen", "claude-haiku-4-5", "raw", "2027-01-01-llm.json",
        _rows(), EXPECTED_MARKET, EXPECTED_DATASET,
    )
    assert data["model"] == "claude-haiku-4-5 (raw)"
    assert data["model_organization"] == "Anthropic"
    validated = validate_file(path, EXPECTED_MARKET, EXPECTED_DATASET)
    assert validated == json.loads(path.read_text())


def test_coverage_refusal_market(tmp_path):
    path = tmp_path / filename_for("2027-01-01", "ocarlsen", "raw")
    with pytest.raises(CoverageError):
        write_forecast_file(
            path, "ocarlsen", "claude-haiku-4-5", "raw", "2027-01-01-llm.json",
            _rows(include_mkt2=False), EXPECTED_MARKET, EXPECTED_DATASET,
        )
    assert not path.exists()


def test_coverage_refusal_dataset_partial_horizon(tmp_path):
    # ds-1 missing one of its two resolution_dates -> that question doesn't count as covered
    rows = [
        ForecastRow("mkt-1", "manifold", 0.6, None, "r"),
        ForecastRow("mkt-2", "kalshi", 0.3, None, "r"),
        ForecastRow("ds-1", "acled", 0.5, "2027-01-08", "r"),  # missing 2027-02-01
        ForecastRow("ds-2", "acled", 0.2, "2027-01-08", "r"),
    ]
    path = tmp_path / filename_for("2027-01-01", "ocarlsen", "raw")
    with pytest.raises(CoverageError):
        write_forecast_file(
            path, "ocarlsen", "claude-haiku-4-5", "raw", "2027-01-01-llm.json",
            rows, EXPECTED_MARKET, EXPECTED_DATASET,
        )


def test_schema_rejects_out_of_range_probability(tmp_path):
    path = tmp_path / filename_for("2027-01-01", "ocarlsen", "raw")
    rows = _rows()
    rows[0] = ForecastRow("mkt-1", "manifold", 1.5, None, "r")
    with pytest.raises(ValidationError):
        write_forecast_file(
            path, "ocarlsen", "claude-haiku-4-5", "raw", "2027-01-01-llm.json",
            rows, EXPECTED_MARKET, EXPECTED_DATASET,
        )


def test_bad_filename_rejected(tmp_path):
    path = tmp_path / "not-a-valid-name.json"
    with pytest.raises(ValidationError):
        write_forecast_file(
            path, "ocarlsen", "claude-haiku-4-5", "raw", "2027-01-01-llm.json",
            _rows(), EXPECTED_MARKET, EXPECTED_DATASET,
        )
