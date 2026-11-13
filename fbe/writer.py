"""Write and validate the three forecast files (raw / control / platt).

Filename: `<due_date>.<organization>.<n>.json`, n in {1,2,3} for raw/control/platt
respectively. `model` field is `"<model-id> (raw|control|platt)"`. Each file must be
>=95% covered for BOTH market and dataset questions (a dataset question counts covered
only when every one of its resolution_dates has a row) or writing refuses.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

MIN_COVERAGE = 0.95
MODEL_ORGANIZATION = "Anthropic"
ARM_FILE_NUMBER = {"raw": 1, "control": 2, "platt": 3}
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.([A-Za-z0-9_-]+)\.([123])\.json$")


class ValidationError(RuntimeError):
    pass


class CoverageError(ValidationError):
    pass


@dataclass(frozen=True)
class ForecastRow:
    id: str
    source: str
    forecast: float
    resolution_date: str | None
    reasoning: str


def build_file_dict(
    organization: str,
    model_id: str,
    arm: str,
    question_set_filename: str,
    rows: list[ForecastRow],
    model_organization: str = MODEL_ORGANIZATION,
) -> dict:
    return {
        "organization": organization,
        "model": f"{model_id} ({arm})",
        "model_organization": model_organization,
        "question_set": question_set_filename,
        "forecasts": [
            {
                "id": r.id,
                "source": r.source,
                "forecast": r.forecast,
                "resolution_date": r.resolution_date,
                "reasoning": r.reasoning,
            }
            for r in rows
        ],
    }


def filename_slug(organization: str) -> str:
    """The organization as it appears in the FILENAME: letters, digits, '_' and '-' only
    ("Owen Carlsen" -> "OwenCarlsen"). FRI's docs never say a space or other punctuation is
    safe in a submission filename, so it is dropped here; the `organization` field inside
    the file keeps the exact name, and that field is what the leaderboard shows."""
    slug = re.sub(r"[^A-Za-z0-9_-]", "", organization)
    if not slug:
        raise ValidationError(f"organization {organization!r} has no filename-safe characters")
    return slug


def filename_for(due_date: str, organization: str, arm: str, slug: str | None = None) -> str:
    """`slug` overrides the filename slug derived from `organization` (used for the
    anonymous arms, whose filename slug is configurable)."""
    return f"{due_date}.{slug or filename_slug(organization)}.{ARM_FILE_NUMBER[arm]}.json"


def compute_coverage(
    forecasts: list[dict],
    expected_market_ids: set[str],
    expected_dataset_question_dates: dict[str, set[str]],
) -> tuple[float, float]:
    present_market_ids = {
        f["id"] for f in forecasts if f.get("resolution_date") is None
    } & expected_market_ids
    market_cov = (
        len(present_market_ids) / len(expected_market_ids) if expected_market_ids else 1.0
    )

    present_by_qid: dict[str, set[str]] = {}
    for f in forecasts:
        if f.get("resolution_date") is not None:
            present_by_qid.setdefault(f["id"], set()).add(f["resolution_date"])
    covered_dataset_qs = sum(
        1
        for qid, expected_dates in expected_dataset_question_dates.items()
        if expected_dates.issubset(present_by_qid.get(qid, set()))
    )
    dataset_cov = (
        covered_dataset_qs / len(expected_dataset_question_dates)
        if expected_dataset_question_dates
        else 1.0
    )
    return market_cov, dataset_cov


def validate_schema(data: dict) -> None:
    required_top = {"organization", "model", "model_organization", "question_set", "forecasts"}
    missing = required_top - set(data)
    if missing:
        raise ValidationError(f"missing top-level keys: {sorted(missing)}")
    if not isinstance(data["forecasts"], list) or not data["forecasts"]:
        raise ValidationError("'forecasts' must be a non-empty list")
    required_row = {"id", "source", "forecast", "resolution_date", "reasoning"}
    for i, row in enumerate(data["forecasts"]):
        missing_row = required_row - set(row)
        if missing_row:
            raise ValidationError(f"forecasts[{i}] missing keys: {sorted(missing_row)}")
        p = row["forecast"]
        if not isinstance(p, (int, float)) or isinstance(p, bool):
            raise ValidationError(f"forecasts[{i}].forecast is not numeric: {p!r}")
        if not math.isfinite(p):
            raise ValidationError(f"forecasts[{i}].forecast is not finite: {p!r}")
        if not (0.0 <= p <= 1.0):
            raise ValidationError(f"forecasts[{i}].forecast {p} out of [0,1]")


def validate_filename(path: Path) -> None:
    if not FILENAME_RE.match(path.name):
        raise ValidationError(
            f"filename {path.name!r} doesn't match <due_date>.<organization>.<1|2|3>.json"
        )


def validate_coverage(
    data: dict,
    expected_market_ids: set[str],
    expected_dataset_question_dates: dict[str, set[str]],
) -> tuple[float, float]:
    market_cov, dataset_cov = compute_coverage(
        data["forecasts"], expected_market_ids, expected_dataset_question_dates
    )
    if market_cov < MIN_COVERAGE:
        raise CoverageError(f"market coverage {market_cov:.1%} < {MIN_COVERAGE:.0%}")
    if dataset_cov < MIN_COVERAGE:
        raise CoverageError(f"dataset coverage {dataset_cov:.1%} < {MIN_COVERAGE:.0%}")
    return market_cov, dataset_cov


def write_forecast_file(
    path: Path,
    organization: str,
    model_id: str,
    arm: str,
    question_set_filename: str,
    rows: list[ForecastRow],
    expected_market_ids: set[str],
    expected_dataset_question_dates: dict[str, set[str]],
    model_organization: str = MODEL_ORGANIZATION,
) -> dict:
    """Builds, coverage-checks, schema-checks, and writes one forecast file. Raises
    ValidationError/CoverageError and writes nothing if any check fails."""
    data = build_file_dict(
        organization, model_id, arm, question_set_filename, rows, model_organization
    )
    validate_schema(data)
    validate_coverage(data, expected_market_ids, expected_dataset_question_dates)
    validate_filename(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return data


def validate_file(
    path: Path,
    expected_market_ids: set[str] | None = None,
    expected_dataset_question_dates: dict[str, set[str]] | None = None,
) -> dict:
    """Standalone validator (`fbe validate FILE`). Coverage is checked only when the caller
    supplies the expected id sets (e.g. by having the question set cached); otherwise schema
    and filename are checked and coverage is reported as "not checked"."""
    data = json.loads(path.read_text())
    validate_schema(data)
    validate_filename(path)
    if expected_market_ids is not None and expected_dataset_question_dates is not None:
        validate_coverage(data, expected_market_ids, expected_dataset_question_dates)
    return data
