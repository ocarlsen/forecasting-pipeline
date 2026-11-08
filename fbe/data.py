"""Fetch, cache, and normalize ForecastBench question/resolution sets.

Network calls here hit only the public forecastbench-datasets GitHub repo (reading
published question/resolution sets is explicitly allowed -- see docs/BUILD_SPEC.md).
Nothing here ever calls the Anthropic API.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_RAW = "https://raw.githubusercontent.com/forecastingresearch/forecastbench-datasets/main"
REPO_API = "https://api.github.com/repos/forecastingresearch/forecastbench-datasets/contents"
QUESTION_SETS_DIR = "datasets/question_sets"
RESOLUTION_SETS_DIR = "datasets/resolution_sets"

# Sets before this date mix in combination questions (id is a list of two question ids) --
# BUILD_SPEC.md says to drop them outright rather than try to score them.
COMBO_CUTOFF_DATE = dt.date(2025, 10, 26)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


class DatasetFetchError(RuntimeError):
    """Raised when a question/resolution set can't be fetched or parsed."""


def cache_dir(base: Path | None = None) -> Path:
    d = base or DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _http_get(url: str, headers: dict | None = None, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "forecasting-pipeline"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (public data only)
            return resp.read()
    except urllib.error.HTTPError as e:
        raise DatasetFetchError(f"GET {url} -> HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise DatasetFetchError(f"GET {url} -> {e}") from e


def fetch_json(url: str, cache_key: str, cache_base: Path | None = None, refresh: bool = False) -> dict:
    """Fetch JSON from `url`, caching the parsed result under data/cache/<cache_key>.json."""
    cdir = cache_dir(cache_base)
    cache_path = cdir / f"{cache_key}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    raw = _http_get(url)
    text = raw.decode("utf-8")
    data = json.loads(text)
    cache_path.write_text(json.dumps(data))
    return data


def list_question_set_dates(cache_base: Path | None = None, refresh: bool = False) -> list[str]:
    """List YYYY-MM-DD dates that have a published `<date>-llm.json` question set.

    Uses the GitHub contents API (more robust than date arithmetic, since real set dates
    have gaps -- e.g. nothing between 2024-07-21 and 2025-03-02).
    """
    cdir = cache_dir(cache_base)
    cache_path = cdir / "question_set_dates.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    raw = _http_get(f"{REPO_API}/{QUESTION_SETS_DIR}")
    entries = json.loads(raw.decode("utf-8"))
    dates = sorted(
        e["name"][: -len("-llm.json")]
        for e in entries
        if e.get("name", "").endswith("-llm.json") and not e["name"].startswith("latest")
    )
    cache_path.write_text(json.dumps(dates))
    return dates


def load_question_set(date_str: str, cache_base: Path | None = None, refresh: bool = False) -> dict:
    url = f"{REPO_RAW}/{QUESTION_SETS_DIR}/{date_str}-llm.json"
    return fetch_json(url, f"qset_{date_str}", cache_base, refresh)


def load_resolution_set(date_str: str, cache_base: Path | None = None, refresh: bool = False) -> dict | None:
    """Return the resolution set for `date_str`, or None if not published yet (404)."""
    url = f"{REPO_RAW}/{RESOLUTION_SETS_DIR}/{date_str}_resolution_set.json"
    try:
        return fetch_json(url, f"rset_{date_str}", cache_base, refresh)
    except DatasetFetchError:
        return None


def is_combo(question: dict) -> bool:
    return isinstance(question.get("id"), list)


def is_dataset_question(question: dict) -> bool:
    return isinstance(question.get("resolution_dates"), list)


def render_template(template: str, forecast_due_date: str, resolution_date: str) -> str:
    return template.format(forecast_due_date=forecast_due_date, resolution_date=resolution_date)


@dataclass(frozen=True)
class NormalizedRow:
    """One forecastable row -- exactly what a single model call must produce a probability for.

    A market question yields one row (resolution_date=None). A dataset question yields one
    row per entry in its `resolution_dates` list, each with the template rendered for that date.
    """

    id: str  # original question id (NOT composite -- a dataset question may have several rows)
    source: str
    kind: str  # "market" | "dataset"
    question: str
    resolution_criteria: str
    background: str
    resolution_date: str | None
    freeze_datetime: str
    freeze_datetime_value: str
    freeze_datetime_value_explanation: str
    forecast_due_date: str


@dataclass(frozen=True)
class NormalizeStats:
    n_market_questions: int
    n_dataset_questions: int
    n_combo_dropped: int
    n_market_rows: int
    n_dataset_rows: int


def normalize_question_set(qset: dict) -> tuple[list[NormalizedRow], NormalizeStats]:
    due_date = qset["forecast_due_date"]
    rows: list[NormalizedRow] = []
    n_market_q = n_dataset_q = n_combo = 0
    for q in qset["questions"]:
        if is_combo(q):
            n_combo += 1
            continue
        freeze = q.get("freeze_datetime", "N/A")
        fval = q.get("freeze_datetime_value", "N/A")
        fexpl = q.get("freeze_datetime_value_explanation", "N/A")
        if is_dataset_question(q):
            n_dataset_q += 1
            for resolution_date in q["resolution_dates"]:
                rows.append(
                    NormalizedRow(
                        id=q["id"],
                        source=q["source"],
                        kind="dataset",
                        question=render_template(q["question"], due_date, resolution_date),
                        resolution_criteria=q.get("resolution_criteria", "N/A"),
                        background=q.get("background", "N/A"),
                        resolution_date=resolution_date,
                        freeze_datetime=freeze,
                        freeze_datetime_value=fval,
                        freeze_datetime_value_explanation=fexpl,
                        forecast_due_date=due_date,
                    )
                )
        else:
            n_market_q += 1
            rows.append(
                NormalizedRow(
                    id=q["id"],
                    source=q["source"],
                    kind="market",
                    question=q["question"],
                    resolution_criteria=q.get("resolution_criteria", "N/A"),
                    background=q.get("background", "N/A"),
                    resolution_date=None,
                    freeze_datetime=freeze,
                    freeze_datetime_value=fval,
                    freeze_datetime_value_explanation=fexpl,
                    forecast_due_date=due_date,
                )
            )
    stats = NormalizeStats(
        n_market_questions=n_market_q,
        n_dataset_questions=n_dataset_q,
        n_combo_dropped=n_combo,
        n_market_rows=sum(1 for r in rows if r.kind == "market"),
        n_dataset_rows=sum(1 for r in rows if r.kind == "dataset"),
    )
    return rows, stats


def resolution_lookup(resolution_set: dict | None) -> dict[str, list[dict]]:
    """Map question id -> its resolved resolution row(s), skipping combo rows and unresolved
    rows (market questions are frequently `resolved: false`; only `resolved: true` is usable).

    Unlike a question's forecast row, a resolution row ALWAYS carries a real
    `resolution_date` -- even for a market question, which has exactly one. Keyed by id alone
    (not (id, resolution_date)) so a market row (whose forecast-row convention is
    resolution_date=None) can still be found; use `find_resolution()` to look one up the way
    a NormalizedRow / forecast row would ask for it.
    """
    out: dict[str, list[dict]] = {}
    if resolution_set is None:
        return out
    for r in resolution_set.get("resolutions", []):
        if isinstance(r.get("id"), list):
            continue  # combo row, dropped like everywhere else
        if not r.get("resolved"):
            continue
        out.setdefault(r["id"], []).append(r)
    return out


def find_resolution(
    lookup: dict[str, list[dict]], qid: str, resolution_date: str | None
) -> dict | None:
    """Look up a resolution row the way a forecast row would ask for it: resolution_date=None
    for a market question (exactly one resolution row expected), or a specific date string for
    one horizon of a dataset question."""
    candidates = lookup.get(qid, [])
    if not candidates:
        return None
    if resolution_date is None:
        return candidates[0]
    for r in candidates:
        if r.get("resolution_date") == resolution_date:
            return r
    return None
