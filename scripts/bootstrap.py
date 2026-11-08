"""Paired bootstrap for a backtest run: are the arm and model differences real or noise?

Rebuilds every scored row of a backtest run (raw, control, platt, copier, outcome) from
runs/history.jsonl and the cached question and resolution sets, using the same fbe functions
the backtest used. It first checks the rebuilt rows reproduce the run's scores.json exactly,
then resamples whole questions (a dataset question's horizons move together) and prints the
mean Brier difference with a 95% interval.

Offline, $0. Usage, from the repo root:
    .venv/bin/python scripts/bootstrap.py --run runs/20261028T003636Z [--model claude-opus-5-5]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fbe import calibrate, control, data, pipeline  # noqa: E402


def rebuild_rows(run: Path, history_path: Path) -> tuple[list[dict], dict]:
    scores = json.loads((run / "scores.json").read_text())
    models, sets = scores["models"], scores["sets"]
    hist = pipeline.load_history(history_path)

    qmeta = {}
    for s in sets:
        for q in data.load_question_set(s)["questions"]:
            if not isinstance(q["id"], list):
                qmeta[(s, q["id"])] = q
    rlook = {s: data.resolution_lookup(data.load_resolution_set(s)) for s in sets}

    rows = []
    for m in models:
        entries = [e for e in hist if e.model_id == m]
        for s in sets:
            due = dt.date.fromisoformat(s)
            earlier = [e for e in entries if e.due_date < s]
            pool = pipeline.leak_free_filter(pipeline.resolve_history_entries(earlier), due)
            fits = {
                kind: calibrate.fit_platt_for_type(
                    [(r.raw_p, r.outcome) for r in pool if r.kind == kind]
                )[0]
                for kind in ("market", "dataset")
            }
            base_rates = pipeline.public_base_rates(due)
            for e in (e for e in entries if e.due_date == s):
                res = data.find_resolution(rlook[s], e.id, e.resolution_date)
                if res is None or res.get("resolution_date") in (None, "N/A"):
                    continue
                q = qmeta[(s, e.id)]
                if e.kind == "market":
                    anchor = control.market_anchor(q.get("freeze_datetime_value", "N/A"))
                else:
                    bucket = control.nearest_horizon_bucket(
                        due, dt.date.fromisoformat(e.resolution_date)
                    )
                    anchor = control.dataset_anchor(base_rates, e.source, bucket)
                rows.append(dict(
                    model=m, set=s, id=e.id, kind=e.kind, resolution_date=e.resolution_date,
                    y=float(res["resolved_to"]), raw=e.raw_p, anchor=anchor,
                    control=control.control_probability(e.raw_p, anchor),
                    platt=calibrate.apply_platt_fit(fits[e.kind], e.raw_p),
                ))
    return rows, scores


def check_against_scores(rows: list[dict], scores: dict) -> int:
    bad = 0
    for m in scores["models"]:
        for s in scores["sets"]:
            for kind in ("market", "dataset"):
                sub = [r for r in rows if r["model"] == m and r["set"] == s and r["kind"] == kind]
                for arm in ("raw", "control", "platt"):
                    want = scores["scores"][m][s][arm][f"{kind}_brier"]
                    got = sum((r[arm] - r["y"]) ** 2 for r in sub) / len(sub) if sub else None
                    if (want is None) != (got is None) or (
                        want is not None and abs(want - got) > 1e-9
                    ):
                        bad += 1
    return bad


def paired_bootstrap(rows, a, b, rng, n_boot):
    """Mean of brier(a) - brier(b) with a 95% interval, resampling whole questions."""
    clusters = collections.defaultdict(list)
    for r in rows:
        clusters[(r["set"], r["id"])].append((r[a] - r["y"]) ** 2 - (r[b] - r["y"]) ** 2)
    keys = list(clusters)
    observed = sum(d for k in keys for d in clusters[k]) / len(rows)
    draws = []
    for _ in range(n_boot):
        total = count = 0
        for k in (rng.choice(keys) for _ in keys):
            total += sum(clusters[k])
            count += len(clusters[k])
        draws.append(total / count)
    draws.sort()
    return observed, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot)], len(keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--history", type=Path, default=REPO / "runs" / "history.jsonl")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows, scores = rebuild_rows(args.run, args.history)
    bad = check_against_scores(rows, scores)
    print(f"rebuilt {len(rows)} scored rows; mismatches vs scores.json: {bad}")
    if bad:
        sys.exit("rebuilt rows do not reproduce scores.json; refusing to report intervals")

    rng = random.Random(args.seed)
    rows_m = [dict(r, copier=r["anchor"]) for r in rows if r["model"] == args.model]
    market = [r for r in rows_m if r["kind"] == "market"]
    dataset = [r for r in rows_m if r["kind"] == "dataset"]

    print(f"\n{args.model}: difference in Brier (negative = first is better), 95% interval")
    for label, rs, a, b in (
        ("market", market, "raw", "copier"),
        ("market", market, "platt", "raw"),
        ("market", market, "control", "raw"),
        ("dataset", dataset, "platt", "raw"),
        ("dataset", dataset, "control", "raw"),
    ):
        d, lo, hi, k = paired_bootstrap(rs, a, b, rng, args.boot)
        print(f"  {label:8s} {a} - {b}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  rows={len(rs)} questions={k}")

    print(f"\n{args.model} vs the other candidates, raw Brier on the same rows")
    by_key = collections.defaultdict(dict)
    for r in rows:
        by_key[(r["set"], r["id"], r["resolution_date"])][r["model"]] = r
    for other in (m for m in scores["models"] if m != args.model):
        for kind in ("market", "dataset"):
            pairs = [
                dict(set=v[args.model]["set"], id=v[args.model]["id"], y=v[args.model]["y"],
                     mine=v[args.model]["raw"], theirs=v[other]["raw"])
                for v in by_key.values()
                if args.model in v and other in v and v[args.model]["kind"] == kind
            ]
            d, lo, hi, k = paired_bootstrap(pairs, "mine", "theirs", rng, args.boot)
            print(f"  {kind:8s} vs {other}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  rows={len(pairs)}")


if __name__ == "__main__":
    main()
