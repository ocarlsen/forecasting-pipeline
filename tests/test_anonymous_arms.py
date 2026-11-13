"""FRI allows at most one public file per round: the --public-arm file carries --org and
model_organization "Anthropic"; the other two arms set BOTH fields to --anon-org. Offline, $0.
"""
import json
from argparse import Namespace

import pytest

import fbe.cli as cli
import fbe.pipeline as pipeline
from fbe import writer
from fbe.writer import validate_file

DATE = "2027-01-01"
ANON = "Anonymous 999"


@pytest.fixture(autouse=True)
def _offline(monkeypatch, question_set, resolution_set, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_question_set", lambda d, **kw: question_set)
    monkeypatch.setattr(cli, "load_resolution_set", lambda d, **kw: resolution_set)
    monkeypatch.setattr(pipeline, "load_resolution_set", lambda d, **kw: resolution_set)
    monkeypatch.setattr(pipeline, "list_question_set_dates", lambda **kw: [])


def _args(**over):
    base = dict(
        model="claude-haiku-4-5", date=DATE, org="Owen Carlsen", max_usd=1000.0, seed=1,
        no_batch=False, dry_run=True, allow_failures=False, resume_batch=None, from_raw=None,
        public_arm="raw", anon_org=ANON, anon_slug=None,
    )
    base.update(over)
    return Namespace(**base)


def _read(slug, n):
    path = cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"{DATE}.{slug}.{n}.json"
    assert path.exists(), path
    return json.loads(path.read_text())


def test_raw_public_control_and_platt_anonymous(capsys):
    cli.cmd_live(_args())
    raw = _read("OwenCarlsen", 1)
    assert (raw["organization"], raw["model_organization"]) == ("Owen Carlsen", "Anthropic")
    for n in (2, 3):
        d = _read("Anonymous999", n)
        assert d["organization"] == ANON and d["model_organization"] == ANON
    out = capsys.readouterr().out
    assert "model_organization='Anonymous 999'" in out and "model_organization='Anthropic'" in out
    for n in (1, 2, 3):
        slug = "OwenCarlsen" if n == 1 else "Anonymous999"
        validate_file(cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"{DATE}.{slug}.{n}.json")


def test_public_arm_control_swaps():
    cli.cmd_live(_args(public_arm="control"))
    pub = _read("OwenCarlsen", 2)
    assert (pub["organization"], pub["model_organization"]) == ("Owen Carlsen", "Anthropic")
    for n in (1, 3):
        d = _read("Anonymous999", n)
        assert d["organization"] == ANON and d["model_organization"] == ANON


def test_anon_slug_override_changes_filenames_only():
    cli.cmd_live(_args(anon_slug="Anon_999"))
    for n in (2, 3):
        d = _read("Anon_999", n)
        assert d["organization"] == ANON and d["model_organization"] == ANON
    assert not (cli.OUT_DIR / cli.DRYRUN_OUT_SUBDIR / f"{DATE}.Anonymous999.2.json").exists()


@pytest.mark.parametrize(
    "over",
    [
        dict(anon_org=None),
        dict(anon_org=""),
        dict(anon_org="   "),
        dict(anon_org="Owen Carlsen"),
        dict(anon_org=" Owen Carlsen "),
        dict(anon_slug="bad slug"),
        dict(anon_slug=""),
        dict(anon_slug="a/b"),
        dict(anon_org="!!!"),
    ],
)
def test_refusals_happen_before_any_work(over, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("work started before arg validation")

    monkeypatch.setattr(cli, "load_question_set", boom)
    with pytest.raises(SystemExit) as e:
        cli.cmd_live(_args(**over))
    assert e.value.code == 2
    assert "REFUSED" in capsys.readouterr().err


def test_parser_flags_and_defaults():
    a = cli.build_parser().parse_args(
        ["live", "--model", "m", "--date", DATE, "--org", "O", "--max-usd", "1"]
    )
    assert a.public_arm == "raw" and a.anon_org is None and a.anon_slug is None


def test_writer_model_organization_param_and_default():
    rows = [writer.ForecastRow("m1", "s", 0.5, None, "r")]
    assert writer.build_file_dict("O", "m", "raw", "q.json", rows)["model_organization"] == "Anthropic"
    d = writer.build_file_dict("A", "m", "raw", "q.json", rows, model_organization="A")
    assert d["organization"] == d["model_organization"] == "A"
    assert writer.filename_for(DATE, "Anonymous 999", "platt") == f"{DATE}.Anonymous999.3.json"
    assert writer.filename_for(DATE, "Anonymous 999", "platt", "X_1") == f"{DATE}.X_1.3.json"
