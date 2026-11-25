from pathlib import Path

import pytest

from orca_assistant.cli import main

WATER = Path(__file__).parent / "data" / "water.out"


@pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")
def test_watch_once_on_finished_file(capsys):
    assert main(["watch", str(WATER), "--once"]) == 0
    assert "finished normally" in capsys.readouterr().out


def test_watch_once_missing_file(tmp_path, capsys):
    assert main(["watch", str(tmp_path / "no.out"), "--once"]) == 2
    assert "waiting" in capsys.readouterr().out


def test_run_reports_missing_input(tmp_path, capsys):
    assert main(["run", str(tmp_path / "no.inp")]) == 3
    assert "input file not found" in capsys.readouterr().err


@pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")
def test_watch_plain_has_no_screen_codes(capsys):
    assert main(["watch", str(WATER), "--once", "--plain"]) == 0
    assert "\033" not in capsys.readouterr().out
