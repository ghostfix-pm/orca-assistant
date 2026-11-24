import json
from pathlib import Path

import pytest

from orca_assistant.cli import main

WATER = Path(__file__).parent / "data" / "water.out"
pytestmark = pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")


def test_parse_shows_extras(capsys):
    main(["parse", str(WATER)])
    out = capsys.readouterr().out
    assert "ORCA version        : 6.1.1" in out
    assert "HOMO/LUMO" in out and "gap 9.528 eV" in out
    assert "O0 -0.720" in out


def test_parse_json_includes_extras(capsys):
    main(["parse", str(WATER), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["orca_version"] == "6.1.1" and data["final_energy_eh"] < -76
    assert len(data["opt_steps"]) == 4


def test_xyz_last_and_all(tmp_path):
    one, many = tmp_path / "last.xyz", tmp_path / "traj.xyz"
    assert main(["xyz", str(WATER), "-o", str(one)]) == 0
    assert one.read_text().splitlines()[0] == "3"
    assert main(["xyz", str(WATER), "--all", "-o", str(many)]) == 0
    assert many.read_text().count("frame ") == 5
