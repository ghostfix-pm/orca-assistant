import json
import shutil
from pathlib import Path

import pytest

from orca_assistant.cli import main

WATER = Path(__file__).parent / "data" / "water.out"
pytestmark = pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")


def two_copies(tmp_path):
    a, b = tmp_path / "conf_a.out", tmp_path / "conf_b.out"
    shutil.copy(WATER, a)
    shutil.copy(WATER, b)
    return a, b


def test_table_markdown(tmp_path, capsys):
    a, b = two_copies(tmp_path)
    assert main(["table", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "conf_a" in out and "conf_b" in out and "50.0" in out


def test_table_json_and_file(tmp_path):
    a, b = two_copies(tmp_path)
    dest = tmp_path / "t.json"
    assert main(["table", str(a), str(b), "--format", "json", "-o", str(dest)]) == 0
    data = json.loads(dest.read_text())
    assert data["basis"] == "G" and data["rows"][0]["rel_g_kcal"] == 0


def test_table_bad_basis_returns_error(tmp_path, capsys):
    a, _ = two_copies(tmp_path)
    assert main(["table", str(a), "--temperature", "-5"]) == 1
    assert "error" in capsys.readouterr().err
