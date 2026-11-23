import json
import os
import stat

from orca_assistant.cli import main
from test_parse import REAL


def test_render_command(tmp_path):
    spec = {
        "molecule": {"atoms": [
            {"element": "O", "x": 0, "y": 0, "z": 0.117},
            {"element": "H", "x": 0, "y": 0.757, "z": -0.469},
            {"element": "H", "x": 0, "y": -0.757, "z": -0.469}]},
        "task": "sp",
    }
    (tmp_path / "s.json").write_text(json.dumps(spec))
    out = tmp_path / "job.inp"
    assert main(["render", str(tmp_path / "s.json"), "-o", str(out)]) == 0
    assert out.read_text().startswith("! B3LYP def2-SVP D3BJ")


def test_run_with_fake_orca(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "bin" / "orca"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\ncat <<'EOF'\n" + REAL + "EOF\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{fake.parent}{os.pathsep}{os.environ['PATH']}")

    inp = tmp_path / "job.inp"
    inp.write_text("! B3LYP\n")
    assert main(["run", str(inp)]) == 0
    assert (tmp_path / "job.out").exists()
    assert "terminated normally : True" in capsys.readouterr().out


def test_run_without_orca(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    inp = tmp_path / "job.inp"
    inp.write_text("! B3LYP\n")
    assert main(["run", str(inp)]) == 2
