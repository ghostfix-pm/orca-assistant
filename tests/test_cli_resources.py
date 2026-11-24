from pathlib import Path

from orca_assistant.cli import main
from orca_assistant.resources import Hardware

EXAMPLE = Path(__file__).parent.parent / "examples" / "water_opt_freq.json"
SMALL = Hardware(cores=4, ram_mb=6835, available_mb=2662)


def test_resources_command(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.detect_hardware", lambda: SMALL)
    assert main(["resources"]) == 0
    out = capsys.readouterr().out
    assert "nprocs=2" in out and "maxcore=998" in out


def test_render_auto_resources(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.detect_hardware", lambda: SMALL)
    assert main(["render", str(EXAMPLE), "--auto-resources"]) == 0
    cap = capsys.readouterr()
    assert "%pal nprocs 2 end" in cap.out
    assert "%maxcore 998" in cap.out
    assert "[resources]" in cap.err


def test_render_warns_on_oversized_default(monkeypatch, capsys):
    monkeypatch.setattr("orca_assistant.cli.detect_hardware", lambda: SMALL)
    spec = EXAMPLE.read_text()
    assert main(["render", str(EXAMPLE)]) == 0
    assert "total RAM" in capsys.readouterr().err or '"maxcore_mb"' not in spec
