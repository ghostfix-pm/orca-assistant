from types import SimpleNamespace

import pytest

from orca_assistant import plan_calculation
from orca_assistant.llm import TOOL_NAME, describe_molecule
from orca_assistant.spec import atoms_from_xyz

WATER = atoms_from_xyz("O 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469")


class FakeClient:
    """Mimics client.messages.create(...) with scripted tool-use responses."""

    def __init__(self, tool_inputs):
        self.inputs = list(tool_inputs)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        block = SimpleNamespace(type="tool_use", id="toolu_1", name=TOOL_NAME,
                                input=self.inputs.pop(0))
        return SimpleNamespace(content=[block])


def test_plan_happy_path():
    client = FakeClient([{
        "task": "opt_freq",
        "method": {"functional": "wB97X-D3", "basis": "def2-TZVP",
                   "dispersion": None, "solvent": "water"},
        "resources": {"nprocs": 8, "maxcore_mb": 3000},
        "assumptions": ["no cores given: used 8"],
    }])
    planned = plan_calculation("opt+freq wB97X-D3 def2-TZVP water", WATER, client=client)
    assert planned.spec.method.functional == "wB97X-D3"
    assert planned.spec.method.dispersion is None
    assert planned.spec.molecule.atoms == WATER
    assert planned.assumptions == ["no cores given: used 8"]
    sent = client.calls[0]
    assert sent["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert "H2O" in sent["messages"][0]["content"]  # molecule described, not coordinates
    assert "0.757" not in sent["messages"][0]["content"]


def test_plan_retries_after_validation_error():
    bad = {"task": "sp", "multiplicity": 2}   # water cannot be a doublet
    good = {"task": "sp", "multiplicity": 1}
    client = FakeClient([bad, good])
    planned = plan_calculation("single point", WATER, client=client)
    assert planned.spec.molecule.multiplicity == 1
    assert len(client.calls) == 2
    last = client.calls[-1]["messages"][-1]["content"][0]
    assert last["is_error"] and "multiplicity" in last["content"]


def test_plan_gives_up_after_retries():
    client = FakeClient([{"task": "sp", "multiplicity": 2}] * 3)
    with pytest.raises(RuntimeError):
        plan_calculation("single point", WATER, client=client, max_retries=2)


def test_describe_molecule():
    d = describe_molecule(WATER)
    assert "H2O" in d and "10 electrons" in d and "0:O" in d
