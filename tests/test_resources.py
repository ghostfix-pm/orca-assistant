from orca_assistant.lint import lint
from orca_assistant.resources import (
    Hardware, detect_hardware, parse_meminfo, parse_physical_cores, suggest_resources,
)
from orca_assistant.spec import Atom, CalcSpec, Molecule, Resources

MEMINFO = "MemTotal:        7000000 kB\nMemFree:          100000 kB\nMemAvailable:    2726000 kB\n"
CPUINFO = (
    "processor\t: 0\nphysical id\t: 0\ncore id\t\t: 0\n\n"
    "processor\t: 1\nphysical id\t: 0\ncore id\t\t: 1\n\n"
    "processor\t: 2\nphysical id\t: 0\ncore id\t\t: 0\n\n"
    "processor\t: 3\nphysical id\t: 0\ncore id\t\t: 1\n"
)


def water(resources=None) -> CalcSpec:
    mol = Molecule(atoms=[Atom(element="O", x=0.0, y=0.0, z=0.117),
                          Atom(element="H", x=0.0, y=0.757, z=-0.469),
                          Atom(element="H", x=0.0, y=-0.757, z=-0.469)])
    return CalcSpec(molecule=mol, resources=resources or Resources())


def test_parse_meminfo():
    assert parse_meminfo(MEMINFO) == (6835, 2662)


def test_parse_meminfo_garbage():
    assert parse_meminfo("nonsense") == (None, None)


def test_parse_physical_cores_ignores_hyperthreads():
    assert parse_physical_cores(CPUINFO) == 2
    assert parse_physical_cores("") is None


def test_suggest_plenty_of_ram():
    r = suggest_resources(Hardware(cores=8, ram_mb=32000, available_mb=16000))
    assert (r.nprocs, r.maxcore_mb) == (8, 1500)


def test_suggest_tight_ram_reduces_procs():
    r = suggest_resources(Hardware(cores=4, ram_mb=6835, available_mb=2662))
    assert r.nprocs == 2
    assert r.nprocs * r.maxcore_mb <= 0.75 * 2662


def test_suggest_caps_maxcore():
    r = suggest_resources(Hardware(cores=1, ram_mb=200000, available_mb=100000))
    assert r.maxcore_mb == 8000


def test_suggest_unknown_ram():
    r = suggest_resources(Hardware(cores=16, ram_mb=None, available_mb=None))
    assert (r.nprocs, r.maxcore_mb) == (4, 2000)


def test_lint_flags_default_on_small_laptop():
    hw = Hardware(cores=4, ram_mb=6835, available_mb=2662)
    msgs = [i.message for i in lint(water(), hw)]
    assert any("total RAM" in m for m in msgs)
    assert any("Suggested for this machine" in m for m in msgs)


def test_lint_oversubscribed_cpu():
    hw = Hardware(cores=4, ram_mb=64000, available_mb=60000)
    msgs = [i.message for i in lint(water(Resources(nprocs=8, maxcore_mb=1000)), hw)]
    assert any("physical cores" in m for m in msgs)


def test_lint_quiet_when_it_fits():
    hw = Hardware(cores=8, ram_mb=32000, available_mb=30000)
    assert lint(water(Resources(nprocs=4, maxcore_mb=2000)), hw) == []


def test_lint_without_hardware_skips_checks():
    assert not any("RAM" in i.message for i in lint(water()))


def test_detect_hardware_returns_sane_values():
    hw = detect_hardware()
    assert hw.cores >= 1
    assert hw.ram_mb is None or hw.ram_mb > 0
