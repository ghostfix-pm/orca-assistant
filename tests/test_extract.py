from pathlib import Path

import pytest

from orca_assistant.extract import (
    extract, parse_charges, parse_frontier, parse_geometries, parse_opt_steps, write_xyz,
)

WATER = Path(__file__).parent / "data" / "water.out"
needs_water = pytest.mark.skipif(not WATER.exists(), reason="tests/data/water.out missing")


@needs_water
def test_water_version_and_geometry():
    ex = extract(WATER.read_text())
    assert ex.orca_version == "6.1.1"
    assert ex.n_atoms == 3 and ex.n_geometries == 5


@needs_water
def test_water_last_geometry_is_optimised():
    geoms = parse_geometries(WATER.read_text())
    el, x, y, z = geoms[-1][0]
    assert el == "O" and abs(z - 0.120662) < 1e-6
    assert geoms[0][0][3] == 0.117


@needs_water
def test_water_frontier_orbitals():
    (fo,) = parse_frontier(WATER.read_text())
    assert fo.channel == "restricted"
    assert (fo.homo_index, fo.lumo_index) == (4, 5)
    assert abs(fo.homo_ev + 8.5602) < 1e-4 and abs(fo.lumo_ev - 0.9678) < 1e-4
    assert abs(fo.gap_ev - 9.5280) < 1e-3


@needs_water
def test_water_charges():
    text = WATER.read_text()
    mull, loew = parse_charges(text, "mulliken"), parse_charges(text, "loewdin")
    assert [c.element for c in mull] == ["O", "H", "H"]
    assert mull[0].charge == -0.719763 and loew[0].charge == -0.347402
    assert mull[0].spin is None


@needs_water
def test_water_opt_steps():
    steps = parse_opt_steps(WATER.read_text())
    assert [s.cycle for s in steps] == [1, 2, 3, 4]
    assert {s.job for s in steps} == {1}
    assert steps[0].energy_eh == -76.437204896778 and steps[3].energy_eh == -76.437330099904


@needs_water
def test_water_ir():
    modes = extract(WATER.read_text()).ir_modes
    assert [m.mode for m in modes] == [6, 7, 8]
    assert modes[0].freq_cm1 == 1598.87 and modes[0].intensity_km_mol == 125.31


def test_write_xyz():
    s = write_xyz([("O", 0.0, 0.0, 0.1), ("H", 0.0, 0.7, -0.4)], "demo")
    lines = s.splitlines()
    assert lines[0] == "2" and lines[1] == "demo" and lines[2].startswith("O ")


def test_open_shell_orbitals_per_spin():
    text = """ORBITAL ENERGIES
----------------

-----------------
SPIN UP ORBITALS
-----------------
  NO   OCC          E(Eh)            E(eV)
   0   1.0000      -0.500       -13.6057
   1   1.0000      -0.300        -8.1634
   2   0.0000       0.100         2.7211
-----------------
SPIN DOWN ORBITALS
-----------------
  NO   OCC          E(Eh)            E(eV)
   0   1.0000      -0.480       -13.0613
   1   0.0000       0.050         1.3606
"""
    a, b = parse_frontier(text)
    assert (a.channel, b.channel) == ("alpha", "beta")
    assert a.homo_index == 1 and a.lumo_index == 2
    assert b.homo_index == 0 and abs(b.gap_ev - 14.4219) < 1e-3


def test_charges_with_spin_populations():
    text = """MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
--------------------------------------------
   0 O :   -0.500000    0.700000
   1 Cl:    0.500000    0.300000
Sum of atomic charges:    0.0000000
"""
    c = parse_charges(text, "mulliken")
    assert c[1].element == "Cl" and c[0].spin == 0.7 and c[1].charge == 0.5


def test_opt_steps_restart_counts_jobs():
    text = ("GEOMETRY OPTIMIZATION CYCLE   1\nFINAL SINGLE POINT ENERGY  -1.5\n"
            "GEOMETRY OPTIMIZATION CYCLE   2\nFINAL SINGLE POINT ENERGY  -1.6\n"
            "GEOMETRY OPTIMIZATION CYCLE   1\nFINAL SINGLE POINT ENERGY  -1.4\n"
            "GEOMETRY OPTIMIZATION CYCLE   2\n")
    steps = parse_opt_steps(text)
    assert [s.job for s in steps] == [1, 1, 2, 2]
    assert steps[3].energy_eh is None


def test_empty_text_is_safe():
    ex = extract("")
    assert ex.orca_version is None and ex.frontier == [] and ex.opt_steps == []
