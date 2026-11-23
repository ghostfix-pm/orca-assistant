from pathlib import Path

import pytest

from orca_assistant import parse_output, parse_file

# Fragments copied from a real ORCA run (water, B3LYP/def2-TZVP, Opt Freq).
REAL = """\
================================================================================
                                        WARNINGS
                       Please study these warnings very carefully!
================================================================================


WARNING: Geometry Optimization
  ===> : Switching off AutoStart
         For restart on a previous wavefunction, please use MOREAD

================================================================================
                                       INPUT FILE
================================================================================
                    ***        THE OPTIMIZATION HAS CONVERGED     ***
FINAL SINGLE POINT ENERGY       -76.437322517640
VIBRATIONAL FREQUENCIES
-----------------------

Scaling factor for frequencies =  1.000000000  (already applied!)

     0:       0.00 cm**-1
     1:       0.00 cm**-1
     2:       0.00 cm**-1
     3:       0.00 cm**-1
     4:       0.00 cm**-1
     5:       0.00 cm**-1
     6:    1598.87 cm**-1
     7:    3763.29 cm**-1
     8:    3846.90 cm**-1


------------
NORMAL MODES
------------
                             ****ORCA TERMINATED NORMALLY****
TOTAL RUN TIME: 0 days 0 hours 0 minutes 17 seconds 282 msec
"""


def test_real_water_fragments():
    r = parse_output(REAL)
    assert r.terminated_normally
    assert r.opt_converged is True
    assert r.final_energy_eh == pytest.approx(-76.437322517640)
    assert r.frequencies_cm1 == [1598.87, 3763.29, 3846.90]
    assert r.n_imaginary == 0
    assert r.runtime_s == pytest.approx(17.282)
    assert any("AutoStart" in w for w in r.warnings)
    assert r.problems == []


def test_imaginary_frequency_flagged():
    text = REAL.replace("6:    1598.87", "6:    -312.40")
    r = parse_output(text)
    assert r.n_imaginary == 1
    assert any("imaginary" in p for p in r.problems)


def test_not_terminated_flagged():
    r = parse_output(REAL.replace("ORCA TERMINATED NORMALLY", "ABORTED"))
    assert "did not terminate normally" in r.problems[0]


def test_empty_output_is_safe():
    r = parse_output("")
    assert r.final_energy_eh is None and r.frequencies_cm1 == []


@pytest.mark.skipif(not Path("tests/data/water.out").exists(), reason="no local ORCA output")
def test_local_real_output():
    r = parse_file("tests/data/water.out")
    assert r.terminated_normally and r.final_energy_eh is not None
