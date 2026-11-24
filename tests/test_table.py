import csv
import io
import json
import math
from types import SimpleNamespace

import pytest

from orca_assistant.table import (
    HARTREE_TO_KCAL, KB_KCAL_PER_MOL_K, build_rows, render_table,
)


def out(e, g=None, problems=None, n_imag=0):
    return SimpleNamespace(final_energy_eh=e, zpe_eh=0.02, enthalpy_eh=None, gibbs_eh=g,
                           n_imaginary=n_imag, problems=problems or [])


def test_relative_and_boltzmann():
    t = build_rows([("a", out(-100.0, -100.0)), ("b", out(-99.998, -99.998))])
    a, b = t.rows
    assert t.basis == "G"
    assert a.rel_g_kcal == 0 and abs(b.rel_g_kcal - 0.002 * HARTREE_TO_KCAL) < 1e-9
    ratio = math.exp(-b.rel_g_kcal / (KB_KCAL_PER_MOL_K * 298.15))
    assert abs(a.population_pct - 100 / (1 + ratio)) < 1e-9
    assert abs(a.population_pct + b.population_pct - 100) < 1e-9


def test_equal_energies_give_equal_populations():
    t = build_rows([("a", out(-1.0, -1.0)), ("b", out(-1.0, -1.0))])
    assert [round(r.population_pct, 6) for r in t.rows] == [50.0, 50.0]


def test_higher_temperature_flattens_populations():
    cold = build_rows([("a", out(-1.0, -1.0)), ("b", out(-0.998, -0.998))], temperature=200)
    hot = build_rows([("a", out(-1.0, -1.0)), ("b", out(-0.998, -0.998))], temperature=600)
    assert hot.rows[1].population_pct > cold.rows[1].population_pct


def test_auto_falls_back_to_electronic_energy_with_warning():
    t = build_rows([("a", out(-1.0, -1.0)), ("b", out(-0.999, None))])
    assert t.basis == "E" and any("missing in b" in w for w in t.warnings)
    assert t.rows[1].rel_e_kcal > 0 and t.rows[1].rel_g_kcal is None


def test_by_g_requires_gibbs():
    with pytest.raises(ValueError, match="Gibbs"):
        build_rows([("a", out(-1.0, None))], by="g")


def test_bad_arguments():
    with pytest.raises(ValueError):
        build_rows([("a", out(-1.0, -1.0))], by="x")
    with pytest.raises(ValueError):
        build_rows([("a", out(-1.0, -1.0))], temperature=0)
    with pytest.raises(ValueError):
        build_rows([])


def test_missing_energy_is_excluded_not_crashing():
    t = build_rows([("a", out(-1.0, -1.0)), ("b", out(None, None))])
    assert t.rows[1].population_pct is None and t.rows[0].population_pct == 100.0
    assert any("excluded" in w for w in t.warnings)


def test_problems_become_notes():
    t = build_rows([("a", out(-1.0, -1.0, problems=["optimisation did not converge"]))])
    assert t.rows[0].notes == ["optimisation did not converge"]
    assert "did not converge" in render_table(t, "md")


def test_sort_puts_most_populated_first():
    t = build_rows([("hi", out(-0.99, -0.99)), ("lo", out(-1.0, -1.0))], sort=True)
    assert [r.name for r in t.rows] == ["lo", "hi"]


def test_formats():
    t = build_rows([("a", out(-1.0, -1.0)), ("b", out(-0.999, -0.999))])
    md = render_table(t, "md")
    assert md.splitlines()[0].startswith("| Name") and "Boltzmann" in md
    rows = list(csv.reader(io.StringIO(render_table(t, "csv"))))
    assert rows[0][0] == "name" and rows[1][0] == "a" and len(rows) == 3
    data = json.loads(render_table(t, "json"))
    assert data["basis"] == "G" and len(data["rows"]) == 2
    with pytest.raises(ValueError):
        render_table(t, "xml")
