"""Energy tables from several parsed ORCA outputs: relative energies in
kcal/mol and Boltzmann populations. Standard library only."""
from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import asdict, dataclass, field

HARTREE_TO_KCAL = 627.5094740631
KB_KCAL_PER_MOL_K = 0.0019872041


@dataclass
class Row:
    name: str
    energy_eh: float | None
    zpe_eh: float | None
    enthalpy_eh: float | None
    gibbs_eh: float | None
    n_imaginary: int | None
    rel_e_kcal: float | None
    rel_g_kcal: float | None
    population_pct: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class Table:
    rows: list[Row]
    basis: str                 # "G" (Gibbs free energy) or "E" (electronic energy)
    temperature_k: float
    warnings: list[str] = field(default_factory=list)


def _relative(values: list[float | None]) -> list[float | None]:
    present = [v for v in values if v is not None]
    if not present:
        return [None] * len(values)
    lo = min(present)
    return [None if v is None else (v - lo) * HARTREE_TO_KCAL for v in values]


def build_rows(entries, temperature: float = 298.15, by: str = "auto",
               sort: bool = False) -> Table:
    """entries: list of (name, parsed_output). `by` is "auto", "g" or "e"."""
    if by not in ("auto", "g", "e"):
        raise ValueError(f"unknown basis {by!r}; use auto, g or e")
    if temperature <= 0:
        raise ValueError("temperature must be positive (kelvin)")
    if not entries:
        raise ValueError("no outputs given")

    warnings: list[str] = []
    missing_g = [n for n, p in entries if p.gibbs_eh is None]
    if by == "g":
        if missing_g:
            raise ValueError("no Gibbs free energy in: " + ", ".join(missing_g)
                             + " (the calculation needs Freq)")
        basis = "G"
    elif by == "e":
        basis = "E"
    else:
        basis = "G" if not missing_g else "E"
        if missing_g and len(missing_g) < len(entries):
            warnings.append("Gibbs free energy is missing in " + ", ".join(missing_g)
                            + "; using electronic energies for all rows.")

    energies = [p.final_energy_eh for _, p in entries]
    gibbs = [p.gibbs_eh for _, p in entries]
    chosen = gibbs if basis == "G" else energies
    for (name, _), v in zip(entries, chosen):
        if v is None:
            warnings.append(f"{name}: no {'Gibbs' if basis == 'G' else 'electronic'} "
                            "energy found; excluded from relative values.")

    rel_e, rel_g, rel_c = _relative(energies), _relative(gibbs), _relative(chosen)
    present = [v for v in chosen if v is not None]
    kt = KB_KCAL_PER_MOL_K * temperature
    weights = [None if v is None else math.exp(-r / kt) for v, r in zip(chosen, rel_c)]
    total = sum(w for w in weights if w is not None)
    pops = [None if w is None or not present else 100.0 * w / total for w in weights]

    rows = []
    for (name, p), re_, rg, pop in zip(entries, rel_e, rel_g, pops):
        notes = list(getattr(p, "problems", None) or [])
        rows.append(Row(name, p.final_energy_eh, p.zpe_eh, p.enthalpy_eh, p.gibbs_eh,
                        p.n_imaginary, re_, rg, pop, notes))
    if sort:
        rows.sort(key=lambda r: (r.population_pct is None, -(r.population_pct or 0.0)))
    return Table(rows, basis, temperature, warnings)


def _f(v, nd):
    return "-" if v is None else f"{v:.{nd}f}"


def to_markdown(t: Table) -> str:
    head = ["Name", "E (Eh)", "ZPE (Eh)", "H (Eh)", "G (Eh)", "N imag",
            "ΔE (kcal/mol)", "ΔG (kcal/mol)", "Pop. (%)", "Notes"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in t.rows:
        cells = [r.name, _f(r.energy_eh, 8), _f(r.zpe_eh, 8), _f(r.enthalpy_eh, 8),
                 _f(r.gibbs_eh, 8), "-" if r.n_imaginary is None else str(r.n_imaginary),
                 _f(r.rel_e_kcal, 2), _f(r.rel_g_kcal, 2), _f(r.population_pct, 1),
                 "; ".join(r.notes) or "-"]
        lines.append("| " + " | ".join(cells) + " |")
    what = "Gibbs free energy" if t.basis == "G" else "electronic energy"
    lines.append("")
    lines.append(f"Populations use the {what} at {t.temperature_k:g} K (Boltzmann).")
    return "\n".join(lines) + "\n"


def to_csv(t: Table) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["name", "energy_eh", "zpe_eh", "enthalpy_eh", "gibbs_eh", "n_imaginary",
                "rel_e_kcal_mol", "rel_g_kcal_mol", "population_pct", "notes"])
    for r in t.rows:
        w.writerow([r.name] + ["" if v is None else repr(v) for v in (
            r.energy_eh, r.zpe_eh, r.enthalpy_eh, r.gibbs_eh, r.n_imaginary,
            r.rel_e_kcal, r.rel_g_kcal, r.population_pct)] + ["; ".join(r.notes)])
    return buf.getvalue()


def to_json(t: Table) -> str:
    return json.dumps({"basis": t.basis, "temperature_k": t.temperature_k,
                       "warnings": t.warnings, "rows": [asdict(r) for r in t.rows]},
                      indent=2, ensure_ascii=False) + "\n"


def render_table(t: Table, fmt: str = "md") -> str:
    try:
        return {"md": to_markdown, "csv": to_csv, "json": to_json}[fmt](t)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; use md, csv or json") from None
