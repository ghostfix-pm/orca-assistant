"""Rule-based sanity checks. Warnings, not errors: the user decides."""
from __future__ import annotations

from dataclasses import dataclass

from .spec import CalcSpec, Task


@dataclass
class Issue:
    level: str  # "warning" | "info"
    message: str


def lint(spec: CalcSpec) -> list[Issue]:
    issues: list[Issue] = []
    m = spec.method
    name = m.functional.lower()

    if m.is_composite and (m.basis or m.dispersion):
        issues.append(Issue("warning",
            "Composite -3c method: basis/dispersion fields are ignored in the input."))
    if not m.is_composite and m.dispersion and ("-d3" in name or "-d4" in name):
        issues.append(Issue("warning",
            f"{m.functional} already contains a dispersion correction; "
            f"also adding {m.dispersion} would double-count it."))
    if not m.is_composite and not m.basis:
        issues.append(Issue("warning", "No basis set specified for a non-composite method."))
    if spec.task == Task.TS:
        issues.append(Issue("info",
            "TS search: verify exactly one imaginary frequency and check the mode "
            "(IRC or displacement) connects the intended minima."))
    if spec.task == Task.SCAN:
        issues.append(Issue("info",
            "Scan indices are 0-based. Re-optimise the scan maximum as a TS."))
    if len(spec.molecule.atoms) > 60 and spec.resources.nprocs < 8:
        issues.append(Issue("info",
            "Large system with few cores; consider increasing nprocs."))
    return issues
