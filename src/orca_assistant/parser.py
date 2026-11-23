"""Parse ORCA output files into a structured result.

Patterns were written against ORCA 5/6 style output. Every field is optional:
if a pattern does not match, the value is None/empty instead of a wrong guess.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

_FLOAT = r"(-?\d+\.\d+)"

_RE_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+" + _FLOAT)
_RE_FREQ = re.compile(r"^\s*\d+:\s+" + _FLOAT + r"\s+cm\*\*-1")
_RE_ZPE = re.compile(r"Zero point energy\s+\.\.\.\s+" + _FLOAT + r"\s+Eh")
_RE_ENTHALPY = re.compile(r"Total [Ee]nthalpy\s+\.\.\.\s+" + _FLOAT + r"\s+Eh")
_RE_GIBBS = re.compile(r"Final Gibbs free energy\s+\.\.\.\s+" + _FLOAT + r"\s+Eh")
_RE_RUNTIME = re.compile(
    r"TOTAL RUN TIME:\s+(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec"
)

IMAG_THRESHOLD_CM1 = -1.0  # frequencies below this count as imaginary


@dataclass
class ParsedOutput:
    terminated_normally: bool = False
    opt_converged: bool | None = None  # None: no optimisation found
    final_energy_eh: float | None = None
    frequencies_cm1: list[float] = field(default_factory=list)  # vibrations only
    n_imaginary: int = 0
    zpe_eh: float | None = None
    enthalpy_eh: float | None = None
    gibbs_eh: float | None = None
    warnings: list[str] = field(default_factory=list)
    runtime_s: float | None = None

    @property
    def problems(self) -> list[str]:
        out: list[str] = []
        if not self.terminated_normally:
            out.append("ORCA did not terminate normally")
        if self.opt_converged is False:
            out.append("geometry optimisation did not converge")
        if self.n_imaginary:
            out.append(
                f"{self.n_imaginary} imaginary frequenc"
                f"{'y' if self.n_imaginary == 1 else 'ies'} "
                "(expected 0 for a minimum, 1 for a transition state)"
            )
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        d["problems"] = self.problems
        return d


def _last_float(pattern: re.Pattern[str], text: str) -> float | None:
    matches = pattern.findall(text)
    return float(matches[-1]) if matches else None


def _parse_frequencies(lines: list[str]) -> list[float]:
    """Vibrational frequencies from the last VIBRATIONAL FREQUENCIES block."""
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "VIBRATIONAL FREQUENCIES"]
    if not starts:
        return []
    freqs: list[float] = []
    for ln in lines[starts[-1] + 1:]:
        m = _RE_FREQ.match(ln)
        if m:
            freqs.append(float(m.group(1)))
        elif freqs and not ln.strip():
            break  # blank line after the list ends the block
    # translations/rotations are printed as 0.00 cm**-1
    return [f for f in freqs if abs(f) > 1e-6]


def _parse_warnings(lines: list[str]) -> list[str]:
    """Collect 'WARNING...' messages: the line itself plus indented continuation."""
    found: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("WARNING"):
            block = [lines[i].strip()]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith("="):
                block.append(lines[i].strip())
                i += 1
            msg = " ".join(block)
            if msg not in found:
                found.append(msg)
        else:
            i += 1
    return found


def parse_output(text: str) -> ParsedOutput:
    lines = text.splitlines()
    res = ParsedOutput()

    res.terminated_normally = "ORCA TERMINATED NORMALLY" in text

    if "THE OPTIMIZATION HAS CONVERGED" in text:
        res.opt_converged = True
    elif "did not converge" in text.lower() and "optimization" in text.lower():
        res.opt_converged = False

    res.final_energy_eh = _last_float(_RE_ENERGY, text)
    res.frequencies_cm1 = _parse_frequencies(lines)
    res.n_imaginary = sum(f < IMAG_THRESHOLD_CM1 for f in res.frequencies_cm1)
    res.zpe_eh = _last_float(_RE_ZPE, text)
    res.enthalpy_eh = _last_float(_RE_ENTHALPY, text)
    res.gibbs_eh = _last_float(_RE_GIBBS, text)
    res.warnings = _parse_warnings(lines)

    m = _RE_RUNTIME.search(text)
    if m:
        d, h, mi, s, ms = map(int, m.groups())
        res.runtime_s = d * 86400 + h * 3600 + mi * 60 + s + ms / 1000

    return res


def parse_file(path: str | Path) -> ParsedOutput:
    return parse_output(Path(path).read_text(errors="replace"))
