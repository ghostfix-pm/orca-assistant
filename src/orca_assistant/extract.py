"""Extra data from ORCA output: version, geometries, frontier orbitals,
atomic charges, optimisation steps, IR intensities. Standard library only."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

Atom = tuple[str, float, float, float]

_VERSION = re.compile(r"Program Version\s+(\S+)")
_GEOM_HDR = re.compile(r"CARTESIAN COORDINATES \(ANGSTROEM\)[ \t]*\n-+[ \t]*\n")
_GEOM_ROW = re.compile(r"^\s*([A-Z][a-z]?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_ORB_ROW = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$")
_CHG_HDR = {"mulliken": re.compile(r"^MULLIKEN ATOMIC CHARGES[^\n]*\n", re.M),
            "loewdin": re.compile(r"^LOEWDIN ATOMIC CHARGES[^\n]*\n", re.M)}
_CHG_ROW = re.compile(r"^\s*(\d+)\s+([A-Z][a-z]?)\s*:\s+(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?\s*$")
_CYCLE = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)")
_FSPE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
_IR_ROW = re.compile(r"^\s*(\d+):\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+\d+\.\d+\s+\(")


@dataclass
class FrontierOrbitals:
    channel: str                  # "restricted", "alpha" or "beta"
    homo_index: int | None
    homo_ev: float | None
    lumo_index: int | None
    lumo_ev: float | None
    gap_ev: float | None


@dataclass
class AtomCharge:
    index: int                    # 0-based, as printed by ORCA
    element: str
    charge: float
    spin: float | None = None


@dataclass
class OptStep:
    job: int                      # 1, 2, ... for consecutive optimisations (scan points, ...)
    cycle: int
    energy_eh: float | None


@dataclass
class IRMode:
    mode: int
    freq_cm1: float
    intensity_km_mol: float


@dataclass
class Extras:
    orca_version: str | None = None
    n_atoms: int | None = None
    n_geometries: int = 0
    frontier: list[FrontierOrbitals] = field(default_factory=list)
    mulliken: list[AtomCharge] = field(default_factory=list)
    loewdin: list[AtomCharge] = field(default_factory=list)
    opt_steps: list[OptStep] = field(default_factory=list)
    ir_modes: list[IRMode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_version(text: str) -> str | None:
    m = _VERSION.search(text)
    return m.group(1) if m else None


def parse_geometries(text: str) -> list[list[Atom]]:
    """Every CARTESIAN COORDINATES (ANGSTROEM) block, in file order."""
    out: list[list[Atom]] = []
    for h in _GEOM_HDR.finditer(text):
        atoms: list[Atom] = []
        for line in text[h.end():].splitlines():
            m = _GEOM_ROW.match(line)
            if not m:
                break
            atoms.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
        if atoms:
            out.append(atoms)
    return out


def write_xyz(atoms: list[Atom], comment: str = "") -> str:
    lines = [str(len(atoms)), comment]
    lines += [f"{el:<2} {x:14.8f} {y:14.8f} {z:14.8f}" for el, x, y, z in atoms]
    return "\n".join(lines) + "\n"


def _frontier(channel: str, rows: list[tuple[int, float, float]]) -> FrontierOrbitals:
    occ = [r for r in rows if r[1] >= 0.5]
    virt = [r for r in rows if r[1] < 0.5]
    homo = max(occ, key=lambda r: r[2]) if occ else None
    lumo = min(virt, key=lambda r: r[2]) if virt else None
    gap = lumo[2] - homo[2] if homo and lumo else None
    return FrontierOrbitals(channel, homo[0] if homo else None, homo[2] if homo else None,
                            lumo[0] if lumo else None, lumo[2] if lumo else None, gap)


def parse_frontier(text: str) -> list[FrontierOrbitals]:
    """HOMO/LUMO (eV) from the last ORBITAL ENERGIES block; one entry per spin channel."""
    idx = text.rfind("ORBITAL ENERGIES")
    if idx < 0:
        return []
    channels: list[list[tuple[int, float, float]]] = [[]]
    for line in text[idx:].splitlines()[1:]:
        s = line.strip()
        m = _ORB_ROW.match(line)
        if m:
            channels[-1].append((int(m.group(1)), float(m.group(2)), float(m.group(4))))
        elif "SPIN DOWN" in s:
            channels.append([])
        elif not s or s.startswith("-") or s.startswith("NO") or "SPIN UP" in s:
            continue
        else:
            break
    channels = [c for c in channels if c]
    if len(channels) == 2:
        return [_frontier("alpha", channels[0]), _frontier("beta", channels[1])]
    return [_frontier("restricted", channels[0])] if channels else []


def parse_charges(text: str, kind: str) -> list[AtomCharge]:
    """Atomic charges ('mulliken' or 'loewdin') from the last such block."""
    hits = list(_CHG_HDR[kind].finditer(text))
    if not hits:
        return []
    rows: list[AtomCharge] = []
    for line in text[hits[-1].end():].splitlines():
        m = _CHG_ROW.match(line)
        if m:
            spin = float(m.group(4)) if m.group(4) else None
            rows.append(AtomCharge(int(m.group(1)), m.group(2), float(m.group(3)), spin))
        elif rows:
            break
    return rows


def parse_opt_steps(text: str) -> list[OptStep]:
    """Energy at every geometry optimisation cycle. `job` increases whenever the
    cycle counter restarts (scan points, consecutive optimisations)."""
    heads = list(_CYCLE.finditer(text))
    steps: list[OptStep] = []
    job, prev = 0, None
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        e = _FSPE.search(text, h.end(), end)
        cyc = int(h.group(1))
        if prev is None or cyc <= prev:
            job += 1
        prev = cyc
        steps.append(OptStep(job, cyc, float(e.group(1)) if e else None))
    return steps


def parse_ir(text: str) -> list[IRMode]:
    idx = text.rfind("IR SPECTRUM")
    if idx < 0:
        return []
    modes: list[IRMode] = []
    for line in text[idx:].splitlines()[1:]:
        m = _IR_ROW.match(line)
        if m:
            modes.append(IRMode(int(m.group(1)), float(m.group(2)), float(m.group(4))))
        elif modes:
            break
    return modes


def extract(text: str) -> Extras:
    geoms = parse_geometries(text)
    return Extras(
        orca_version=parse_version(text),
        n_atoms=len(geoms[-1]) if geoms else None,
        n_geometries=len(geoms),
        frontier=parse_frontier(text),
        mulliken=parse_charges(text, "mulliken"),
        loewdin=parse_charges(text, "loewdin"),
        opt_steps=parse_opt_steps(text),
        ir_modes=parse_ir(text),
    )
