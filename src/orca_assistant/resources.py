"""Detect machine resources and suggest ORCA %pal / %maxcore settings.

Standard library only. ORCA's %maxcore is a soft limit per process and real
usage can exceed it, so suggestions keep headroom.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .spec import Resources

HEADROOM = 0.75        # fraction of the memory budget handed to %maxcore
MIN_MAXCORE_MB = 900   # below this, use fewer processes instead
MAX_MAXCORE_MB = 8000


@dataclass(frozen=True)
class Hardware:
    cores: int                   # physical cores when detectable, else logical
    ram_mb: int | None           # total RAM
    available_mb: int | None     # currently available RAM (Linux only)


def parse_meminfo(text: str) -> tuple[int | None, int | None]:
    """Parse /proc/meminfo -> (total_mb, available_mb)."""
    vals: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            vals[key.strip()] = int(parts[0])  # kB
    total, avail = vals.get("MemTotal"), vals.get("MemAvailable")
    return (total // 1024 if total else None, avail // 1024 if avail else None)


def parse_physical_cores(text: str) -> int | None:
    """Count unique (physical id, core id) pairs in /proc/cpuinfo."""
    pairs: set[tuple[str, str]] = set()
    phys = core = None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if core is not None:
                pairs.add((phys or "0", core))
            phys = core = None
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "physical id":
            phys = val
        elif key == "core id":
            core = val
    return len(pairs) or None


def _logical_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _windows_ram_mb() -> int | None:
    try:
        import ctypes

        class MEMSTATUS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMSTATUS()
        st.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))  # type: ignore[attr-defined]
        return int(st.ullTotalPhys // (1024 * 1024))
    except Exception:
        return None


def detect_hardware() -> Hardware:
    logical = _logical_cores()
    cores = logical
    ram = avail = None
    meminfo, cpuinfo = Path("/proc/meminfo"), Path("/proc/cpuinfo")
    try:
        if meminfo.exists():
            ram, avail = parse_meminfo(meminfo.read_text())
        if cpuinfo.exists():
            phys = parse_physical_cores(cpuinfo.read_text())
            if phys:
                cores = min(phys, logical)
    except OSError:
        pass
    if ram is None:
        try:  # macOS and other POSIX
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1024 * 1024)
        except (ValueError, OSError, AttributeError):
            ram = _windows_ram_mb()
    return Hardware(cores=cores, ram_mb=ram, available_mb=avail)


def suggest_resources(hw: Hardware) -> Resources:
    """nprocs and maxcore (MB per core) that fit this machine."""
    if hw.ram_mb is None and hw.available_mb is None:
        return Resources(nprocs=min(hw.cores, 4), maxcore_mb=2000)
    budget = hw.available_mb if hw.available_mb is not None else int(hw.ram_mb * 0.8)
    usable = int(budget * HEADROOM)
    nprocs = max(hw.cores, 1)
    while nprocs > 1 and usable // nprocs < MIN_MAXCORE_MB:
        nprocs -= 1
    maxcore = min(max(usable // nprocs, 200), MAX_MAXCORE_MB)
    return Resources(nprocs=nprocs, maxcore_mb=maxcore)
