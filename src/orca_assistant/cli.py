from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .lint import lint
from .render import render_input
from .spec import CalcSpec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orca-assistant")
    p.add_argument("spec", type=Path, help="JSON file with a CalcSpec")
    p.add_argument("-o", "--output", type=Path, help="write .inp here (default: stdout)")
    args = p.parse_args(argv)

    spec = CalcSpec.model_validate_json(args.spec.read_text())
    for issue in lint(spec):
        print(f"[{issue.level}] {issue.message}", file=sys.stderr)

    text = render_input(spec)
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def parse_main(argv: list[str] | None = None) -> int:
    import json

    from .parser import parse_file

    p = argparse.ArgumentParser(prog="orca-parse")
    p.add_argument("output", type=Path, help="ORCA .out file")
    p.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    args = p.parse_args(argv)

    res = parse_file(args.output)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        return 0

    def fmt(v, unit=""):
        return "n/a" if v is None else f"{v:.6f}{unit}"

    print(f"terminated normally : {res.terminated_normally}")
    print(f"optimisation        : {res.opt_converged}")
    print(f"final energy        : {fmt(res.final_energy_eh, ' Eh')}")
    if res.frequencies_cm1:
        print(f"frequencies (cm-1)  : {', '.join(f'{f:.1f}' for f in res.frequencies_cm1)}")
        print(f"imaginary           : {res.n_imaginary}")
    print(f"ZPE                 : {fmt(res.zpe_eh, ' Eh')}")
    print(f"enthalpy            : {fmt(res.enthalpy_eh, ' Eh')}")
    print(f"Gibbs free energy   : {fmt(res.gibbs_eh, ' Eh')}")
    if res.runtime_s is not None:
        print(f"run time            : {res.runtime_s:.1f} s")
    for w in res.warnings:
        print(f"[warning] {w}")
    for pr in res.problems:
        print(f"[PROBLEM] {pr}")
    return 0 if not res.problems else 1
