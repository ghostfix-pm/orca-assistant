from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .lint import lint
from .parser import ParsedOutput, parse_file
from .render import render_input
from .resources import detect_hardware, suggest_resources
from .spec import CalcSpec, atoms_from_xyz


def _print_lint(spec: CalcSpec) -> None:
    for issue in lint(spec, detect_hardware()):
        print(f"[{issue.level}] {issue.message}", file=sys.stderr)


def print_summary(res: ParsedOutput) -> None:
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


def _emit(text: str, path: Path | None) -> None:
    if path:
        path.write_text(text)
        print(f"wrote {path}", file=sys.stderr)
    else:
        sys.stdout.write(text)


def _auto_resources(spec: CalcSpec) -> CalcSpec:
    res = suggest_resources(detect_hardware())
    print(f"[resources] nprocs={res.nprocs}, maxcore={res.maxcore_mb} MB per core (auto)",
          file=sys.stderr)
    return spec.model_copy(update={"resources": res})


def cmd_resources(args) -> int:
    hw = detect_hardware()
    sug = suggest_resources(hw)

    def gib(mb):
        return "n/a" if mb is None else f"{mb / 1024:.1f} GiB"

    print(f"cores (physical)    : {hw.cores}")
    print(f"RAM total           : {gib(hw.ram_mb)}")
    print(f"RAM available       : {gib(hw.available_mb)}")
    print(f"suggested           : nprocs={sug.nprocs}, maxcore={sug.maxcore_mb} MB per core")
    print(f"                      ({sug.nprocs} x {sug.maxcore_mb} MB = "
          f"{gib(sug.nprocs * sug.maxcore_mb)} in total)")
    return 0


def cmd_render(args) -> int:
    spec = CalcSpec.model_validate_json(args.spec.read_text())
    if args.auto_resources:
        spec = _auto_resources(spec)
    _print_lint(spec)
    _emit(render_input(spec), args.output)
    return 0


def cmd_ask(args) -> int:
    from .llm import make_backend, plan_calculation

    atoms = atoms_from_xyz(args.xyz.read_text())
    backend = make_backend(args.backend, args.model, args.base_url, args.api_key_env)
    planned = plan_calculation(args.request, atoms, backend=backend)
    for note in planned.assumptions:
        print(f"[assumption] {note}", file=sys.stderr)
    spec = _auto_resources(planned.spec) if args.auto_resources else planned.spec
    _print_lint(spec)
    if args.spec_out:
        args.spec_out.write_text(spec.model_dump_json(indent=2))
        print(f"wrote {args.spec_out}", file=sys.stderr)
    _emit(render_input(spec), args.output)
    return 0


def cmd_run(args) -> int:
    orca = shutil.which("orca")
    if not orca:
        print("ORCA not found in PATH.", file=sys.stderr)
        return 2
    inp = args.input.resolve()
    out = inp.with_suffix(".out")
    print(f"running {orca} {inp.name} -> {out.name}", file=sys.stderr)
    with open(out, "w") as fh:
        subprocess.run([orca, inp.name], cwd=inp.parent, stdout=fh, stderr=subprocess.STDOUT)
    res = parse_file(out)
    print_summary(res)
    return 0 if not res.problems else 1


def cmd_parse(args) -> int:
    res = parse_file(args.output)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
        return 0
    print_summary(res)
    return 0 if not res.problems else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="orca-assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="CalcSpec JSON -> .inp")
    r.add_argument("spec", type=Path)
    r.add_argument("-o", "--output", type=Path)
    r.add_argument("--auto-resources", action="store_true",
                   help="set nprocs/maxcore from this machine's cores and RAM")
    r.set_defaults(func=cmd_render)

    a = sub.add_parser("ask", help="plain-language request + .xyz -> .inp")
    a.add_argument("request", help='e.g. "optimise and compute frequencies, wB97X-D3, def2-TZVP, water"')
    a.add_argument("xyz", type=Path)
    a.add_argument("-o", "--output", type=Path)
    a.add_argument("--spec-out", type=Path, help="also save the validated spec as JSON")
    a.add_argument("--backend", choices=["ollama", "openai", "anthropic"], default="ollama",
                   help="ollama: local and free (default); openai: any OpenAI-compatible "
                        "server; anthropic: Claude API")
    a.add_argument("--model", help="model name (for ollama see `ollama list`)")
    a.add_argument("--base-url", help="server URL for --backend openai")
    a.add_argument("--api-key-env", help="env var holding the API key for --backend openai")
    a.add_argument("--auto-resources", action="store_true",
                   help="set nprocs/maxcore from this machine's cores and RAM")
    a.set_defaults(func=cmd_ask)

    rs = sub.add_parser("resources", help="show detected cores/RAM and suggested nprocs/maxcore")
    rs.set_defaults(func=cmd_resources)

    x = sub.add_parser("run", help="run ORCA on an .inp, then parse the result")
    x.add_argument("input", type=Path)
    x.set_defaults(func=cmd_run)

    pa = sub.add_parser("parse", help="summarise an ORCA .out file")
    pa.add_argument("output", type=Path)
    pa.add_argument("--json", action="store_true")
    pa.set_defaults(func=cmd_parse)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def parse_main(argv: list[str] | None = None) -> int:
    """Entry point kept for the `orca-parse` command."""
    return main(["parse", *(argv if argv is not None else sys.argv[1:])])


if __name__ == "__main__":
    raise SystemExit(main())
