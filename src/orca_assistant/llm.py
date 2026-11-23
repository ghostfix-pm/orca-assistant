"""Natural-language request -> validated CalcSpec.

The model never sees or writes coordinates. It fills a structured request via
forced tool use; the result is validated with pydantic, merged with the atoms
from the user's XYZ file, and validation errors are fed back for a retry.
"""
from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .spec import (ATOMIC_NUMBER, Atom, CalcSpec, Method, Molecule, Resources,
                   ScanSpec, Task)

DEFAULT_MODEL = "claude-sonnet-5-5"  # override with ORCA_ASSISTANT_MODEL or --model
TOOL_NAME = "submit_calculation_request"

SYSTEM_PROMPT = """You translate a chemist's plain-language request into a structured \
ORCA calculation request by calling the provided tool.

Rules:
- Use only what the user specified. For anything unspecified keep the schema default \
and add a short note to `assumptions` (e.g. "no basis given: used def2-SVP").
- Never invent or output coordinates; geometry is supplied separately.
- Scan atom indices are 0-based; use the numbered atom list given in the request.
- Functional and basis names must be valid ORCA keywords (B3LYP, PBE0, wB97X-D3, \
r2SCAN-3c, def2-TZVP, ...).
- If dispersion is part of the functional name (wB97X-D3, wB97X-D4), set dispersion to null.
- Composite -3c methods (r2SCAN-3c, B97-3c): basis and dispersion must be null.
- Solvent is a lowercase CPCM solvent name (water, methanol, ...) or null for gas phase.
- Charge and multiplicity: take them from the request if stated. Otherwise assume neutral \
and the lowest multiplicity consistent with the electron count, and note it in assumptions.
- Do not choose a more expensive method or larger basis than the user asked for.
- If the request is ambiguous, pick the most conventional reading and say so in assumptions."""


class RequestSpec(BaseModel):
    """CalcSpec minus coordinates, plus the model's stated assumptions."""
    task: Task = Task.OPT_FREQ
    method: Method = Field(default_factory=Method)
    resources: Resources = Field(default_factory=Resources)
    scan: ScanSpec | None = None
    tight_scf: bool = True
    charge: int = 0
    multiplicity: int = Field(1, ge=1)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Every choice made because the user did not specify it.",
    )


@dataclass
class PlannedCalc:
    spec: CalcSpec
    assumptions: list[str]


def describe_molecule(atoms: list[Atom]) -> str:
    counts = Counter(a.element for a in atoms)
    formula = "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(counts.items()))
    n_el = sum(ATOMIC_NUMBER[a.element] for a in atoms)
    if len(atoms) <= 60:
        listing = ", ".join(f"{i}:{a.element}" for i, a in enumerate(atoms))
    else:
        listing = "(too many atoms to list)"
    return (f"Molecule: {formula}; {n_el} electrons at charge 0; "
            f"atoms (0-based index:element): {listing}")


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            'The LLM layer needs the anthropic package: pip install -e ".[llm]"'
        ) from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
    return anthropic.Anthropic()


def plan_calculation(
    text: str,
    atoms: list[Atom],
    client: Any = None,
    model: str | None = None,
    max_retries: int = 2,
) -> PlannedCalc:
    """Turn a plain-language request plus atoms into a validated CalcSpec."""
    client = client or _default_client()
    model = model or os.environ.get("ORCA_ASSISTANT_MODEL", DEFAULT_MODEL)
    tool = {
        "name": TOOL_NAME,
        "description": "Submit the structured ORCA calculation request.",
        "input_schema": RequestSpec.model_json_schema(),
    }
    messages: list[dict] = [
        {"role": "user", "content": f"{describe_molecule(atoms)}\n\nRequest: {text}"}
    ]

    last_error = "no attempt made"
    for _ in range(max_retries + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=messages,
        )
        block = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None:
            raise RuntimeError("Model did not return a tool call.")
        try:
            req = RequestSpec.model_validate(block.input)
            spec = CalcSpec(
                molecule=Molecule(atoms=atoms, charge=req.charge,
                                  multiplicity=req.multiplicity),
                task=req.task, method=req.method, resources=req.resources,
                scan=req.scan, tight_scf=req.tight_scf,
            )
            return PlannedCalc(spec=spec, assumptions=req.assumptions)
        except ValidationError as e:
            last_error = str(e)
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                "content": f"Validation failed, fix and resubmit:\n{last_error}",
            }]})
    raise RuntimeError(f"Could not produce a valid spec after retries:\n{last_error}")
