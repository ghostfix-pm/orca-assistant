"""Natural-language request -> validated CalcSpec, with pluggable LLM backends.

The model never sees or writes coordinates. It returns a structured request;
the result is validated with pydantic, merged with the atoms from the user's XYZ
file, and validation errors are fed back for a retry. Because every answer is
validated, small local models are usable: they may produce valid JSON with wrong
content, and the validator plus retry loop catches part of that.

Backends:
  ollama     local, free, offline (OpenAI-compatible endpoint of Ollama)
  openai     any OpenAI-compatible server (llama.cpp, LM Studio, vLLM, Groq, ...)
  anthropic  Claude API (paid, needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from .spec import (ATOMIC_NUMBER, Atom, CalcSpec, Method, Molecule, Resources,
                   ScanSpec, Task)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5"
TOOL_NAME = "submit_calculation_request"

SYSTEM_PROMPT = """You translate a chemist's plain-language request into a structured \
ORCA calculation request.

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
- If the request is ambiguous, pick the most conventional reading and say so in assumptions.

Example of a valid answer:
{"task": "opt_freq", "method": {"functional": "wB97X-D3", "basis": "def2-TZVP", \
"dispersion": null, "solvent": "water"}, "resources": {"nprocs": 4, "maxcore_mb": 2000}, \
"scan": null, "tight_scf": true, "charge": 0, "multiplicity": 1, \
"assumptions": ["cores not given: used 4"]}"""


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


class BadModelOutput(Exception):
    """The model answered, but not with usable JSON. `raw` is what it said."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


class Backend(Protocol):
    def complete(self, system: str, messages: list[dict], schema: dict) -> dict:
        """Return the model's answer as a dict (not yet validated)."""


class AnthropicBackend:
    def __init__(self, model: str | None = None, client: Any = None):
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise RuntimeError('pip install -e ".[llm]" to use the anthropic backend') from e
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: list[dict], schema: dict) -> dict:
        tool = {"name": TOOL_NAME,
                "description": "Submit the structured ORCA calculation request.",
                "input_schema": schema}
        resp = self._get_client().messages.create(
            model=self.model, max_tokens=1500, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME}, messages=messages,
        )
        block = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        if block is None:
            raise BadModelOutput("model returned no tool call")
        return dict(block.input)


class OpenAICompatBackend:
    """Any server speaking POST {base_url}/chat/completions (Ollama, llama.cpp, ...).

    Uses response_format=json_schema so the server constrains the output to the
    schema where supported, and also states the schema in the prompt.
    """

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, system: str, messages: list[dict], schema: dict) -> dict:
        system = (f"{system}\n\nRespond with ONLY a JSON object matching this JSON Schema "
                  f"(no prose, no code fences):\n{json.dumps(schema)}")
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "calculation_request", "schema": schema}},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.load(r)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise RuntimeError(f"HTTP {e.code} from {url}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach {url} ({e.reason}). Is the server running? "
                "For Ollama: `ollama serve`."
            ) from e

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise BadModelOutput("unexpected response shape", raw=str(payload)[:500]) from e
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (content or "").strip())
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise BadModelOutput(f"not valid JSON: {e}", raw=content or "") from e
        if not isinstance(data, dict):
            raise BadModelOutput("JSON answer is not an object", raw=content)
        return data


def make_backend(name: str = "ollama", model: str | None = None,
                 base_url: str | None = None, api_key_env: str | None = None) -> Backend:
    model = model or os.environ.get("ORCA_ASSISTANT_MODEL")
    if name == "anthropic":
        return AnthropicBackend(model=model)
    if name == "ollama":
        if not model:
            raise RuntimeError("Pass --model (names are listed by `ollama list`) "
                               "or set ORCA_ASSISTANT_MODEL.")
        return OpenAICompatBackend(base_url or "http://localhost:11434/v1", model,
                                   api_key="ollama")
    if name == "openai":
        if not (model and base_url):
            raise RuntimeError("--backend openai needs --base-url and --model.")
        key = None
        if api_key_env:
            key = os.environ.get(api_key_env)
            if not key:
                raise RuntimeError(f"Set the {api_key_env} environment variable.")
        return OpenAICompatBackend(base_url, model, api_key=key)
    raise RuntimeError(f"unknown backend: {name!r}")


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


def plan_calculation(text: str, atoms: list[Atom], backend: Backend | None = None,
                     max_retries: int = 2) -> PlannedCalc:
    """Turn a plain-language request plus atoms into a validated CalcSpec."""
    backend = backend or make_backend()
    schema = RequestSpec.model_json_schema()
    messages: list[dict] = [
        {"role": "user", "content": f"{describe_molecule(atoms)}\n\nRequest: {text}"}
    ]

    last_error = "no attempt made"
    for _ in range(max_retries + 1):
        try:
            raw = backend.complete(SYSTEM_PROMPT, messages, schema)
            req = RequestSpec.model_validate(raw)
            spec = CalcSpec(
                molecule=Molecule(atoms=atoms, charge=req.charge,
                                  multiplicity=req.multiplicity),
                task=req.task, method=req.method, resources=req.resources,
                scan=req.scan, tight_scf=req.tight_scf,
            )
            return PlannedCalc(spec=spec, assumptions=req.assumptions)
        except BadModelOutput as e:
            last_error = str(e)
            previous = e.raw or "(no usable output)"
        except ValidationError as e:
            last_error = str(e)
            previous = json.dumps(raw)
        messages.append({"role": "assistant", "content": previous})
        messages.append({"role": "user", "content":
                         f"That answer was rejected:\n{last_error}\n"
                         "Fix it and answer again with the complete JSON."})
    raise RuntimeError(f"Could not produce a valid spec after retries:\n{last_error}")
