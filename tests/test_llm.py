import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from orca_assistant import plan_calculation
from orca_assistant.llm import (TOOL_NAME, AnthropicBackend, BadModelOutput,
                                OpenAICompatBackend, describe_molecule, make_backend)
from orca_assistant.spec import atoms_from_xyz

WATER = atoms_from_xyz("O 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469")


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, system, messages, schema):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_plan_happy_path():
    backend = FakeBackend([{
        "task": "opt_freq",
        "method": {"functional": "wB97X-D3", "basis": "def2-TZVP",
                   "dispersion": None, "solvent": "water"},
        "resources": {"nprocs": 8, "maxcore_mb": 3000},
        "assumptions": ["no cores given: used 8"],
    }])
    planned = plan_calculation("opt+freq wB97X-D3 def2-TZVP water", WATER, backend=backend)
    assert planned.spec.method.functional == "wB97X-D3"
    assert planned.spec.molecule.atoms == WATER
    assert planned.assumptions == ["no cores given: used 8"]
    first = backend.calls[0]["messages"][0]["content"]
    assert "H2O" in first and "0.757" not in first  # molecule described, no coordinates


def test_retry_after_validation_error():
    backend = FakeBackend([{"task": "sp", "multiplicity": 2},   # water cannot be a doublet
                           {"task": "sp", "multiplicity": 1}])
    planned = plan_calculation("single point", WATER, backend=backend)
    assert planned.spec.molecule.multiplicity == 1
    feedback = backend.calls[1]["messages"][-1]["content"]
    assert "multiplicity" in feedback and "rejected" in feedback


def test_retry_after_unusable_output():
    backend = FakeBackend([BadModelOutput("not valid JSON", raw="Sure! Here you go"),
                           {"task": "sp"}])
    planned = plan_calculation("single point", WATER, backend=backend)
    assert planned.spec.task.value == "sp"
    msgs = backend.calls[1]["messages"]
    assert msgs[1] == {"role": "assistant", "content": "Sure! Here you go"}


def test_gives_up_after_retries():
    backend = FakeBackend([{"task": "sp", "multiplicity": 2}] * 3)
    with pytest.raises(RuntimeError):
        plan_calculation("single point", WATER, backend=backend, max_retries=2)


def test_describe_molecule():
    d = describe_molecule(WATER)
    assert "H2O" in d and "10 electrons" in d and "0:O" in d


# --- Anthropic backend (fake client) --------------------------------------
def test_anthropic_backend_forces_tool_use():
    class Client:
        def __init__(self):
            self.messages = self
            self.kw = None

        def create(self, **kw):
            self.kw = kw
            return SimpleNamespace(content=[SimpleNamespace(
                type="tool_use", id="t", name=TOOL_NAME, input={"task": "sp"})])

    client = Client()
    out = AnthropicBackend(client=client).complete("sys", [{"role": "user", "content": "x"}], {"type": "object"})
    assert out == {"task": "sp"}
    assert client.kw["tool_choice"] == {"type": "tool", "name": TOOL_NAME}


# --- OpenAI-compatible backend against a real local HTTP server ------------
class _Handler(BaseHTTPRequestHandler):
    captured: dict = {}

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Handler.captured.update(path=self.path, body=json.loads(self.rfile.read(n)),
                                 auth=self.headers.get("Authorization"))
        answer = json.dumps({"task": "sp", "assumptions": ["demo"]})
        reply = json.dumps({"choices": [{"message": {"content": f"```json\n{answer}\n```"}}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(reply.encode())

    def log_message(self, *args):
        pass


def test_openai_compat_backend_roundtrip():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = OpenAICompatBackend(f"http://127.0.0.1:{server.server_port}/v1",
                                      "some-model", api_key="ollama")
        planned = plan_calculation("single point", WATER, backend=backend)
    finally:
        server.shutdown()
    sent = _Handler.captured
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == "Bearer ollama"
    body = sent["body"]
    assert body["model"] == "some-model" and body["temperature"] == 0
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"][0]["role"] == "system" and "JSON Schema" in body["messages"][0]["content"]
    assert planned.assumptions == ["demo"]   # code fences were stripped


def test_unreachable_server_message():
    backend = OpenAICompatBackend("http://127.0.0.1:1/v1", "m", timeout=3)
    with pytest.raises(RuntimeError, match="Cannot reach"):
        backend.complete("s", [{"role": "user", "content": "x"}], {})


def test_make_backend_requirements(monkeypatch):
    monkeypatch.delenv("ORCA_ASSISTANT_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="--model"):
        make_backend("ollama")
    with pytest.raises(RuntimeError, match="--base-url"):
        make_backend("openai", model="m")
    assert isinstance(make_backend("ollama", model="m"), OpenAICompatBackend)
