# orca-assistant

Typed spec -> validated ORCA input. The LLM layer (planned) only produces a
`CalcSpec`; all ORCA syntax comes from deterministic code.

```
natural language --(LLM)--> CalcSpec --(validate)--> render_input --> .inp
                                          |
                                        lint()
```

## Quick start
```bash
pip install -e ".[dev]"
pytest

orca-assistant render examples/water_opt_freq.json -o water.inp   # spec -> .inp
orca-assistant run water.inp                                       # run ORCA + summary
orca-assistant parse water.out [--json]                            # summarise any .out
```

## Plain-language requests (LLM layer)
```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=...           # your own key
orca-assistant ask "optimise and compute frequencies, wB97X-D3, def2-TZVP, in water" mol.xyz \
    -o job.inp --spec-out job.json
```
The model never sees or writes coordinates: it fills a structured request
(task, method, resources, charge, multiplicity). The result is validated, merged
with the atoms from your XYZ file, and validation errors are sent back to the
model for a retry. Every default it chose is printed as an `[assumption]`, and
`--spec-out` stores the exact spec for reproducibility.

## Roadmap
1. [x] Spec + renderer + linter (this MVP)
2. [x] Output parser: `orca-parse file.out` (convergence, imaginary freqs, energy, thermochemistry, warnings)
3. [x] `run` command: execute ORCA, parse, report
4. [x] LLM layer: text -> CalcSpec (forced tool use + validation retries)
4b. [ ] RAG over the ORCA manual, benchmark of request -> input pairs
4c. [ ] SLURM script generator
5. [ ] Error diagnosis and guarded auto-restart
6. [ ] Benchmark set of reference inputs
7. [ ] MCP server exposing generate / lint / parse tools
