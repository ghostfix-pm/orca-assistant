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
orca-assistant examples/water_opt_freq.json -o water.inp
pytest
```

## Roadmap
1. [x] Spec + renderer + linter (this MVP)
2. [x] Output parser: `orca-parse file.out` (convergence, imaginary freqs, energy, thermochemistry, warnings)
3. [ ] SLURM script generator
4. [ ] LLM layer: text -> CalcSpec (structured output) + RAG over ORCA manual
5. [ ] Error diagnosis and guarded auto-restart
6. [ ] Benchmark set of reference inputs
7. [ ] MCP server exposing generate / lint / parse tools
