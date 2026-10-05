# quantization-interpretability

Where, what and how NVFP4 and FP8 weight quantization damage Gemma 3 4B, measured with interventions and
Gemma Scope 2 sparse autoencoders, ending in a feature-guided mixed-precision scheme.

- Plan: https://claude.ai/code/artifact/31a1dbec-e0d1-4bc1-bff1-ef66e7f07387
- Operational brief for Claude Code sessions: `CLAUDE.md`
- Numbers: `results/LEDGER.md` (script-written only)

```
uv sync --extra dev            # laptop: quantization + metrics code and tests
uv sync --extra dev --extra box  # GPU box: adds vLLM, llm-compressor, sae-lens, lm-eval
uv run pytest -q
```
