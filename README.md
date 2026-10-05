# quantization-interpretability

Where, what and how NVFP4 and FP8 weight quantization damage Gemma 3 4B, measured with interventions and
Gemma Scope 2 sparse autoencoders, ending in a feature-guided mixed-precision scheme.

- Plan: https://claude.ai/code/artifact/31a1dbec-e0d1-4bc1-bff1-ef66e7f07387
- Operational brief for Claude Code sessions: `CLAUDE.md`
- Numbers: `results/LEDGER.md` (script-written only)
- Write-up with figures: `docs/REPORT.md`; running log of findings: `docs/FINDINGS.md`; decisions: `docs/DECISIONS.md`

## Headline results (sprint of 2026-10-05, paired bootstrap 95% CIs)

| Question | Answer |
| --- | --- |
| How much damage? | NVFP4 round-to-nearest costs 7.1 GSM8K points [4.7, 9.4] and 1.3 MMLU [0.8, 1.8]; FP8 per-channel costs nothing measurable; the fp32 noise floor is +0.1 |
| Which layers? | None in particular: restoring the 8 most damaging layers recovers 41% [9%, 74%] of the GSM8K loss, a random 8 recover 48% |
| Which tokens? | Where the model is uncertain: the highest-entropy fifth of positions flips 2.4x the average rate; the first 256 tokens of a document carry 3.9x the median KL |
| Which SAE features? | Not the damaged ones: restoring the 10% least-surviving features recovers nothing; the 10% highest-energy features recover 66% of the gap, the whole residual 106% |
| Can it be fixed? | A mean-shift bias at layer 17 recovers +2.6 [+0.2, +5.2] points at no cost; 2% of weights kept in bf16 by magnitude (92 MB) recovers 48%, as much as 8 whole layers at 1.1 GB; SAE-guided channel selection ties with magnitude |

![budget sweep](results/figures/f4_budget_sweep.png)

The quantizer is bit-exact against llm-compressor's NVFP4A16 export on all 238 matrices. Every analysis script
writes its numbers and CIs into `results/LEDGER.md`; `scripts/figures/make_all.py` regenerates the figures from
results JSON; `webui/` is a local viewer for the raw generations and per-token KL (`cd webui && uv run server.py`).

```
uv sync --extra dev            # laptop: quantization + metrics code and tests
uv sync --extra dev --extra box  # GPU box: adds vLLM, llm-compressor, sae-lens, lm-eval
uv run pytest -q
```
