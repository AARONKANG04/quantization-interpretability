# Number ledger

Written only by scripts. Every row carries its control in the same table. Definitions: plan doc section 8.

| Placeholder | Value | 95% CI | Control | Source file | Updated |
| --- | --- | --- | --- | --- | --- |
| [X]% of loss in [N] layers | | | random ordering | results/phase1/restore_curve.json | |
| concentrated on [token class] | | | C1, C2 | results/phase1/token_buckets.parquet | |
| [N]% of features ([category]) carry [X]% of shift | | | C2 curve, random-feature categories, error-term share | results/phase2/shift_share.json | |
| steering recovered [X] of [Y] points | | | mean-shift bias, random-feature gain, bf16 specificity | results/phase2/steering.json | |
| [Z]% of weights in bf16 | | | RAND, MAG, KLG | results/phase3/budget_sweep.json | |
| recovering [X]% of FP4 loss | | | RAND, MAG, KLG | results/phase3/budget_sweep.json | |
| [M]% of memory of full-layer fallback | | | LAYER curve | results/phase3/memory.json | |

## GPU-hour tally

| Phase | H100-hours | RTX PRO 6000-hours | Notes |
| --- | --- | --- | --- |

## Damage table (ref bf16, paired bootstrap, 2026-10-05T03:29:08)

| Condition | Task | Accuracy | Delta vs ref | 95% CI | Flips c->i / i->c |
| --- | --- | --- | --- | --- | --- |
| bf16 | gsm8k_cot | 0.4049 | | | |
| bf16 | mmlu | 0.5962 | | | |
| bf16_arc_hs | arc_challenge | 0.5452 | | | |
| bf16_arc_hs | hellaswag | 0.7579 | | | |
| c1_fp32 | gsm8k_cot | 0.4056 | +0.0008 | [-0.0136, +0.0152] | 0.034 / 0.035 |
| q1 | gsm8k_cot | 0.3343 | -0.0705 | [-0.0940, -0.0470] | 0.134 / 0.064 |
| q1 | mmlu | 0.5833 | -0.0129 | [-0.0180, -0.0077] | 0.055 / 0.042 |
| q1_arc_hs | arc_challenge | 0.5256 | -0.0196 | [-0.0333, -0.0060] | 0.038 / 0.018 |
| q1_arc_hs | hellaswag | 0.7512 | -0.0067 | [-0.0104, -0.0032] | 0.020 / 0.013 |
| q3_arc_hs | arc_challenge | 0.5503 | +0.0051 | [-0.0017, +0.0128] | 0.005 / 0.010 |
| q3_arc_hs | hellaswag | 0.7581 | +0.0002 | [-0.0018, +0.0021] | 0.005 / 0.005 |

### Noise floor (C1: bf16 vs fp32)

| Task | bf16 | fp32 | delta | paired 95% CI |
| --- | --- | --- | --- | --- |
| gsm8k_cot | 0.4049 | 0.4056 | +0.0008 | [-0.0136, +0.0152] |
