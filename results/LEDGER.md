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

## Noise floor (C1: bf16 vs fp32)

| Suite | bf16 | fp32 | delta | paired 95% CI |
| --- | --- | --- | --- | --- |

## GPU-hour tally

| Phase | H100-hours | RTX PRO 6000-hours | Notes |
| --- | --- | --- | --- |
