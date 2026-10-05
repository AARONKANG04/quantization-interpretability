# Number ledger

Written only by scripts. Every row carries its control in the same table. Definitions: plan doc section 8.

| Placeholder | Value | 95% CI | Control | Source file | Updated |
| --- | --- | --- | --- | --- | --- |
| [X]% of loss in [N] layers | 41% of the GSM8K loss with the 8 top-ranked layers (24% of layers); 80% recovery not reached; avg GSM8K+MMLU at N=8: 28% | [9.2, 73.6] | random orderings recover 48%, 21%, 30%; ranked minus random +0.6 pts | results/phase1/restore_curve.json | 2026-10-05 |
| concentrated on [token class] | | | C1, C2 | results/phase1/token_buckets.parquet | |
| [N]% of features ([category]) carry [X]% of shift | | | C2 curve, random-feature categories, error-term share | results/phase2/shift_share.json | |
| steering recovered [X] of [Y] points | +0.8 of 6.6 points with gain correction on the 641 least-surviving features (n.s.); mean-shift bias +2.6 [+0.2, +5.2] | [-1.2, +2.8] | random-feature gain +0.0; oracle 1% -0.6; full residual +7.4 | results/phase2/steering.json | 2026-10-05 |
| [Z]% of weights in bf16 | 5% (80% recovery not reached at any budget; best feature-guided (FG-proj) budget) |  | RAND, MAG, KLG at the same budgets | results/phase3/budget_sweep.json | 2026-10-05 |
| recovering [X]% of FP4 loss | 45% of the GSM8K loss at 5% (231 MB); avg GSM8K+MMLU 19% | [14.9, 74.7] | KL-gradient: ties; magnitude: ties; random channels: ties | results/phase3/budget_sweep.json | 2026-10-05 |
| [M]% of memory of full-layer fallback | whole-layer fallback never reaches this recovery within 8 layers (1085 MB, 41%); 231 MB is 21% of that |  | LAYER curve (ranked restore evals) | results/phase3/memory.json | 2026-10-05 |

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

## Phase 1: layer restore curve (ref bf16_rep, quant q1_rep, paired bootstrap, 2026-10-05T02:34:26)

**gsm8k_cot**: bf16_rep 40.41, q1_rep 33.81, loss 6.60 [+4.2, +8.9] on 1319 items. 50% recovery at N = None, 80% at N = None.

| N layers in bf16 | Ranked acc | Recovered pts [95% CI] | Recovery | Random orderings acc | Ranked minus random (mean) |
| --- | --- | --- | --- | --- | --- |
| 1 | 34.42 | +0.6 [-0.9, +2.2] | 9% | 33.51, 34.12, 35.10 | +0.2 (n.s.) |
| 2 | 34.65 | +0.8 [-1.0, +2.7] | 13% | 34.27, 34.12, 34.34 | +0.4 (n.s.) |
| 3 | 34.95 | +1.1 [-0.7, +3.0] | 17% | 34.04, 36.24, 35.63 | -0.4 (n.s.) |
| 5 | 36.01 | +2.2 [+0.2, +4.2] | 33% | 35.63, 35.78, 36.09 | +0.2 (n.s.) |
| 8 | 36.54 | +2.7 [+0.6, +4.9] | 41% | 37.00, 35.18, 35.78 | +0.6 (n.s.) |

**mmlu**: bf16_rep 60.95, q1_rep 59.62, loss 1.33 [+0.4, +2.3] on 3990 items. 50% recovery at N = None, 80% at N = None.

| N layers in bf16 | Ranked acc | Recovered pts [95% CI] | Recovery | Random orderings acc | Ranked minus random (mean) |
| --- | --- | --- | --- | --- | --- |
| 1 | 59.90 | +0.3 [-0.3, +0.9] | 21% | 59.42, 59.75, 59.77 | +0.3 (n.s.) |
| 2 | 59.87 | +0.3 [-0.5, +1.0] | 19% | 59.20, 59.35, 59.80 | +0.4 (n.s.) |
| 3 | 60.13 | +0.5 [-0.3, +1.3] | 38% | 59.42, 59.97, 59.90 | +0.4 (n.s.) |
| 5 | 59.92 | +0.3 [-0.5, +1.1] | 23% | 59.50, 60.03, 59.72 | +0.2 (n.s.) |
| 8 | 59.82 | +0.2 [-0.7, +1.1] | 15% | 60.03, 60.00, 59.72 | -0.1 (n.s.) |

## Phase 3: mixed precision budget sweep (ref bf16_rep, quant q1_rep, paired bootstrap, 2026-10-05T02:34:36)

**gsm8k_cot**: loss 6.60 [+4.2, +8.9] pts on 1319 items.

| Method | Budget | Extra MB | Acc | Recovered pts [95% CI] | Recovery |
| --- | --- | --- | --- | --- | --- |
| feature-guided (FG-proj) | 1% | 46 | 35.10 | +1.3 [-0.3, +3.0] | 20% |
| feature-guided (FG-proj) | 2% | 92 | 34.95 | +1.1 [-0.7, +3.0] | 17% |
| feature-guided (FG-proj) | 5% | 231 | 36.77 | +3.0 [+1.0, +4.9] | 45% |
| KL-gradient | 1% | 46 | 34.57 | +0.8 [-0.8, +2.4] | 11% |
| KL-gradient | 2% | 92 | 35.33 | +1.5 [-0.2, +3.3] | 23% |
| KL-gradient | 5% | 231 | 36.39 | +2.6 [+0.8, +4.4] | 39% |
| magnitude | 1% | 46 | 35.48 | +1.7 [-0.3, +3.6] | 25% |
| magnitude | 2% | 92 | 37.00 | +3.2 [+1.1, +5.3] | 48% |
| magnitude | 5% | 231 | 37.15 | +3.3 [+1.4, +5.4] | 51% |
| random channels | 1% | 46 | 35.33 | +1.5 [+0.0, +3.1] | 23% |
| random channels | 2% | 92 | 35.25 | +1.4 [-0.4, +3.3] | 22% |
| random channels | 5% | 231 | 35.18 | +1.4 [-0.5, +3.3] | 21% |
| whole-layer fallback | 1 layers | 136 | 34.42 | +0.6 [-0.9, +2.2] | 9% |
| whole-layer fallback | 2 layers | 271 | 34.65 | +0.8 [-1.0, +2.7] | 13% |
| whole-layer fallback | 3 layers | 407 | 34.95 | +1.1 [-0.7, +3.0] | 17% |
| whole-layer fallback | 5 layers | 678 | 36.01 | +2.2 [+0.2, +4.2] | 33% |
| whole-layer fallback | 8 layers | 1085 | 36.54 | +2.7 [+0.6, +4.9] | 41% |

Decision rule (feature-guided (FG-proj) vs baseline, paired, needs a win at two of three budgets):

- vs KL-gradient: **ties** (1%: +0.5 [-1.3, +2.4]; 2%: -0.4 [-2.3, +1.5]; 5%: +0.4 [-1.7, +2.4])
- vs magnitude: **ties** (1%: -0.4 [-2.4, +1.6]; 2%: -2.0 [-4.1, +0.1]; 5%: -0.4 [-2.6, +1.8])
- vs random channels: **ties** (1%: -0.2 [-2.0, +1.5]; 2%: -0.3 [-2.2, +1.7]; 5%: +1.6 [-0.4, +3.5])

**mmlu**: loss 1.33 [+0.4, +2.3] pts on 3990 items.

| Method | Budget | Extra MB | Acc | Recovered pts [95% CI] | Recovery |
| --- | --- | --- | --- | --- | --- |
| feature-guided (FG-proj) | 1% | 46 | 59.77 | +0.2 [-0.4, +0.7] | 11% |
| feature-guided (FG-proj) | 2% | 92 | 59.75 | +0.1 [-0.5, +0.7] | 9% |
| feature-guided (FG-proj) | 5% | 231 | 59.52 | -0.1 [-0.7, +0.6] | -8% |
| KL-gradient | 1% | 46 | 59.60 | -0.0 [-0.6, +0.6] | -2% |
| KL-gradient | 2% | 92 | 59.70 | +0.1 [-0.6, +0.7] | 6% |
| KL-gradient | 5% | 231 | 59.67 | +0.1 [-0.6, +0.7] | 4% |
| magnitude | 1% | 46 | 59.47 | -0.2 [-0.9, +0.6] | -11% |
| magnitude | 2% | 92 | 59.90 | +0.3 [-0.5, +1.0] | 21% |
| magnitude | 5% | 231 | 59.92 | +0.3 [-0.5, +1.1] | 23% |
| random channels | 1% | 46 | 59.75 | +0.1 [-0.4, +0.7] | 9% |
| random channels | 2% | 92 | 59.60 | -0.0 [-0.6, +0.6] | -2% |
| random channels | 5% | 231 | 59.57 | -0.1 [-0.7, +0.6] | -4% |
| whole-layer fallback | 1 layers | 136 | 59.90 | +0.3 [-0.3, +0.9] | 21% |
| whole-layer fallback | 2 layers | 271 | 59.87 | +0.3 [-0.5, +1.0] | 19% |
| whole-layer fallback | 3 layers | 407 | 60.13 | +0.5 [-0.3, +1.3] | 38% |
| whole-layer fallback | 5 layers | 678 | 59.92 | +0.3 [-0.5, +1.1] | 23% |
| whole-layer fallback | 8 layers | 1085 | 59.82 | +0.2 [-0.7, +1.1] | 15% |

Decision rule (feature-guided (FG-proj) vs baseline, paired, needs a win at two of three budgets):

- vs KL-gradient: **ties** (1%: +0.2 [-0.5, +0.8]; 2%: +0.1 [-0.7, +0.7]; 5%: -0.2 [-0.9, +0.6])
- vs magnitude: **ties** (1%: +0.3 [-0.4, +1.0]; 2%: -0.2 [-1.0, +0.6]; 5%: -0.4 [-1.2, +0.4])
- vs random channels: **ties** (1%: +0.0 [-0.6, +0.6]; 2%: +0.2 [-0.5, +0.9]; 5%: -0.1 [-0.8, +0.7])

## Phase 2: steering at layer 17 (GSM8K subset, 500 items, paired bootstrap vs no steering, 2026-10-05T02:34:41)

No steering (NVFP4, HF engine): 34.6. Same items, vLLM engine: bf16 41.6, NVFP4 35.6.

| Arm | Feature set | Features | Acc | Delta vs none [95% CI] | Share of bf16 gap |
| --- | --- | --- | --- | --- | --- |
| gain | survival | 641 | 35.4 | +0.8 [-1.2, +2.8] | 11% |
| gain_random | survival | 641 | 34.6 | +0.0 [-2.0, +2.0] | 0% |
| gain | top | 641 | 37.0 | +2.4 [+0.0, +5.0] | 34% |
| mean_shift | survival | 641 | 37.2 | +2.6 [+0.2, +5.2] * | 37% |
| oracle | survival | 641 | 34.0 | -0.6 [-2.4, +1.2] | -9% |
| oracle_random | survival | 641 | 36.2 | +1.6 [-0.6, +3.8] | 23% |
| oracle | top | 641 | 35.6 | +1.0 [-1.8, +3.8] | 14% |
| oracle | survival_5pct | 3205 | 35.2 | +0.6 [-1.2, +2.6] | 9% |
| oracle | control_5pct | 3205 | 35.6 | +1.0 [-1.0, +3.2] | 14% |
| oracle | top_5pct | 3205 | 35.8 | +1.2 [-1.8, +4.2] | 17% |
| oracle | survival_10pct | 6411 | 35.4 | +0.8 [-1.0, +2.6] | 11% |
| oracle | control_10pct | 6411 | 35.0 | +0.4 [-2.2, +3.2] | 6% |
| oracle | top_10pct | 6411 | 39.2 | +4.6 [+1.2, +8.0] * | 66% |
| oracle | all | 65536 | 39.0 | +4.4 [+0.8, +8.0] * | 63% |
| full_residual | survival | 641 | 42.0 | +7.4 [+3.6, +11.2] * | 106% |
| bf16_gain | survival | 641 | 42.4 | +7.8 [+3.8, +11.8] * | 111% |

Engine check on the full set: HF backend 33.66 vs vLLM 33.81, delta -0.2 [-1.3, +1.0].

## Damage table (ref bf16_rep, paired bootstrap, 2026-10-05T02:35:36)

| Condition | Task | Accuracy | Delta vs ref | 95% CI | Flips c->i / i->c |
| --- | --- | --- | --- | --- | --- |
| bf16_rep | gsm8k_cot | 0.4041 | | | |
| bf16_rep_mmlu | mmlu | 0.6095 | | | |
| c1_fp32_mmlu | mmlu | 0.6100 | +0.0005 | [-0.0043, +0.0055] | 0.012 / 0.013 |
| mix_fg_proj_0010 | gsm8k_cot | 0.3510 | -0.0531 | [-0.0766, -0.0288] | 0.124 / 0.071 |
| mix_fg_proj_0010_mmlu | mmlu | 0.5977 | -0.0118 | [-0.0221, -0.0020] | 0.058 / 0.046 |
| mix_fg_proj_0020 | gsm8k_cot | 0.3495 | -0.0546 | [-0.0773, -0.0318] | 0.120 / 0.065 |
| mix_fg_proj_0020_mmlu | mmlu | 0.5975 | -0.0120 | [-0.0223, -0.0020] | 0.059 / 0.047 |
| mix_fg_proj_0050 | gsm8k_cot | 0.3677 | -0.0364 | [-0.0591, -0.0136] | 0.108 / 0.071 |
| mix_fg_proj_0050_mmlu | mmlu | 0.5952 | -0.0143 | [-0.0238, -0.0048] | 0.055 / 0.041 |
| mix_klg_0010 | gsm8k_cot | 0.3457 | -0.0584 | [-0.0811, -0.0356] | 0.121 / 0.063 |
| mix_klg_0010_mmlu | mmlu | 0.5960 | -0.0135 | [-0.0236, -0.0035] | 0.059 / 0.045 |
| mix_klg_0020 | gsm8k_cot | 0.3533 | -0.0508 | [-0.0751, -0.0265] | 0.126 / 0.075 |
| mix_klg_0020_mmlu | mmlu | 0.5970 | -0.0125 | [-0.0223, -0.0028] | 0.056 / 0.043 |
| mix_klg_0050 | gsm8k_cot | 0.3639 | -0.0402 | [-0.0629, -0.0174] | 0.113 / 0.073 |
| mix_klg_0050_mmlu | mmlu | 0.5967 | -0.0128 | [-0.0223, -0.0033] | 0.054 / 0.041 |
| mix_mag_0010 | gsm8k_cot | 0.3548 | -0.0493 | [-0.0720, -0.0265] | 0.116 / 0.067 |
| mix_mag_0010_mmlu | mmlu | 0.5947 | -0.0148 | [-0.0246, -0.0050] | 0.056 / 0.042 |
| mix_mag_0020 | gsm8k_cot | 0.3700 | -0.0341 | [-0.0569, -0.0114] | 0.108 / 0.074 |
| mix_mag_0020_mmlu | mmlu | 0.5990 | -0.0105 | [-0.0201, -0.0010] | 0.053 / 0.042 |
| mix_mag_0050 | gsm8k_cot | 0.3715 | -0.0326 | [-0.0553, -0.0106] | 0.104 / 0.071 |
| mix_mag_0050_mmlu | mmlu | 0.5992 | -0.0103 | [-0.0195, -0.0013] | 0.048 / 0.038 |
| mix_rand_0010 | gsm8k_cot | 0.3533 | -0.0508 | [-0.0743, -0.0273] | 0.121 / 0.070 |
| mix_rand_0010_mmlu | mmlu | 0.5975 | -0.0120 | [-0.0221, -0.0023] | 0.056 / 0.044 |
| mix_rand_0020 | gsm8k_cot | 0.3525 | -0.0516 | [-0.0751, -0.0281] | 0.123 / 0.071 |
| mix_rand_0020_mmlu | mmlu | 0.5960 | -0.0135 | [-0.0233, -0.0038] | 0.056 / 0.043 |
| mix_rand_0050 | gsm8k_cot | 0.3518 | -0.0523 | [-0.0758, -0.0288] | 0.122 / 0.070 |
| mix_rand_0050_mmlu | mmlu | 0.5957 | -0.0138 | [-0.0233, -0.0045] | 0.052 / 0.038 |
| q1_rep | gsm8k_cot | 0.3381 | -0.0660 | [-0.0895, -0.0425] | 0.131 / 0.065 |
| q1_rep_mmlu | mmlu | 0.5962 | -0.0133 | [-0.0233, -0.0035] | 0.058 / 0.045 |
| q3 | gsm8k_cot | 0.4011 | -0.0030 | [-0.0197, +0.0129] | 0.047 / 0.044 |
| q3 | mmlu | 0.5941 | -0.0015 | [-0.0075, +0.0043] | 0.019 / 0.017 |
| rc_random_s0_top1 | gsm8k_cot | 0.3351 | -0.0690 | [-0.0933, -0.0455] | 0.136 / 0.067 |
| rc_random_s0_top1_mmlu | mmlu | 0.5942 | -0.0153 | [-0.0256, -0.0053] | 0.060 / 0.045 |
| rc_random_s0_top2 | gsm8k_cot | 0.3427 | -0.0614 | [-0.0857, -0.0371] | 0.132 / 0.071 |
| rc_random_s0_top2_mmlu | mmlu | 0.5920 | -0.0175 | [-0.0276, -0.0078] | 0.059 / 0.042 |
| rc_random_s0_top3 | gsm8k_cot | 0.3404 | -0.0637 | [-0.0879, -0.0402] | 0.131 / 0.067 |
| rc_random_s0_top3_mmlu | mmlu | 0.5942 | -0.0153 | [-0.0256, -0.0055] | 0.060 / 0.045 |
| rc_random_s0_top5 | gsm8k_cot | 0.3563 | -0.0478 | [-0.0713, -0.0250] | 0.118 / 0.070 |
| rc_random_s0_top5_mmlu | mmlu | 0.5950 | -0.0145 | [-0.0243, -0.0048] | 0.057 / 0.042 |
| rc_random_s0_top8 | gsm8k_cot | 0.3700 | -0.0341 | [-0.0569, -0.0114] | 0.107 / 0.073 |
| rc_random_s0_top8_mmlu | mmlu | 0.6003 | -0.0093 | [-0.0188, +0.0003] | 0.053 / 0.044 |
| rc_random_s1_top1 | gsm8k_cot | 0.3412 | -0.0629 | [-0.0864, -0.0402] | 0.124 / 0.061 |
| rc_random_s1_top1_mmlu | mmlu | 0.5975 | -0.0120 | [-0.0221, -0.0020] | 0.058 / 0.046 |
| rc_random_s1_top2 | gsm8k_cot | 0.3412 | -0.0629 | [-0.0864, -0.0394] | 0.126 / 0.063 |
| rc_random_s1_top2_mmlu | mmlu | 0.5935 | -0.0160 | [-0.0261, -0.0060] | 0.061 / 0.045 |
| rc_random_s1_top3 | gsm8k_cot | 0.3624 | -0.0417 | [-0.0652, -0.0182] | 0.113 / 0.071 |
| rc_random_s1_top3_mmlu | mmlu | 0.5997 | -0.0098 | [-0.0198, +0.0003] | 0.056 / 0.046 |
| rc_random_s1_top5 | gsm8k_cot | 0.3578 | -0.0462 | [-0.0690, -0.0235] | 0.112 / 0.066 |
| rc_random_s1_top5_mmlu | mmlu | 0.6003 | -0.0093 | [-0.0193, +0.0008] | 0.055 / 0.045 |
| rc_random_s1_top8 | gsm8k_cot | 0.3518 | -0.0523 | [-0.0758, -0.0296] | 0.120 / 0.067 |
| rc_random_s1_top8_mmlu | mmlu | 0.6000 | -0.0095 | [-0.0193, +0.0000] | 0.053 / 0.043 |
| rc_random_s2_top1 | gsm8k_cot | 0.3510 | -0.0531 | [-0.0766, -0.0296] | 0.123 / 0.070 |
| rc_random_s2_top1_mmlu | mmlu | 0.5977 | -0.0118 | [-0.0216, -0.0020] | 0.056 / 0.044 |
| rc_random_s2_top2 | gsm8k_cot | 0.3434 | -0.0607 | [-0.0842, -0.0371] | 0.127 / 0.066 |
| rc_random_s2_top2_mmlu | mmlu | 0.5980 | -0.0115 | [-0.0213, -0.0018] | 0.056 / 0.045 |
| rc_random_s2_top3 | gsm8k_cot | 0.3563 | -0.0478 | [-0.0713, -0.0243] | 0.120 / 0.072 |
| rc_random_s2_top3_mmlu | mmlu | 0.5990 | -0.0105 | [-0.0206, -0.0005] | 0.056 / 0.046 |
| rc_random_s2_top5 | gsm8k_cot | 0.3609 | -0.0432 | [-0.0667, -0.0205] | 0.114 / 0.071 |
| rc_random_s2_top5_mmlu | mmlu | 0.5972 | -0.0123 | [-0.0223, -0.0025] | 0.058 / 0.045 |
| rc_random_s2_top8 | gsm8k_cot | 0.3578 | -0.0462 | [-0.0690, -0.0227] | 0.116 / 0.070 |
| rc_random_s2_top8_mmlu | mmlu | 0.5972 | -0.0123 | [-0.0223, -0.0025] | 0.056 / 0.044 |
| rc_ranked_top1 | gsm8k_cot | 0.3442 | -0.0599 | [-0.0834, -0.0364] | 0.130 / 0.070 |
| rc_ranked_top1_mmlu | mmlu | 0.5990 | -0.0105 | [-0.0201, -0.0010] | 0.053 / 0.042 |
| rc_ranked_top2 | gsm8k_cot | 0.3465 | -0.0576 | [-0.0804, -0.0349] | 0.121 / 0.063 |
| rc_ranked_top2_mmlu | mmlu | 0.5987 | -0.0108 | [-0.0201, -0.0015] | 0.050 / 0.040 |
| rc_ranked_top3 | gsm8k_cot | 0.3495 | -0.0546 | [-0.0766, -0.0326] | 0.114 / 0.060 |
| rc_ranked_top3_mmlu | mmlu | 0.6013 | -0.0083 | [-0.0173, +0.0010] | 0.048 / 0.040 |
| rc_ranked_top5 | gsm8k_cot | 0.3601 | -0.0440 | [-0.0652, -0.0220] | 0.104 / 0.060 |
| rc_ranked_top5_mmlu | mmlu | 0.5992 | -0.0103 | [-0.0190, -0.0015] | 0.045 / 0.035 |
| rc_ranked_top8 | gsm8k_cot | 0.3654 | -0.0387 | [-0.0607, -0.0167] | 0.102 / 0.063 |
| rc_ranked_top8_mmlu | mmlu | 0.5982 | -0.0113 | [-0.0198, -0.0030] | 0.042 / 0.031 |

### Noise floor (C1: bf16 vs fp32)

| Task | bf16 | fp32 | delta | paired 95% CI |
| --- | --- | --- | --- | --- |
| mmlu | 0.6095 | 0.6100 | +0.0005 | [-0.0043, +0.0055] |
