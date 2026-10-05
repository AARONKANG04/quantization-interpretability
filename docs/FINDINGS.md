# Findings log

Running record of results with their evidence files. Numbers here are copied from script output; the ledger
(results/LEDGER.md) is the canonical, script-written source for the resume placeholders.

## 2026-10-05, Phase 0 (boxS, 1x H100)

- NVFP4 round-to-nearest W4A16 (Q1) on google/gemma-3-4b-pt: GSM8K 8-shot CoT strict 40.49% -> 33.43%
  (paired delta -7.05, 95% CI [-9.40, -4.70]; flips 13.4% correct->incorrect, 6.4% incorrect->correct, so 19.8% of
  answers change for a 7-point net); MMLU 5-shot 59.62% -> 58.33% (-1.29 [-1.80, -0.77]; flips 5.5% / 4.2%).
  results/phase0/damage_table.json
- The QDQ instrument reproduces llm-compressor's NVFP4A16 export bit-for-bit on all 238 matrices once its arithmetic
  is matched (bf16 reciprocal global scale, bf16 block-max/6, shared q/k/v and gate/up global scales).
  results/phase0/crosscheck.json
- Block-outlier shadowing: 7.22% of nonzero weights fall under block_max/24 and 7.23% round to exactly zero
  (weighted over all matrices), so the bound predicts the zeroing almost exactly. Weighted relative error 9.47%
  (NVFP4) vs 2.64% (FP8 per-channel). results/phase0/pilot/q1.json, q3.json
- Fully quantized model vs bf16 on the 500k-token held-out set: mean KL 0.081, median 0.037, top-1 disagreement
  12.3%. results/phase1/layer_sweep/all_x.json

## 2026-10-05, Phase 1 KL-level (boxS)

- Layer damage is diffuse: restoring the single most sensitive layer (11) removes 6.5% of the KL, the top five
  together about 24%, and the 34 single-layer effects sum to 83% of the total (near-additive, little
  interaction). results/phase1/ranked_order.json
- Global-attention layers (5, 11, 17, 23, 29) are the most sensitive: 4.16% of KL per restored global layer vs
  2.14% per local layer (median 2.12); four of the five sit in the top seven; permutation test p = 0.0025.
- Per-token (Q1, 2M general tokens + GSM8K): top-1 flips concentrate on high-entropy positions (highest entropy
  fifth: 2.4x the average flip rate, 2.7x the median KL; lowest fifth: 0.01x) and on non-Latin tokens (2.6x median
  KL). Numerals look robust overall (0.59x flips) because most digits are low-entropy, but at matched uncertainty
  (middle entropy quintile) numerals flip 1.93x and code symbols 1.37x the average. results/phase1/token_buckets_q1.json
- Position (PG-19 up to 32k): median KL is 3.9x the overall median in positions 0-256, 1.7x in 256-1024, and flat
  (about 1.0x) from 1k to 32k; bf16 entropy is flat across positions, so this is context scarcity, not long context.
  results/phase1/token_buckets_q1_pg19.json
- FP8 (Q3) contrast on the same tokens: median KL 0.0041 (Q1: 0.0123 on GSM8K text), top-1 agreement 95.5%.

## 2026-10-05, Phase 2 setup (boxS)

- Gemma Scope 2 resid_post 65k (L0 medium) is a valid ruler across models: layer 17 variance explained 0.958 (bf16)
  vs 0.954 (Q1), L0 66.5 vs 65.3, splice CE +0.131 vs +0.124 nats; layer 29: 0.797 vs 0.786, L0 73 vs 72. The
  reconstruction gap between models is under 1 point at both layers. results/phase2/sae_validation_L{17,29}.json
- Residual norms are large (mean |h| about 32k at layer 17, 76k at layer 29), as expected for Gemma; thresholds
  scale accordingly (means 141 and 667).

## 2026-10-05, Phase 2 shift accounting, Q1 at layer 17 (boxS, Gemma Scope 2 65k, 12.25M tokens)

- The SAE basis captures the shift: the feature-explained part is 98% of ||dh||^2, the SAE error term 10.9%, their
  cross term -8.9%. 64,107 of 65,536 features are active in at least one model.
- Per-feature shift energy is highly concentrated: top 0.1% of features = 41% of the summed energy, top 1% = 63%,
  top 5% = 82%, top 10% = 91%; Gini 0.94. Normalised per activation it is still concentrated but less so (50% of
  energy within 2.2% of features, 80% within 37%).
- Caveat that shapes the interpretation: the summed per-feature energies are only 11.5% of the explained energy,
  so most of the shift is features moving in correlated directions (cross terms), consistent with a shared bias or
  scale component; the mean-shift and top-direction shares are now measured directly, and the steering experiment's
  mean-shift baseline tests whether that component carries the accuracy loss.
- Labels (frequency ratio Q1/bf16): 2,168 suppressed (<0.5x), 6,330 amplified (>2x), 352 newly dead, 338 newly
  alive, 54,919 stable. results/phase2/shift/q1_L17_summary.json; C2 control and layer 29 pending.
