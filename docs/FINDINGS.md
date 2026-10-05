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
- Noise floor (C1, G0.3): fp32 vs bf16 on GSM8K strict is 40.56% vs 40.49% (+0.08 [-1.36, +1.52]; flips 3.4% /
  3.5%). The NVFP4 net loss is about 90x this delta and well outside its CI, so G0.3 passes for GSM8K; note that
  even numerically near-identical models flip about 7% of GSM8K items in both directions, so flip rates are read
  against that baseline (NVFP4: 19.8%). MMLU floor runs on boxA. results/phase0/damage_table.json
- ARC-Challenge and HellaSwag (0-shot, acc_norm) confirm GSM8K as the sensitive suite. NVFP4: ARC-C 54.52% ->
  52.56% (-1.96 [-3.33, -0.60]; flips 3.8% / 1.8%), HellaSwag 75.79% -> 75.12% (-0.67 [-1.04, -0.32]; flips 2.0% /
  1.3%). FP8 per-channel (Q3) is indistinguishable from bf16 on both: ARC-C +0.51 [-0.17, +1.28], HellaSwag +0.02
  [-0.18, +0.21], 0.5% flips each way. Q3 on GSM8K and MMLU and the fp32 noise floor run on boxA.
  results/phase0/damage_table.json, results/LEDGER.md

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

## 2026-10-05, Phase 2: NVFP4 vs matched noise at layer 17 (boxS)

- Energy concentration is a property of the SAE basis, not of quantization: the matched-noise control shows the
  same curve (top 1% of features = 62.6% of per-feature shift energy vs 62.8% for NVFP4; Gini 0.945 vs 0.942), and
  the top-energy features are a handful of extremely frequent, very large-activation dense features (one fires on
  24% of tokens with mean activation about 550; several others on 1-3% of tokens with activations above 1,000).
  Per the pre-registered rule, the "concentrated energy" claim is withdrawn. results/phase2/shift/{q1,c2}_L17_summary.json
- The survival metric separates the conditions: among 27,106 features with at least 200 bf16 activations, NVFP4
  drives 12.7% below a survival correlation of 0.5 (37.7% below 0.8; median 0.86) while matched Gaussian noise of
  the same Frobenius error drives 22.4% below 0.5 (46.7% below 0.8; median 0.82). NVFP4 is less destructive at
  the feature level than isotropic noise of the same size. results/phase2/survival_L17.json
- Damaged features are fragile features: median bf16 activation count 558 vs 5,174 overall and median peak 211 vs
  594, matching Duan's finding that survival is predictable from full-precision statistics. 3,424 of the 3,446
  NVFP4-damaged features are also noise-damaged (Jaccard 0.56 with the larger noise set).
- Label asymmetry: NVFP4 amplifies more features than it suppresses (6,330 vs 2,168, frequency ratio >2 or <0.5),
  noise mostly suppresses (2,745 vs 14,043). Only 7 of the top-641 energy features are suppressed, so steering
  and saliency now use the 641 least-surviving features (correlation at or below 0.37) as the primary set.

## 2026-10-05, Phase 2 shift accounting, Q1 at layer 29 (boxS, Gemma Scope 2 65k, 12.25M tokens)

- The SAE is a weaker ruler at layer 29: the feature-explained part is 84% of ||dh||^2 and the error term 59%
  (cross term -43%), against 98% / 11% at layer 17, in line with the lower variance explained (0.80 vs 0.96).
  Per-feature sums cover 44% of the explained energy (12% at layer 17), so less of the shift is correlated motion.
- The shift at layer 29 is dominated by one direction: the top direction of the uncentred shift covariance holds 49%
  of the shift energy, while the mean shift (a constant bias) holds only 2.9%. Per-feature energy is less concentrated
  than at layer 17 (top 1% of features = 37% of the summed energy vs 63%; Gini 0.85 vs 0.94).
- Far fewer features change frequency at layer 29: 787 suppressed and 466 amplified (vs 2,168 and 6,330 at layer 17),
  347 newly dead, 304 newly alive. The C2 control and the survival comparison at layer 29 follow.
  results/phase2/shift/q1_L29_summary.json

## 2026-10-05, Phase 2: NVFP4 vs matched noise at layer 29 (boxS)

- The noise control reproduces the layer-29 shift structure too: the top direction holds 45% of the shift energy under
  matched noise vs 49% under NVFP4, per-feature concentration is identical (top 1% = 37.3% vs 37.4%, Gini 0.86 vs
  0.85), and the mean-shift share is small in both (7.1% vs 2.9%). The single dominant direction is therefore a
  property of the layer-29 residual stream (how it responds to any weight perturbation), not of NVFP4.
- Features survive far better at layer 29 than at layer 17: among 47,320 eligible features (at least 200 bf16
  activations), NVFP4 drives 0.54% below a survival correlation of 0.5 (18.1% below 0.8; median 0.92) vs 12.7% /
  37.7% / 0.86 at layer 17. Matched noise is again more destructive (1.48% below 0.5; 23.1% below 0.8; median 0.90).
  Jaccard of the damaged sets (corr < 0.5) is 0.36. NVFP4 suppresses 787 features at layer 29 vs 2,323 for noise.
  results/phase2/shift/{q1,c2}_L29_summary.json, results/phase2/survival_L29.json
- Reading across layers: the layer-17 feature basis records most of the damage (one third of its features lose
  survival correlation below 0.8), while at layer 29 the shift is large in norm but mostly along one shared direction
  that the SAE features ride together. Steering and saliency therefore use the layer-17 survival set.

## 2026-10-05, Phase 1 accuracy level: layer restore curve (boxA, paired bootstrap vs NVFP4 on 1319 GSM8K items)

- Restoring the N top-ranked layers to bf16 (ranked by single-layer KL recovery: 11, 17, 10, 4, 14, 5, 23, 0, ...)
  recovers 9% / 13% / 17% / 33% / 41% of the GSM8K loss at N = 1 / 2 / 3 / 5 / 8 (N = 8: +2.7 [+0.6, +4.9] pts, recovery
  41% [9%, 74%]). A random ordering of layers recovers 48% at N = 8 (37.0 vs 36.5); ranked minus random is -0.5 pts
  at N = 8 and under +1 pt at every N, never significant. Neither 50% nor 80% recovery is reached with 8 layers (24% of
  the network, 1.1 GB of extra memory). MMLU moves within noise throughout (loss 1.3 pts).
- H1 is therefore NOT supported: no minority of layers carries most of the NVFP4 accuracy loss, and the KL ranking has
  no predictive value for accuracy recovery over a random ranking. The single-layer KL recoveries already said so
  (the best layer recovers 6.5% of the KL). Damage accumulates diffusely across layers.
  results/phase1/restore_curve.json, results/LEDGER.md

## 2026-10-05, Phase 0 completion on boxA (8x H100)

- Replication across boxes and engines: bf16 40.41% and NVFP4 33.81% on GSM8K (boxS: 40.49 / 33.43), paired loss 6.60
  [4.25, 8.95]; MMLU on the 4k subset 60.95% vs 59.62% (loss 1.33 [0.35, 2.33]). The unhooked transformers backend
  gives 33.66% for NVFP4 vs 33.81% with vLLM (-0.2 [-1.3, +1.0]), so the two engines agree within noise.
- FP8 per-channel (Q3) is indistinguishable from bf16 on every suite: GSM8K -0.30 [-1.97, +1.29], MMLU -0.15
  [-0.75, +0.43] (plus the ARC-C and HellaSwag nulls from boxS). FP8 is a clean negative control for NVFP4.
- fp32 noise floor on MMLU: +0.05 [-0.43, +0.55]; with GSM8K +0.08 [-1.36, +1.52] from boxS, G0.3 passes on both
  headline suites. results/phase0/damage_table_bf16_rep.json, results/LEDGER.md

## 2026-10-05, Phase 3: mixed precision budget sweep (boxA, paired bootstrap vs NVFP4 on 1319 GSM8K items)

- Channel-level protection beats whole-layer fallback at equal memory, by a wide margin: magnitude-selected channels at
  2% of weights (92 MB extra) recover 48% of the GSM8K loss (+3.2 [+1.1, +5.3] pts), the same as restoring the 8
  top-ranked whole layers (1,085 MB, 41%, +2.7 [+0.6, +4.9]); at 5% (231 MB) magnitude recovers 51% (+3.3 [+1.4, +5.4]).
  Whole-layer fallback never reaches 45% recovery within 8 layers.
- The feature-guided saliency (FG-proj, primary) recovers 20% / 17% / 45% at 1% / 2% / 5% (+3.0 [+1.0, +4.9] at 5%) and
  TIES with every baseline under the pre-registered rule (a paired win at two of three budgets): vs magnitude -0.4, -2.0,
  -0.4 pts; vs KL-gradient +0.5, -0.4, +0.4; vs random channels -0.2, -0.3, +1.6 (all CIs cross zero). Random channels
  recover about 22% at every budget. The alternative resume wording from the plan applies: channel-level protection
  beats whole-layer fallback; feature guidance does not beat magnitude.
- 80% recovery is not reached at any budget up to 5%; MMLU deltas are all within noise (loss only 1.3 pts).
  results/phase3/budget_sweep.json, results/phase3/memory.json, results/LEDGER.md

## 2026-10-05, Phase 2: interventions at layer 17 (boxA, 500 GSM8K items, paired bootstrap vs no steering)

- No 1% feature set carries the accuracy loss. Gain correction on the 641 least-surviving features +0.8 [-1.2, +2.8]
  (gains are all within a few percent of 1.0, so this was expected to be null), random-feature gain +0.0; the ORACLE
  patch that sets the same 641 features to their bf16 values gives -0.6 [-2.4, +1.2], the 641 highest-energy features
  +1.0 [-1.8, +3.8], 641 random features +1.6 [-0.6, +3.8], and the 3,205 (5%) least-surviving features +0.6 [-1.2, +2.6].
- The damage is nevertheless largely in the SAE basis, and it sits in the dense high-energy features, not in the
  fragile ones: patching ALL 65,536 features (the whole SAE-explained shift, leaving the SAE error term) recovers
  +4.4 [+0.8, +8.0] pts, 63% of the bf16 gap on these items, and patching only the 6,411 highest-energy features (10%)
  recovers the same, +4.6 [+1.2, +8.0] (66%). The 10% least-surviving features recover +0.8 [-1.0, +2.6], 10% random
  features +0.4 [-2.2, +3.2], and the 5% highest-energy features +1.2 [-1.8, +4.2], so the accuracy-relevant part of
  the shift needs most of the top-10% energy set. Replacing the entire layer-17 residual stream recovers +7.4
  [+3.6, +11.2] (106%, i.e. bf16 accuracy) even though layers 18 to 33 stay quantized: the loss is fully formed by
  layer 17, two thirds of it inside the SAE basis (in the frequent, high-energy features) and one third in the SAE
  error term. The features that quantization visibly destroys (low survival correlation) are rare and do not matter
  for GSM8K.
- The best deployable intervention is the simplest: a constant mean-shift bias at layer 17 (estimated on 64 calibration
  sequences, no bf16 model at inference) recovers +2.6 [+0.2, +5.2] pts, 37% of the gap and significant; gain correction
  on the highest-energy features +2.4 [+0.0, +5.0] is borderline. The gated full-set gain run was skipped (subset delta
  under 1 pt). Sanity arms: the bf16 model with the gain hook 42.4 vs bf16 41.6 on the same items.
  results/phase2/steering.json, results/phase2/steer/*.json, results/LEDGER.md
