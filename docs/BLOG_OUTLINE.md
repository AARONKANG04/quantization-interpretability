# Blog post outline: "Quantization damage is diffuse"

Working title: What 4-bit quantization actually breaks in Gemma 3 4B (nothing in particular, everywhere a little).
Audience: ML engineers who quantize models and people who use SAEs for interpretability. Length: about 2,000 words,
six figures from `results/figures/`.

1. Hook: NVFP4 costs Gemma 3 4B 7 GSM8K points, FP8 costs nothing. Where did the 7 points go? (f0_damage)
2. The instrument: a bit-exact NVFP4 quantizer, block-outlier shadowing, why a noise control and an fp32 floor
   are mandatory before any interpretability claim.
3. Layers: the ranking that looked clean in KL space (f1_layer_ranking) and collapsed to random in accuracy space
   (f3_restore_curve). Lesson: KL recoveries are near-additive and tiny; accuracy needs paired CIs and a random
   ordering baseline.
4. Tokens: damage lands where the model is uncertain and in the first 256 tokens of a document (f2_token_buckets).
5. Features: three traps and one finding. Trap 1, energy concentration is a basis artifact (the noise control
   matches). Trap 2, the features quantization visibly damages are rare and carry no accuracy. Trap 3, gain
   correction cannot work when gains are 1.0. Finding: the 10% highest-energy features carry two thirds of the
   recoverable loss, the full residual all of it, and a constant bias recovers a third (f5_steering, f6_survival).
6. The fix: thin protection spread over every matrix beats thick protection of whole layers at a tenth of the
   memory; SAE guidance ties with magnitude (f4_budget_sweep). Say plainly that the fancy method did not win.
7. What a practitioner should do: eval on multi-step generation, add a bias correction, protect by magnitude at
   1 to 2%, keep random baselines.
8. Methods box: paired bootstrap, pre-registered decision rules, the job queue that ran 100 jobs on 8 GPUs in
   2.5 hours with zero failures, the trace viewer.
