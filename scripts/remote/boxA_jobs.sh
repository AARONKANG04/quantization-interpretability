#!/usr/bin/env bash
# Generate the boxA job list (priority order, `name|deps|command`). usage: boxA_jobs.sh [layer] > jobs.txt
L="${1:-17}"
P="uv run python"
T=/data/kl/kl_500k.pt; R=/data/kl/ref_kl_500k_top256.pt; SW=results/phase1/layer_sweep
SHIFT_TOK="/data/kl/kl_2m.pt /data/kl/lens_10m.pt /data/kl/gsm8k_test.pt"
FE=results/phase2/feats_q1_L$L.json
e() { echo "$1|$2|$3"; }
# --- critical path ---
e sae_bf16_L$L "" "$P scripts/phase2_train_sae.py --model-ckpt /data/ckpt/bf16 --name bf16 --layer $L --tokens 100000000"
e sae_q1_L$L   "" "$P scripts/phase2_train_sae.py --model-ckpt /data/ckpt/q1   --name q1   --layer $L --tokens 100000000"
e lens_train   "" "$P scripts/phase1_tuned_lens.py train --tokens /data/kl/lens_10m.pt --out /data/lens/gemma-3-4b-pt.pt"
e ranked_build "" "$P scripts/phase1_ranked_curve.py --sweep $SW --out results/phase1/ranked_order.json"
e shift_q1_L${L}_v2 "" "$P scripts/phase2_shift.py --layer $L --ckpt-b /data/ckpt/q1 --name q1v2 --tokens $SHIFT_TOK --batch 4"
for n in ranked_top1 ranked_top2 ranked_top3 ranked_top5 ranked_top8 random_s0_top1 random_s0_top2 random_s0_top3 random_s0_top5 random_s0_top8; do
  e rc_eval_$n ranked_build "$P scripts/phase0_eval.py --ckpt /data/ckpt/$n --name rc_$n --tasks gsm8k_cot --gpu-util 0.8 && $P scripts/phase0_eval.py --ckpt /data/ckpt/$n --name rc_${n}_mmlu --tasks mmlu --limit 70 --gpu-util 0.8"
done
e lens_apply_q1 lens_train "$P scripts/phase1_tuned_lens.py apply --lens /data/lens/gemma-3-4b-pt.pt --ckpt-b /data/ckpt/q1 --tokens $T --out results/phase1/lens_q1.json"
# --- needs the feature set from boxS (mark with: queue.sh done feats_ready) ---
e saliency feats_ready "$P scripts/phase3_saliency.py --feats $FE --set top --sae-layer $L --out /data/saliency/scores_L$L.pt"
e select "saliency,ranked_build" "$P scripts/phase3_select.py --scores /data/saliency/scores_L$L.pt --budgets 0.01,0.02,0.05 --ranked results/phase1/ranked_order.json --out results/phase3/selections.json"
for m in fg_proj mag klg rand; do for b in 0010 0020 0050; do
  e mix_build_${m}_$b select "$P scripts/phase3_build.py --selection /data/selections/${m}_$b.json --out /data/ckpt/mix_${m}_$b"
  e mix_eval_${m}_$b mix_build_${m}_$b "$P scripts/phase0_eval.py --ckpt /data/ckpt/mix_${m}_$b --name mix_${m}_$b --tasks gsm8k_cot --gpu-util 0.8 && $P scripts/phase0_eval.py --ckpt /data/ckpt/mix_${m}_$b --name mix_${m}_${b}_mmlu --tasks mmlu --limit 70 --gpu-util 0.8"
done; done
for mode in none gain gain_random mean_shift oracle oracle_random full_residual bf16_gain; do
  e steer_$mode feats_ready "$P scripts/phase2_steer_eval.py --ckpt-b /data/ckpt/q1 --layer $L --feats $FE --mode $mode --set top --limit 500 --out results/phase2/steer/q1_L${L}_$mode.json"
done
e steer_gain_full steer_gain "$P scripts/phase2_steer_eval.py --ckpt-b /data/ckpt/q1 --layer $L --feats $FE --mode gain --set top --limit 1319 --out results/phase2/steer/q1_L${L}_gain_full.json"
e steer_none_full steer_none "$P scripts/phase2_steer_eval.py --ckpt-b /data/ckpt/q1 --layer $L --mode none --limit 1319 --out results/phase2/steer/q1_L${L}_none_full.json"
# --- fp32 noise floor through the transformers backend ---
e c1_fp32_gsm8k "" "$P scripts/phase0_eval.py --ckpt /data/ckpt/bf16 --name c1_fp32 --tasks gsm8k_cot --backend hf --dtype float32 --hf-batch 16"
e c1_fp32_mmlu  "" "$P scripts/phase0_eval.py --ckpt /data/ckpt/bf16 --name c1_fp32_mmlu --tasks mmlu --limit 70 --backend hf --dtype float32 --hf-batch 16"
# --- fast fillers: quantize-one-layer sweep, controls, extra SAE layers ---
for r in 0-8 9-16 17-25 26-33; do e qone_$r "" "$P scripts/phase1_layer_sweep.py --mode quantize_one --layers $r --tokens $T --ref $R --batch 2 --out $SW"; done
e ckpt_q3 "" "$P scripts/phase0_pilot.py --scheme fp8_e4m3 --out /data/ckpt/q3 --stats-out results/phase0/pilot/q3.json"
e ckpt_c3 "" "$P scripts/phase0_pilot.py --scheme error_permuted --out /data/ckpt/c3 --stats-out results/phase0/pilot/c3.json"
e lens_apply_q3 "lens_train,ckpt_q3" "$P scripts/phase1_tuned_lens.py apply --lens /data/lens/gemma-3-4b-pt.pt --ckpt-b /data/ckpt/q3 --tokens $T --out results/phase1/lens_q3.json"
e dump_c2_kl2m "" "$P scripts/phase1_token_dump.py --ckpt-b /data/ckpt/c2 --name c2 --tokens /data/kl/kl_2m.pt --batch 2"
e shift_c3_L$L ckpt_c3 "$P scripts/phase2_shift.py --layer $L --ckpt-b /data/ckpt/c3 --name c3 --tokens $SHIFT_TOK --batch 4"
for LL in 9 22; do for c in q1 c2; do e shift_${c}_L$LL "" "$P scripts/phase2_shift.py --layer $LL --ckpt-b /data/ckpt/$c --name $c --tokens $SHIFT_TOK --batch 4"; done; done
# --- slower fillers ---
e ranked_build_s12 ranked_build "$P scripts/phase1_ranked_curve.py --sweep $SW --out results/phase1/ranked_order_s12.json --random-seeds 1,2 --skip-ranked"
for s in 1 2; do for n in 1 2 3 5 8; do
  e rc_eval_random_s${s}_top$n ranked_build_s12 "$P scripts/phase0_eval.py --ckpt /data/ckpt/random_s${s}_top$n --name rc_random_s${s}_top$n --tasks gsm8k_cot --gpu-util 0.8 && $P scripts/phase0_eval.py --ckpt /data/ckpt/random_s${s}_top$n --name rc_random_s${s}_top${n}_mmlu --tasks mmlu --limit 70 --gpu-util 0.8"
done; done
for r in 0-3 4-7 8-11 12-15 16-19 20-23 24-28 29-33; do e qmat_$r "" "$P scripts/phase1_layer_sweep.py --mode quantize_one_matrix --layers $r --tokens $T --ref $R --batch 2 --out $SW"; done
e sae_bf16_L29 "" "$P scripts/phase2_train_sae.py --model-ckpt /data/ckpt/bf16 --name bf16 --layer 29 --tokens 100000000"
e sae_q1_L29   "" "$P scripts/phase2_train_sae.py --model-ckpt /data/ckpt/q1   --name q1   --layer 29 --tokens 100000000"
