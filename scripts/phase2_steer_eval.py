"""E2.5: GSM8K (8-shot CoT) accuracy of a perturbed model under an intervention at one layer, through lm-eval's
HF backend so hooks stay attached during generation. All modes share this path, including `none`, so deltas are
paired on identical items and prompts.

usage: python scripts/phase2_steer_eval.py --ckpt-b /data/ckpt/q1 --layer 17 --feats results/phase2/feats_q1_L17.json \
         --mode gain --set suppressed --limit 500 --out results/phase2/steer/q1_L17_gain_suppressed.json
modes: none | gain (gain correction on --set) | gain_random (gains applied to the control set) | mean_shift (bias =
mean residual delta from the twin on --calib-seqs sequences) | oracle (feature patch from the lockstep bf16 twin on
--set) | oracle_random | full_residual (upper bound) | bf16_gain (sanity: the intervention applied to the bf16 model).
"""
import argparse
import json
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.models import ResidualCapture, ResidualPatch, load_gemma3_text  # noqa: E402
from qi.sae.gemma_scope import load_gemma_scope2  # noqa: E402
from qi.sae.jumprelu import JumpReLUSAE  # noqa: E402
from qi.sae.steer import TwinRunner, full_residual_patch, gain_correction, mean_shift, oracle_patch  # noqa: E402


@torch.no_grad()
def mean_delta(model_a, model_b, layer, tokens_file, n_seqs, batch=4):
    ids = torch.load(tokens_file)["input_ids"][:n_seqs]
    acc, n = None, 0
    for s in range(0, ids.shape[0], batch):
        x = ids[s : s + batch].to("cuda", torch.long)
        with ResidualCapture(model_a, layers=[layer]) as ca:
            model_a(input_ids=x)
        with ResidualCapture(model_b, layers=[layer]) as cb:
            model_b(input_ids=x)
        d = (ca.acts[layer].float() - cb.acts[layer].float()).reshape(-1, ca.acts[layer].shape[-1]).sum(0)
        acc = d if acc is None else acc + d
        n += x.numel()
    return acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-b", required=True); ap.add_argument("--model-a", default="google/gemma-3-4b-pt")
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--feats", default=None)
    ap.add_argument("--sae-file", default=None); ap.add_argument("--width", default="65k"); ap.add_argument("--l0", default="medium")
    ap.add_argument("--mode", required=True); ap.add_argument("--set", default="suppressed")
    ap.add_argument("--limit", type=int, default=500); ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib-tokens", default="/data/kl/kl_500k.pt"); ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_b, tok = load_gemma3_text(args.ckpt_b)
    needs_twin = args.mode in ("oracle", "oracle_random", "full_residual", "mean_shift")
    model_a = load_gemma3_text(args.model_a)[0] if needs_twin else None
    sae = None
    if args.mode not in ("none", "mean_shift", "full_residual"):
        sae = JumpReLUSAE.load(args.sae_file, device="cuda") if args.sae_file else load_gemma_scope2(args.layer, args.width, args.l0, device="cuda")
    feats_info = json.loads(Path(args.feats).read_text()) if args.feats else {}
    ids = feats_info.get("control" if args.mode.endswith("_random") else args.set, [])
    gain_src = feats_info.get(args.set, []) if args.mode.endswith("_random") else ids  # the control reuses the set's gains
    ids = ids[: len(gain_src)] if args.mode.endswith("_random") else ids
    feats = torch.tensor(ids, device="cuda", dtype=torch.long)
    gains = torch.tensor([feats_info.get("gains_top", {}).get(str(i), 1.0) for i in gain_src], device="cuda")
    if args.mode.endswith("_random") and args.mode.startswith("gain"):
        # random control gets the same gain multiset, shuffled
        gains = gains[torch.randperm(len(gains), device="cuda")] if len(gains) else gains
    print(f"{args.mode}: {len(ids)} features, gains median {gains.median().item() if len(gains) else 1.0:.3f}", flush=True)

    target = model_b
    if args.mode == "bf16_gain":
        target = load_gemma3_text(args.model_a)[0]
    ctx = []
    if args.mode in ("gain", "gain_random", "bf16_gain"):
        ctx.append(ResidualPatch(target, args.layer, gain_correction(sae, feats, gains)))
    elif args.mode == "mean_shift":
        bias = mean_delta(model_a, model_b, args.layer, args.calib_tokens, args.calib_seqs)
        ctx.append(ResidualPatch(target, args.layer, mean_shift(bias)))
    elif args.mode in ("oracle", "oracle_random"):
        twin = TwinRunner(model_b, model_a, args.layer); ctx.append(twin)
        ctx.append(ResidualPatch(target, args.layer, oracle_patch(sae, feats, twin)))
    elif args.mode == "full_residual":
        twin = TwinRunner(model_b, model_a, args.layer); ctx.append(twin)
        ctx.append(ResidualPatch(target, args.layer, full_residual_patch(twin)))
    elif args.mode != "none":
        raise SystemExit(f"unknown mode {args.mode}")

    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=target, tokenizer=tok, batch_size=args.batch_size, max_length=4096)
    for c in ctx:
        c.__enter__()
    try:
        res = lm_eval.simple_evaluate(model=lm, tasks=["gsm8k_cot"], num_fewshot=8, limit=args.limit, log_samples=True, random_seed=1234)
    finally:
        for c in reversed(ctx):
            c.__exit__(None, None, None)
    r = res["results"]["gsm8k_cot"]
    items = [{"doc_id": s["doc_id"], "strict": s.get("exact_match,strict-match", s.get("exact_match")),
              "flexible": s.get("exact_match,flexible-extract")} for s in res["samples"]["gsm8k_cot"]]
    write_json(args.out, {**run_meta(args), "mode": args.mode, "set": args.set, "n_feats": len(ids), "layer": args.layer,
                          "metrics": {k: v for k, v in r.items() if isinstance(v, (int, float))}, "items": items})
    print(args.mode, args.set, {k: round(v, 4) for k, v in r.items() if isinstance(v, (int, float)) and "stderr" not in k})


if __name__ == "__main__":
    main()
