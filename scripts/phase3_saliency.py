"""E3.1 / E3.2 scores for every row and column of every quantizable matrix: fg_proj, mag, klg, rand.

usage: python scripts/phase3_saliency.py --feats results/phase2/feats_q1_L17.json --sae-layer 17 \
         [--set top] [--ckpt-b /data/ckpt/q1] [--calib-tokens /data/kl/kl_500k.pt] [--calib-seqs 64] \
         [--ref /data/kl/ref_kl_500k_top256.pt] [--skip-klg] --out /data/saliency/scores_L17.pt
"""
import argparse
import json
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.mixed.saliency import InputStats, fg_proj_scores, kl_gradients, klg_scores, mag_scores, quant_errors, random_scores  # noqa: E402
from qi.models import load_gemma3_text  # noqa: E402
from qi.quant import iter_quantizable  # noqa: E402
from qi.sae.gemma_scope import load_gemma_scope2  # noqa: E402
from qi.sae.jumprelu import JumpReLUSAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="google/gemma-3-4b-pt"); ap.add_argument("--ckpt-b", default="/data/ckpt/q1")
    ap.add_argument("--feats", required=True); ap.add_argument("--set", default="top"); ap.add_argument("--sae-layer", type=int, required=True)
    ap.add_argument("--sae-file", default=None); ap.add_argument("--width", default="65k"); ap.add_argument("--l0", default="medium")
    ap.add_argument("--calib-tokens", default="/data/kl/kl_500k.pt"); ap.add_argument("--calib-seqs", type=int, default=64)
    ap.add_argument("--ref", default="/data/kl/ref_kl_500k_top256.pt"); ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--skip-klg", action="store_true"); ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model_a, _ = load_gemma3_text(args.model_a)
    shapes = {n: tuple(m.weight.shape) for n, m in iter_quantizable(model_a)}
    print("computing NVFP4 errors for", len(shapes), "matrices", flush=True)
    errors = {}
    for name, dW in quant_errors(model_a).items():
        errors[name] = dW.cpu()
    # feature-guided
    feats_info = json.loads(Path(args.feats).read_text())
    ids = torch.tensor(feats_info[args.set], dtype=torch.long)
    sae = JumpReLUSAE.load(args.sae_file, device="cuda") if args.sae_file else load_gemma_scope2(args.sae_layer, args.width, args.l0, device="cuda")
    shift = torch.load(feats_info["shift"]) if "shift" in feats_info else None
    energy = torch.tensor([feats_info.get("energy", {}).get(str(int(i)), 1.0) for i in ids]) if "energy" in feats_info else torch.ones(len(ids))
    directions = sae.W_dec[ids.to("cuda")].float().cpu()
    weights = (energy / energy.sum()).float()
    fg = fg_proj_scores(model_a, {k: v for k, v in errors.items()}, directions, weights, args.sae_layer)
    # AWQ-style magnitude
    ids_tok = torch.load(args.calib_tokens)["input_ids"][: args.calib_seqs]
    stats = InputStats(model_a)
    with torch.no_grad():
        for s in range(0, ids_tok.shape[0], args.batch):
            model_a(input_ids=ids_tok[s : s + args.batch].to("cuda", torch.long))
    stats.remove()
    mag = mag_scores(errors, stats.mean_abs())
    rnd = random_scores(errors, seed=0)
    scores = {"fg_proj": fg, "mag": mag, "rand": rnd}
    del model_a, sae
    torch.cuda.empty_cache()
    if not args.skip_klg:
        model_q, _ = load_gemma3_text(args.ckpt_b)
        ref = torch.load(args.ref)
        Lm1 = ids_tok.shape[1] - 1

        def ref_fn(bi):
            s = bi * args.batch * Lm1; e = s + args.batch * Lm1
            return ref["logp"][s:e].to("cuda"), ref["idx"][s:e].to("cuda"), ref["log_tail"][s:e].to("cuda")

        batches = [ids_tok[s : s + args.batch].to("cuda", torch.long) for s in range(0, ids_tok.shape[0], args.batch)]
        grads = kl_gradients(model_q, batches, ref_fn)
        scores["klg"] = klg_scores(model_q, errors, grads)
        del model_q
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores, "shapes": shapes, "sae_layer": args.sae_layer, "set": args.set, "n_feats": int(len(ids))}, args.out)
    write_json(str(Path(args.out).with_suffix(".json")), {**run_meta(args), "methods": list(scores), "n_matrices": len(shapes), "n_feats": int(len(ids))})
    print("saved", args.out, "methods:", list(scores))


if __name__ == "__main__":
    main()
