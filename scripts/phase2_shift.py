"""E2.2: streaming feature-shift accounting between the bf16 model and a perturbed model at one SAE layer.

usage: python scripts/phase2_shift.py --layer 17 --ckpt-b /data/ckpt/q1 --name q1 \
         --tokens /data/kl/kl_2m.pt /data/kl/lens_10m.pt /data/kl/gsm8k_test.pt --batch 4 --out results/phase2/shift
Writes <out>/<name>_L<layer>_summary.json (concentration, shares, labels) and /data/shift/<name>_L<layer>.pt
(per-feature statistics and top-20 example pointers; sequence indices are global over the token files in order).
"""
import argparse
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.models import ResidualCapture, load_gemma3_text  # noqa: E402
from qi.sae.gemma_scope import load_gemma_scope2  # noqa: E402
from qi.sae.jumprelu import JumpReLUSAE  # noqa: E402
from qi.sae.shift import ShiftAccumulator  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", default="65k"); ap.add_argument("--l0", default="medium"); ap.add_argument("--variant", default="4b-pt")
    ap.add_argument("--sae-file", default=None, help="use a saved JumpReLUSAE instead of Gemma Scope 2")
    ap.add_argument("--model-a", default="google/gemma-3-4b-pt"); ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--tokens", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--out", default="results/phase2/shift"); ap.add_argument("--store-dir", default="/data/shift")
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()

    sae = JumpReLUSAE.load(args.sae_file, device="cuda") if args.sae_file else load_gemma_scope2(args.layer, args.width, args.l0, variant=args.variant, device="cuda")
    model_a, _ = load_gemma3_text(args.model_a)
    model_b, _ = load_gemma3_text(args.ckpt_b)
    acc = ShiftAccumulator(sae.W_dec)
    seq_offset = 0
    files = []
    for tf in args.tokens:
        ids = torch.load(tf)["input_ids"]
        if args.max_seqs:
            ids = ids[: args.max_seqs]
        files.append({"file": tf, "n_seqs": int(ids.shape[0]), "seq_offset": seq_offset})
        for s in range(0, ids.shape[0], args.batch):
            x = ids[s : s + args.batch].to("cuda", torch.long)
            with ResidualCapture(model_a, layers=[args.layer]) as ca:
                model_a(input_ids=x)
            with ResidualCapture(model_b, layers=[args.layer]) as cb:
                model_b(input_ids=x)
            B, L, d = ca.acts[args.layer].shape
            ha = ca.acts[args.layer].reshape(-1, d); hb = cb.acts[args.layer].reshape(-1, d)
            seq_ids = (torch.arange(B, device="cuda") + seq_offset + s).repeat_interleave(L)
            pos = torch.arange(L, device="cuda").repeat(B)
            for c in range(0, ha.shape[0], args.chunk):
                a, b = ha[c : c + args.chunk].float(), hb[c : c + args.chunk].float()
                fa, fb = sae.encode(a), sae.encode(b)
                acc.update(fa, fb, a, b, sae.decode(fa), sae.decode(fb), seq_ids[c : c + args.chunk], pos[c : c + args.chunk])
            if (s // args.batch) % 25 == 0:
                print(f"{tf}: {s + B}/{ids.shape[0]} seqs, {acc.n} tokens", flush=True)
        seq_offset += int(ids.shape[0])
    res = acc.finalize()
    per_feature = res.pop("per_feature")
    Path(args.store_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"per_feature": per_feature, "files": files, "layer": args.layer, "sae": args.sae_file or f"gemma-scope-2-{args.variant} L{args.layer} {args.width} {args.l0}"},
               Path(args.store_dir) / f"{args.name}_L{args.layer}.pt")
    write_json(Path(args.out) / f"{args.name}_L{args.layer}_summary.json", {**run_meta(args), **res, "files": files})
    print({k: res[k] for k in ("n_tokens", "shares", "concentration", "label_counts")})


if __name__ == "__main__":
    main()
