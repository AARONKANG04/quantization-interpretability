"""E2.1: load a Gemma Scope 2 SAE and measure reconstruction on bf16 activations and on a perturbed model's
activations, plus the cross-entropy change when the reconstruction is spliced back in.

usage: python scripts/phase2_validate_sae.py --layer 17 --width 65k --l0 medium --ckpt-b /data/ckpt/q1 \
         --tokens /data/kl/kl_500k.pt --n-seqs 48 --out results/phase2/sae_validation_L17.json
"""
import argparse

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.models import ResidualCapture, load_gemma3_text  # noqa: E402
from qi.sae.gemma_scope import load_gemma_scope2, splice_ce_delta  # noqa: E402
from qi.sae.jumprelu import reconstruction_stats  # noqa: E402


@torch.no_grad()
def capture(model, layer, ids, batch):
    parts = []
    for s in range(0, ids.shape[0], batch):
        x = ids[s : s + batch].to(model.device, torch.long)
        with ResidualCapture(model, layers=[layer]) as cap:
            model(input_ids=x)
        parts.append(cap.acts[layer].reshape(-1, cap.acts[layer].shape[-1]).float())
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", default="65k"); ap.add_argument("--l0", default="medium"); ap.add_argument("--variant", default="4b-pt")
    ap.add_argument("--model-a", default="google/gemma-3-4b-pt"); ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--tokens", default="/data/kl/kl_500k.pt"); ap.add_argument("--n-seqs", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ids = torch.load(args.tokens)["input_ids"][: args.n_seqs]
    sae = load_gemma_scope2(args.layer, args.width, args.l0, variant=args.variant, device="cuda")
    model_a, _ = load_gemma3_text(args.model_a)
    model_b, _ = load_gemma3_text(args.ckpt_b)
    out = {**run_meta(args), "sae": {"d_in": sae.d_in, "d_sae": sae.d_sae, "apply_b_dec_to_input": sae.apply_b_dec_to_input,
                                     "threshold_mean": float(sae.threshold.mean()), "dec_norm_mean": float(sae.dec_norms.mean())}}
    for name, model in (("a_bf16", model_a), ("b", model_b)):
        h = capture(model, args.layer, ids, args.batch)
        out[name] = {"recon": reconstruction_stats(sae, h), "resid_norm_mean": float(h.norm(dim=-1).mean()),
                     "splice": splice_ce_delta(model, sae, args.layer, ids[:4].to(model.device, torch.long))}
        del h
    # the SAE as a ruler across models: are the two models' activations equally well reconstructed?
    out["recon_gap"] = out["b"]["recon"]["variance_explained"] - out["a_bf16"]["recon"]["variance_explained"]
    write_json(args.out, out)
    print({k: (v["recon"] if isinstance(v, dict) and "recon" in v else v) for k, v in out.items() if k in ("a_bf16", "b", "recon_gap", "sae")})


if __name__ == "__main__":
    main()
