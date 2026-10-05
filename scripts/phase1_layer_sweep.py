"""Restore-one-layer (or quantize-one-layer) sweep: per-layer KL against a cached top-k bf16 reference.

usage: python scripts/phase1_layer_sweep.py --mode restore_one --layers 0-5 --tokens /data/kl/kl_500k.pt \
         --ref /data/kl/ref_kl_500k_top256.pt --out results/phase1/layer_sweep [--scheme nvfp4_rtn] [--batch 8]
Build the reference once (any GPU): python scripts/phase1_layer_sweep.py --build-ref --tokens ... --ref ...
Shard --layers across GPUs with scripts/remote/run.sh; each layer writes its own JSON, so reruns are incremental.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.metrics import bucketed_kl_from_reference, topk_reference  # noqa: E402
from qi.models import load_gemma3_text  # noqa: E402
from qi.quant import apply_scheme, iter_quantizable  # noqa: E402


def _ints(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


@torch.no_grad()
def logits_iter(model, input_ids, batch):
    for s in range(0, input_ids.shape[0], batch):
        x = input_ids[s : s + batch].to(model.device, torch.long)
        yield model(input_ids=x).logits[:, :-1].reshape(-1, model.config.vocab_size)


@torch.no_grad()
def build_reference(model, input_ids, batch, k, path):
    logp, idx, tail = [], [], []
    for lg in logits_iter(model, input_ids, batch):
        a, b, c = topk_reference(lg, k=k)
        logp.append(a.cpu()); idx.append(b.cpu()); tail.append(c.cpu())
    torch.save({"logp": torch.cat(logp), "idx": torch.cat(idx), "log_tail": torch.cat(tail), "k": k}, path)
    print("reference saved", path)


@torch.no_grad()
def sweep_kl(model, input_ids, batch, ref):
    kls, agrees, pos = [], [], 0
    for lg in logits_iter(model, input_ids, batch):
        n = lg.shape[0]
        m = bucketed_kl_from_reference(ref["logp"][pos : pos + n].to(lg.device), ref["idx"][pos : pos + n].to(lg.device),
                                       ref["log_tail"][pos : pos + n].to(lg.device), lg)
        kls.append(m["kl"].cpu()); agrees.append(m["top1_agree"].cpu()); pos += n
    kl = torch.cat(kls); ag = torch.cat(agrees)
    return {"mean_kl": float(kl.mean()), "median_kl": float(kl.median()), "p90_kl": float(kl.quantile(0.9)),
            "top1_agree": float(ag.float().mean()), "n_tokens": int(kl.numel())}, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["restore_one", "quantize_one", "all", "none"], default="restore_one")
    ap.add_argument("--layers", default="0-33")
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default="results/phase1/layer_sweep")
    ap.add_argument("--scheme", default="nvfp4_rtn")
    ap.add_argument("--model", default="google/gemma-3-4b-pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--build-ref", action="store_true")
    ap.add_argument("--save-token-kl", action="store_true")
    args = ap.parse_args()

    data = torch.load(args.tokens)
    input_ids = data["input_ids"]
    model, _ = load_gemma3_text(args.model)
    if args.build_ref:
        build_reference(model, input_ids, args.batch, args.topk, args.ref)
        return
    ref = torch.load(args.ref)
    originals = {name: mod.weight.data.to("cpu", copy=True) for name, mod in iter_quantizable(model)}

    def restore_all():
        for name, mod in iter_quantizable(model):
            mod.weight.data.copy_(originals[name])

    configs = {"all": [None], "none": [None]}.get(args.mode, _ints(args.layers))
    for li in configs:
        t0 = time.time()
        restore_all()
        if args.mode == "restore_one":
            stats = apply_scheme(model, args.scheme, exclude_layers=[li])
        elif args.mode == "quantize_one":
            stats = apply_scheme(model, args.scheme, layers=[li])
        elif args.mode == "all":
            stats = apply_scheme(model, args.scheme)
        else:
            stats = {}
        metrics, kl = sweep_kl(model, input_ids, args.batch, ref)
        tag = f"{args.mode}_{li if li is not None else 'x'}"
        payload = {**run_meta(args), "mode": args.mode, "layer": li, "scheme": args.scheme, **metrics,
                   "n_matrices_quantized": len(stats), "seconds": time.time() - t0}
        write_json(Path(args.out) / f"{tag}.json", payload)
        if args.save_token_kl:
            torch.save(kl, Path(args.out) / f"{tag}_token_kl.pt")
        print(tag, json.dumps(metrics))


if __name__ == "__main__":
    main()
