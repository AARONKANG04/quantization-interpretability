"""Per-token metrics of a perturbed model against the bf16 reference (plan E1.4), one parquet per token set.

usage: python scripts/phase1_token_dump.py --ckpt-b /data/ckpt/q1 --name q1 --tokens /data/kl/kl_2m.pt \
         [--model-a google/gemma-3-4b-pt] [--batch 4] [--out results/phase1/token_dump]
Columns: kl, top1_agree, entropy_a, target, position, doc_id, source, token_class, position_bucket.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.metrics import paired_forward_metrics, position_bucket, token_class  # noqa: E402
from qi.models import load_gemma3_text  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--model-a", default="google/gemma-3-4b-pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="results/phase1/token_dump")
    args = ap.parse_args()

    data = torch.load(args.tokens)
    ids, doc_id, source = data["input_ids"], data["doc_id"], data["source"]
    model_a, tok = load_gemma3_text(args.model_a)
    model_b, _ = load_gemma3_text(args.ckpt_b)
    vocab_class = [token_class(t if t is not None else "") for t in tok.convert_ids_to_tokens(list(range(len(tok))))]
    frames = []
    for s in range(0, ids.shape[0], args.batch):
        x = ids[s : s + args.batch].to(model_a.device, torch.long)
        m = paired_forward_metrics(model_a, model_b, x)
        B, Lm1 = x.shape[0], x.shape[1] - 1
        tgt = m["target"].cpu().numpy()
        frames.append(pd.DataFrame({
            "kl": m["kl"].cpu().numpy(), "top1_agree": m["top1_agree"].cpu().numpy(),
            "entropy_a": m["entropy_a"].cpu().numpy(), "target": tgt, "position": m["position"].cpu().numpy(),
            "doc_id": doc_id[s : s + B].repeat_interleave(Lm1).numpy(), "source": source[s : s + B].repeat_interleave(Lm1).numpy(),
        }))
        if (s // args.batch) % 20 == 0:
            print(f"{s + B}/{ids.shape[0]} sequences", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df["token_class"] = pd.Categorical([vocab_class[t] if t < len(vocab_class) else "other" for t in df["target"]])
    df["position_bucket"] = pd.Categorical([position_bucket(int(p)) for p in df["position"]])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.name}_{Path(args.tokens).stem}"
    df.to_parquet(out / f"{stem}.parquet", index=False)
    summary = {
        "n_tokens": int(len(df)), "mean_kl": float(df.kl.mean()), "median_kl": float(df.kl.median()),
        "p90_kl": float(df.kl.quantile(0.9)), "top1_agree": float(df.top1_agree.mean()),
        "by_class": df.groupby("token_class", observed=True).agg(n=("kl", "size"), median_kl=("kl", "median"),
                                                                p90_kl=("kl", lambda v: float(v.quantile(0.9))),
                                                                flip_rate=("top1_agree", lambda v: float(1 - v.mean()))).to_dict("index"),
        "by_position": df.groupby("position_bucket", observed=True).agg(n=("kl", "size"), median_kl=("kl", "median"),
                                                                       flip_rate=("top1_agree", lambda v: float(1 - v.mean()))).to_dict("index"),
    }
    write_json(out / f"{stem}_summary.json", {**run_meta(args), **summary})
    print({k: summary[k] for k in ("n_tokens", "mean_kl", "median_kl", "top1_agree")})


if __name__ == "__main__":
    main()
