"""E2.4: train a BatchTopK SAE on resid_post of one layer of one model (bf16 or a QDQ checkpoint), activations
generated on the fly from FineWeb-Edu, then convert to JumpReLU and save.

usage: python scripts/phase2_train_sae.py --model-ckpt /data/ckpt/bf16 --name bf16 --layer 17 \
         [--width 32768] [--k 64] [--tokens 100000000] [--batch 4096] [--out-dir /data/saes]
Checkpoints every --ckpt-every steps so a preempted run resumes with --resume.
"""
import argparse
import json
import time
from pathlib import Path

import torch

from _common import load_env, run_meta, write_json

load_env()
from qi.data import ActivationStream, fineweb_edu_texts, stream_token_chunks  # noqa: E402
from qi.models import load_gemma3_text  # noqa: E402
from qi.sae.train import BatchTopKSAE, train_sae  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-ckpt", required=True); ap.add_argument("--name", required=True); ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--width", type=int, default=32768); ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=100_000_000); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--buffer-tokens", type=int, default=262_144); ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--out-dir", default="/data/saes"); ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.name}_L{args.layer}_w{args.width}_k{args.k}"
    model, tok = load_gemma3_text(args.model_ckpt)
    chunks = stream_token_chunks(tok, fineweb_edu_texts(), seq_len=args.seq_len)
    stream = ActivationStream(model, args.layer, chunks, batch_seqs=8, buffer_tokens=args.buffer_tokens, batch_size=args.batch)
    sae = BatchTopKSAE(model.config.hidden_size, args.width, args.k).to("cuda")
    steps = args.tokens // args.batch
    start = 0
    ck = out_dir / f"{tag}_ckpt.pt"
    if args.resume and ck.exists():
        state = torch.load(ck); sae.load_state_dict(state["sae"]); start = state["step"]; print("resumed at step", start)
    t0 = time.time()
    log_path = Path("results/phase2/sae_train"); log_path.mkdir(parents=True, exist_ok=True)
    history = []

    def log(rec):
        rec = {**rec, "elapsed_s": round(time.time() - t0, 1)}
        history.append(rec); print(json.dumps(rec), flush=True)
        if rec["step"] % args.ckpt_every == 0:
            torch.save({"sae": sae.state_dict(), "step": rec["step"]}, ck)

    train_sae(sae, stream, steps=steps - start, lr=args.lr, log_every=50, log=lambda r: log({**r, "step": r["step"] + start}))
    j = sae.to_jumprelu()
    j.save(out_dir / f"{tag}.pt")
    write_json(log_path / f"{tag}.json", {**run_meta(args), "steps": steps, "history": history, "sae_file": str(out_dir / f"{tag}.pt")})
    print("saved", out_dir / f"{tag}.pt")


if __name__ == "__main__":
    main()
