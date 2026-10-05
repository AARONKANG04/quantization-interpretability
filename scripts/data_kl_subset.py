"""Build the held-out token sets (configs/data/kl_set.yaml) as [N, seq_len] int32 tensors under /data/kl.

usage: python scripts/data_kl_subset.py --set kl_500k [--out-dir /data/kl]
Sources are streamed from the Hub; GSM8K train is rendered as "question\\nanswer"; documents are concatenated with
EOS and chunked, and a doc id per chunk is kept for the cluster bootstrap.
"""
import argparse
from pathlib import Path

import torch
import yaml
from datasets import load_dataset

from _common import ROOT, load_env

load_env()

SOURCES = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
    "wikitext2": ("wikitext", "wikitext-2-raw-v1", "test", "text"),
    "gsm8k_train": ("openai/gsm8k", "main", "train", None),
    "gsm8k_test": ("openai/gsm8k", "main", "test", None),
    "code_python": ("codeparrot/codeparrot-clean", None, "train", "content"),  # ungated Python corpus
    "pg19": ("emozilla/pg19", None, "test", "text"),  # parquet mirror; deepmind/pg19 is a script dataset
}


def stream(name):
    repo, cfg, split, field = SOURCES[name]
    ds = load_dataset(repo, cfg, split=split, streaming=True)
    for row in ds:
        if name in ("gsm8k_train", "gsm8k_test"):
            yield row["question"] + "\n" + row["answer"]
        else:
            t = row[field]
            if t and len(t.strip()) > 50:
                yield t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--out-dir", default="/data/kl")
    ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/data/kl_set.yaml").read_text())
    spec = cfg["sets"][args.set]
    L = spec.get("seq_len", cfg["seq_len"])
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    chunks, doc_ids, src_ids = [], [], []
    doc_counter = 0
    for si, (src, frac) in enumerate(spec["sources"].items()):
        budget = int(spec["tokens"] * frac)
        got = 0
        buf, buf_doc = [], None  # documents are packed back to back with EOS so short items (GSM8K) are kept
        for text in stream(src):
            ids = tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            if buf_doc is None:
                buf_doc = doc_counter
            buf.extend(ids)
            doc_counter += 1
            while len(buf) >= L:
                chunks.append(torch.tensor(buf[:L], dtype=torch.int32))
                doc_ids.append(buf_doc); src_ids.append(si)
                buf = buf[L:]; buf_doc = doc_counter if buf else None
                got += L
                if got >= budget:
                    break
            if got >= budget:
                break
        print(f"{src}: {got} tokens, {doc_counter} docs so far", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": torch.stack(chunks), "doc_id": torch.tensor(doc_ids), "source": torch.tensor(src_ids),
                "sources": list(spec["sources"]), "seq_len": L}, out / f"{args.set}.pt")
    print("saved", out / f"{args.set}.pt", "shape", (len(chunks), L))


if __name__ == "__main__":
    main()
