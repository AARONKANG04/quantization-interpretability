"""Top-shifted features: statistics and top activating snippets (decoded), for interpretation and auto-interp.

usage: python scripts/phase2_inspect_feats.py --shift /data/shift/q1_L17.pt --k 15 --out results/phase2/top_feats_q1_L17.json
"""
import argparse
import json

import torch

from _common import load_env, write_json

load_env()
LABELS = {0: "suppressed", 1: "amplified", 2: "newly_dead", 3: "newly_alive", 4: "stable", -1: "inactive"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True); ap.add_argument("--k", type=int, default=15); ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--examples", type=int, default=4); ap.add_argument("--tokenizer", default="google/gemma-3-4b-pt"); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    d = torch.load(args.shift)
    pf, files = d["per_feature"], d["files"]
    token_files = {f["file"]: None for f in files}

    def seq_tokens(gidx):
        for f in files:
            if f["seq_offset"] <= gidx < f["seq_offset"] + f["n_seqs"]:
                if token_files[f["file"]] is None:
                    token_files[f["file"]] = torch.load(f["file"])["input_ids"]
                return token_files[f["file"]][gidx - f["seq_offset"]], f["file"].split("/")[-1]
        return None, None

    n = pf["cnt_a"].numel()
    energy = pf["energy"]; tot = energy.sum()
    order = energy.argsort(descending=True)[: args.k]
    n_tok = max(float(pf["cnt_a"].max()), 1.0)  # not exact token count; use stored summary instead if present
    rows = []
    for i in order.tolist():
        ex = []
        for j in range(min(args.examples, pf["ex_val"].shape[1])):
            g, pos, val = int(pf["ex_seq"][i, j]), int(pf["ex_pos"][i, j]), float(pf["ex_val"][i, j])
            if g < 0:
                continue
            ids, src = seq_tokens(g)
            if ids is None:
                continue
            lo, hi = max(0, pos - args.window), min(len(ids), pos + 4)
            toks = tok.convert_ids_to_tokens(ids[lo:hi].tolist())
            text = "".join((f"[[{t}]]" if lo + k == pos else t) for k, t in enumerate(toks)).replace("▁", " ").replace("\n", "\\n")
            ex.append({"src": src, "pos": pos, "act": round(val, 1), "text": text})
        rows.append({"feature": i, "energy_share": float(energy[i] / tot), "cnt_a": int(pf["cnt_a"][i]), "cnt_b": int(pf["cnt_b"][i]),
                     "freq_ratio": float(pf["freq_ratio"][i]), "mean_act_a": float(pf["mean_act_a"][i]), "mean_act_b": float(pf["mean_act_b"][i]),
                     "corr": float(pf["corr"][i]), "peak_a": float(pf["peak_a"][i]), "label": LABELS[int(pf["label"][i])], "examples": ex})
    write_json(args.out, {"shift": args.shift, "layer": d["layer"], "rows": rows})
    for r in rows:
        print(f"f{r['feature']:6d} share {100*r['energy_share']:5.2f}%  cnt_a {r['cnt_a']:9d}  ratio {r['freq_ratio']:.2f}  act {r['mean_act_a']:.0f}->{r['mean_act_b']:.0f}  corr {r['corr']:.3f}  {r['label']}")
        for e in r["examples"][:2]:
            print(f"        ({e['src']} pos {e['pos']} act {e['act']}) {e['text'][:150]}")


if __name__ == "__main__":
    main()
