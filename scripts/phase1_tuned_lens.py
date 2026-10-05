"""Tuned lens for Gemma 3 4B (plan E1.2): one affine translator per layer, trained so that
final_norm(translator_l(resid_post_l)) -> lm_head matches the model's own final distribution (KL).
Then applied to the bf16 and the quantized model on the same tokens: per-layer KL between lens-decoded
predictions, and the divergence onset layer.

usage: train: python scripts/phase1_tuned_lens.py train --tokens /data/kl/lens_10m.pt --out /data/lens/gemma-3-4b-pt.pt
       apply: python scripts/phase1_tuned_lens.py apply --lens /data/lens/gemma-3-4b-pt.pt --ckpt-b /data/ckpt/q1 \
                --tokens /data/kl/kl_500k.pt --out results/phase1/lens_q1.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from _common import load_env, run_meta, write_json

load_env()
from qi.models import ResidualCapture, decoder_layers, load_gemma3_text  # noqa: E402


class Lens(nn.Module):
    def __init__(self, n_layers: int, d: int):
        super().__init__()
        self.maps = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
        for m in self.maps:
            nn.init.eye_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        return self.maps[layer](h)


def decode(model, h):
    """final norm + unembed -> logits (fp32)."""
    inner = model.model
    return model.lm_head(inner.norm(h)).float()


def train(args):
    model, _ = load_gemma3_text(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    lens = Lens(n_layers, d).to(model.device, torch.float32)
    opt = torch.optim.Adam(lens.parameters(), lr=args.lr)
    data = torch.load(args.tokens)["input_ids"]
    steps = 0
    t0 = time.time()
    for s in range(0, data.shape[0], args.batch):
        x = data[s : s + args.batch].to(model.device, torch.long)
        with torch.no_grad(), ResidualCapture(model) as cap:
            teacher = model(input_ids=x).logits
            B, L, V = teacher.shape
            pos = torch.randperm(B * L, device=x.device)[: args.positions]
            t_logp = teacher.reshape(-1, V)[pos].float().log_softmax(-1)
            t_p = t_logp.exp()
        total = 0.0
        for li in range(n_layers):
            h = cap.acts[li].reshape(-1, d)[pos].float()
            s_logp = decode(model, lens(li, h).to(model.dtype)).log_softmax(-1)
            loss = (t_p * (t_logp - s_logp)).sum(-1).mean()
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        steps += 1
        if steps % 20 == 0:
            print(f"step {steps} seqs {s + B}/{data.shape[0]} mean-layer KL {total / n_layers:.4f} ({time.time() - t0:.0f}s)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": lens.state_dict(), "n_layers": n_layers, "d": d, "model": args.model, "steps": steps}, args.out)
    print("saved", args.out)


@torch.no_grad()
def apply(args):
    model_a, _ = load_gemma3_text(args.model)
    model_b, _ = load_gemma3_text(args.ckpt_b)
    ck = torch.load(args.lens)
    lens = Lens(ck["n_layers"], ck["d"]).to(model_a.device, torch.float32)
    lens.load_state_dict(ck["state_dict"])
    data = torch.load(args.tokens)["input_ids"]
    n_layers = ck["n_layers"]
    kl_ab = torch.zeros(n_layers, dtype=torch.float64)   # KL(lens(a) || lens(b)) per layer
    kl_a_final = torch.zeros(n_layers, dtype=torch.float64)  # how good the lens is on a: KL(final_a || lens_l(a))
    count = 0
    for s in range(0, data.shape[0], args.batch):
        x = data[s : s + args.batch].to(model_a.device, torch.long)
        with ResidualCapture(model_a) as ca:
            la = model_a(input_ids=x).logits
        with ResidualCapture(model_b) as cb:
            model_b(input_ids=x)
        B, L, V = la.shape
        pos = torch.randperm(B * L, device=x.device)[: args.positions]
        fa = la.reshape(-1, V)[pos].float().log_softmax(-1); pa = fa.exp()
        for li in range(n_layers):
            ha = ca.acts[li].reshape(-1, ck["d"])[pos].float(); hb = cb.acts[li].reshape(-1, ck["d"])[pos].float()
            za = decode(model_a, lens(li, ha).to(model_a.dtype)).log_softmax(-1)
            zb = decode(model_a, lens(li, hb).to(model_a.dtype)).log_softmax(-1)  # same lens, same unembed, by design
            qa = za.exp()
            kl_ab[li] += (qa * (za - zb)).sum(-1).sum().item()
            kl_a_final[li] += (pa * (fa - za)).sum(-1).sum().item()
        count += len(pos)
        if s == 0 or (s // args.batch) % 20 == 0:
            print(f"{s + B}/{data.shape[0]} sequences", flush=True)
    kl_ab /= count; kl_a_final /= count
    final = float(kl_ab[-1])
    onset = next((i for i in range(n_layers) if kl_ab[i] >= 0.5 * final), n_layers - 1)
    out = {**run_meta(args), "n_positions": count, "kl_lens_a_vs_b": kl_ab.tolist(), "kl_final_a_vs_lens_a": kl_a_final.tolist(),
           "divergence_onset_layer": onset, "final_layer_kl": final}
    write_json(args.out, out)
    print(json.dumps({"onset": onset, "final": final, "per_layer": [round(v, 4) for v in kl_ab.tolist()]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--tokens", required=True); t.add_argument("--out", required=True)
    t.add_argument("--model", default="google/gemma-3-4b-pt"); t.add_argument("--batch", type=int, default=4)
    t.add_argument("--positions", type=int, default=512); t.add_argument("--lr", type=float, default=1e-3)
    a = sub.add_parser("apply"); a.add_argument("--lens", required=True); a.add_argument("--ckpt-b", required=True)
    a.add_argument("--tokens", required=True); a.add_argument("--out", required=True)
    a.add_argument("--model", default="google/gemma-3-4b-pt"); a.add_argument("--batch", type=int, default=2)
    a.add_argument("--positions", type=int, default=512)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else apply(args)


if __name__ == "__main__":
    main()
