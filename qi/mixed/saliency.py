"""Phase 3: per-channel saliency scores, budgeted selection, memory accounting.

Candidates are output rows (writers into the residual stream) and input columns (readers) of every quantizable
matrix. A selection is {matrix: {"rows": [...], "cols": [...]}} and plugs into qi.quant.apply_scheme(protect=...).

Scores (higher = protect first):
  fg_proj  feature-guided, no gradients: alignment of a row/column with the implicated SAE decoder directions
           (through Gemma 3's RMSNorm gains) times that channel's NVFP4 error norm
  mag      AWQ-style: mean |input activation| of the column times the column's error norm (columns only)
  klg      first-order KL: -sum(grad_KL * dW) over the channel, the predicted KL drop from restoring it
  rand     uniform random
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn

from ..quant.apply import fused_global_amax, iter_quantizable, layer_index
from ..quant.nvfp4 import nvfp4_qdq

WRITERS = ("o_proj", "down_proj")
READERS = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


@dataclass
class Candidate:
    matrix: str
    kind: str  # "row" | "col"
    index: int
    n_weights: int
    score: float


def quant_errors(model: nn.Module, block: int = 16) -> dict[str, torch.Tensor]:
    """{matrix name: W_q - W} for every quantizable matrix (bf16 model, exact llm-compressor arithmetic)."""
    shared = fused_global_amax(model)
    out = {}
    for name, mod in iter_quantizable(model):
        dq, _ = nvfp4_qdq(mod.weight.data, block=block, with_stats=False, global_amax=shared.get(name))
        out[name] = (dq.float() - mod.weight.data.float())
    return out


def norm_gains(model: nn.Module) -> dict[str, torch.Tensor]:
    """Per-dimension (1 + w) gains of the RMSNorms that sit between the residual stream and each matrix."""
    gains = {}
    for li, layer in enumerate(model.model.layers):
        g = {}
        g["o_proj"] = 1.0 + layer.post_attention_layernorm.weight.detach().float()
        g["down_proj"] = 1.0 + layer.post_feedforward_layernorm.weight.detach().float()
        pre_attn = 1.0 + layer.input_layernorm.weight.detach().float()
        pre_mlp = 1.0 + layer.pre_feedforward_layernorm.weight.detach().float()
        for s in ("q_proj", "k_proj", "v_proj"):
            g[s] = pre_attn
        for s in ("gate_proj", "up_proj"):
            g[s] = pre_mlp
        for s, v in g.items():
            gains[f"model.layers.{li}.{'self_attn' if s in ('q_proj', 'k_proj', 'v_proj', 'o_proj') else 'mlp'}.{s}"] = v
    return gains


@torch.no_grad()
def fg_proj_scores(model: nn.Module, errors: dict[str, torch.Tensor], directions: torch.Tensor, weights: torch.Tensor,
                   sae_layer: int) -> dict[str, dict[str, torch.Tensor]]:
    """directions [s, d] decoder rows of the implicated features at `sae_layer`, weights [s] (e.g. energy share)."""
    gains = norm_gains(model)
    align_raw = (weights[:, None] * directions.abs()).sum(0)  # [d]
    scores = {}
    for name, dW in errors.items():
        li = layer_index(name)
        suffix = name.split(".")[-1]
        g = gains[name].to(dW.device)
        align = align_raw.to(dW.device) * g
        rows = torch.zeros(dW.shape[0], device=dW.device)
        cols = torch.zeros(dW.shape[1], device=dW.device)
        if suffix in WRITERS and li <= sae_layer:
            rows = align[: dW.shape[0]] * dW.norm(dim=1)
        if suffix in READERS and li > sae_layer:
            cols = align[: dW.shape[1]] * dW.norm(dim=0)
        scores[name] = {"rows": rows.cpu(), "cols": cols.cpu()}
    return scores


class InputStats:
    """Mean |x| per input channel of every quantizable linear, from calibration forwards."""

    def __init__(self, model: nn.Module):
        self.sums: dict[str, torch.Tensor] = {}
        self.n: dict[str, int] = {}
        self._handles = []
        for name, mod in iter_quantizable(model):
            self._handles.append(mod.register_forward_pre_hook(self._hook(name)))

    def _hook(self, name):
        def fn(_mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            self.sums[name] = self.sums.get(name, 0) + x.abs().sum(0)
            self.n[name] = self.n.get(name, 0) + x.shape[0]
        return fn

    def remove(self):
        for h in self._handles:
            h.remove()

    def mean_abs(self) -> dict[str, torch.Tensor]:
        return {k: v / self.n[k] for k, v in self.sums.items()}


@torch.no_grad()
def mag_scores(errors: dict[str, torch.Tensor], mean_abs: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    scores = {}
    for name, dW in errors.items():
        cols = mean_abs[name].to(dW.device) * dW.norm(dim=0)
        scores[name] = {"rows": torch.zeros(dW.shape[0]), "cols": cols.cpu()}
    return scores


def klg_scores(model_q: nn.Module, errors: dict[str, torch.Tensor], grads: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    """grads: {matrix: dKL/dW at the quantized weights}. Restoring a channel moves W_q by -dW, so the first-order
    KL change is -sum(grad * dW) over the channel; we score by the predicted KL decrease."""
    scores = {}
    for name, dW in errors.items():
        G = grads[name].to(dW.device).float()
        contrib = -(G * dW)
        scores[name] = {"rows": contrib.sum(1).cpu(), "cols": contrib.sum(0).cpu()}
    return scores


def kl_gradients(model_q: nn.Module, batches, ref_logp_fn, chunk: int = 1024) -> dict[str, torch.Tensor]:
    """Accumulate dKL(ref || q)/dW over calibration batches. ref_logp_fn(batch_index) -> (logp [T, k], idx [T, k],
    log_tail [T]) for the reference distribution of that batch (top-k bucketed, exact on the support)."""
    params = {name: mod.weight for name, mod in iter_quantizable(model_q)}
    for p in model_q.parameters():
        p.requires_grad_(False)
    for p in params.values():
        p.requires_grad_(True)
    grads = {name: torch.zeros_like(p, dtype=torch.float32, device="cpu") for name, p in params.items()}
    n = 0
    for bi, x in enumerate(batches):
        logits = model_q(input_ids=x).logits[:, :-1]
        V = logits.shape[-1]
        logits = logits.reshape(-1, V)
        logp, idx, log_tail = ref_logp_fn(bi)
        loss = 0.0
        for s in range(0, logits.shape[0], chunk):
            lb = logits[s : s + chunk].float().log_softmax(-1)
            i = idx[s : s + chunk].long(); la = logp[s : s + chunk]
            lb_top = lb.gather(-1, i)
            pa = la.exp()
            rest_b = (1.0 - lb_top.exp().sum(-1)).clamp(min=1e-7)
            lt = log_tail[s : s + chunk]; pt = lt.exp()
            tail = torch.where(pt > 1e-7, pt * (lt - rest_b.log()), torch.zeros_like(pt))
            loss = loss + ((pa * (la - lb_top)).sum(-1) + tail).sum()
        loss = loss / logits.shape[0]
        loss.backward()
        for name, p in params.items():
            grads[name] += p.grad.detach().float().cpu()
            p.grad = None
        n += 1
    for p in params.values():
        p.requires_grad_(False)
    return {k: v / max(n, 1) for k, v in grads.items()}


def random_scores(errors: dict[str, torch.Tensor], seed: int = 0) -> dict[str, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return {name: {"rows": torch.rand(dW.shape[0], generator=g), "cols": torch.rand(dW.shape[1], generator=g)} for name, dW in errors.items()}


def select_budget(scores: dict[str, dict[str, torch.Tensor]], shapes: dict[str, tuple[int, int]], budget_weights: int,
                  per_weight: bool = True) -> tuple[dict[str, dict[str, list[int]]], int]:
    """Greedy by score (per protected weight if per_weight) until the weight budget is reached.
    Returns (selection, protected weight count, counted as rows + cols, overlaps ignored)."""
    cands = []
    for name, sc in scores.items():
        out_f, in_f = shapes[name]
        r, c = sc["rows"], sc["cols"]
        if r.numel():
            cands.append((r / (in_f if per_weight else 1.0), "row", name, in_f))
        if c.numel():
            cands.append((c / (out_f if per_weight else 1.0), "col", name, out_f))
    all_scores = torch.cat([t for t, *_ in cands])
    meta = []
    for t, kind, name, nw in cands:
        meta.extend([(kind, name, i, nw) for i in range(t.numel())])
    order = all_scores.argsort(descending=True)
    sel: dict[str, dict[str, list[int]]] = {}
    used = 0
    for j in order.tolist():
        kind, name, i, nw = meta[j]
        if all_scores[j] <= 0:
            break
        if used + nw > budget_weights:
            continue
        sel.setdefault(name, {"rows": [], "cols": []})[f"{kind}s"].append(i)
        used += nw
        if used >= budget_weights:
            break
    return sel, used


NVFP4_BITS = 4.5
BF16_BITS = 16.0


def memory_bytes(total_weights: int, protected_weights: int) -> dict:
    extra = protected_weights * (BF16_BITS - NVFP4_BITS) / 8
    return {"total_weights": total_weights, "protected_weights": protected_weights,
            "protected_frac": protected_weights / total_weights,
            "bytes_nvfp4": total_weights * NVFP4_BITS / 8, "bytes_mixed": total_weights * NVFP4_BITS / 8 + extra,
            "extra_bytes": extra, "bytes_bf16": total_weights * BF16_BITS / 8}


def layer_fallback_bytes(layer_weights: int, n_layers: int) -> float:
    return n_layers * layer_weights * (BF16_BITS - NVFP4_BITS) / 8
