"""Per-token distance metrics between a reference model (a) and a perturbed model (b).

Everything is chunked over tokens and done in fp32 so the 262k-vocabulary logits are never materialized
in full for a whole sequence set. The reference can be compressed to its top-k log-probs plus one tail
bucket; the KL against that bucketing is a lower bound on the true KL (coarse-graining never increases KL)
and exact on the top-k support, which is what the layer sweeps need.
"""
from __future__ import annotations

import torch

_EPS = 1e-7


@torch.no_grad()
def token_metrics_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, chunk: int = 1024) -> dict:
    """logits_* are [T, V]. Returns {'kl': KL(a||b) [T], 'top1_agree': bool [T], 'entropy_a': [T]} in fp32."""
    T = logits_a.shape[0]
    dev = logits_a.device
    kl = torch.empty(T, dtype=torch.float32, device=dev)
    ent = torch.empty(T, dtype=torch.float32, device=dev)
    agree = torch.empty(T, dtype=torch.bool, device=dev)
    for s in range(0, T, chunk):
        la = logits_a[s : s + chunk].float().log_softmax(-1)
        lb = logits_b[s : s + chunk].float().log_softmax(-1)
        pa = la.exp()
        kl[s : s + chunk] = (pa * (la - lb)).sum(-1)
        ent[s : s + chunk] = -(pa * la).sum(-1)
        agree[s : s + chunk] = la.argmax(-1) == lb.argmax(-1)
    return {"kl": kl, "top1_agree": agree, "entropy_a": ent}


@torch.no_grad()
def topk_reference(logits: torch.Tensor, k: int = 256, chunk: int = 1024):
    """Compress reference logits [T, V] to (logp [T, k] fp32, idx [T, k] int32, log_tail [T] fp32), sorted by prob."""
    T, V = logits.shape
    k = min(k, V)
    dev = logits.device
    logp = torch.empty(T, k, dtype=torch.float32, device=dev)
    idx = torch.empty(T, k, dtype=torch.int32, device=dev)
    log_tail = torch.empty(T, dtype=torch.float32, device=dev)
    for s in range(0, T, chunk):
        la = logits[s : s + chunk].float().log_softmax(-1)
        v, i = la.topk(k, dim=-1)
        logp[s : s + chunk] = v
        idx[s : s + chunk] = i.to(torch.int32)
        rest = (1.0 - v.exp().sum(-1)).clamp(min=0.0)
        log_tail[s : s + chunk] = torch.where(rest > _EPS, rest.clamp(min=_EPS).log(), torch.full_like(rest, -float("inf")))
    return logp, idx, log_tail


@torch.no_grad()
def bucketed_kl_from_reference(
    ref_logp: torch.Tensor, ref_idx: torch.Tensor, ref_log_tail: torch.Tensor, logits_b: torch.Tensor, chunk: int = 1024
) -> dict:
    """KL(a||b) with both distributions bucketed into the reference's top-k atoms plus one tail bucket.
    Returns {'kl': [T], 'top1_agree': [T]} where top-1 agreement compares b's argmax with the reference's."""
    T = logits_b.shape[0]
    dev = logits_b.device
    kl = torch.empty(T, dtype=torch.float32, device=dev)
    agree = torch.empty(T, dtype=torch.bool, device=dev)
    for s in range(0, T, chunk):
        lb = logits_b[s : s + chunk].float().log_softmax(-1)
        idx = ref_idx[s : s + chunk].long()
        la_top = ref_logp[s : s + chunk]
        lb_top = lb.gather(-1, idx)
        pa_top = la_top.exp()
        rest_b = (1.0 - lb_top.exp().sum(-1)).clamp(min=_EPS)
        lt_a = ref_log_tail[s : s + chunk]
        pa_tail = lt_a.exp()
        tail_term = torch.where(pa_tail > _EPS, pa_tail * (lt_a - rest_b.log()), torch.zeros_like(pa_tail))
        kl[s : s + chunk] = (pa_top * (la_top - lb_top)).sum(-1) + tail_term
        agree[s : s + chunk] = lb.argmax(-1) == idx[:, 0]
    return {"kl": kl, "top1_agree": agree}


@torch.no_grad()
def paired_forward_metrics(model_a, model_b, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                           chunk: int = 1024) -> dict:
    """Run both models on the same [B, L] batch and return next-token metrics for positions 0..L-2.
    Output tensors are flattened [B*(L-1)] in fp32/bool, plus 'target' (the token being predicted) and
    'position' (its position in the sequence)."""
    out_a = model_a(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    out_b = model_b(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    B, Lm1, V = out_a.shape
    m = token_metrics_from_logits(out_a.reshape(-1, V), out_b.reshape(-1, V), chunk=chunk)
    m["target"] = input_ids[:, 1:].reshape(-1)
    m["position"] = torch.arange(1, Lm1 + 1, device=input_ids.device).repeat(B)
    if attention_mask is not None:
        m["valid"] = attention_mask[:, 1:].reshape(-1).bool()
    return m
