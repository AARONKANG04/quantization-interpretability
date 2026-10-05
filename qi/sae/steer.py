"""Intervention hooks for plan E2.5. All return fn(h) -> h' for ResidualPatch at one layer.

gain_correction: deployable; rescales selected features of the running model's own activations.
mean_shift:      deployable; adds a fixed bias.
oracle_patch:    diagnostic; replaces selected features' contribution with the bf16 twin's, which runs in
                 lockstep through TwinRunner (its own KV cache, same tokens, teacher-forced on model_b's outputs).
"""
from __future__ import annotations

import torch

from ..models import ResidualCapture
from .jumprelu import JumpReLUSAE


def gain_correction(sae: JumpReLUSAE, feats: torch.Tensor, gains: torch.Tensor):
    W = sae.W_dec[feats]  # [s, d]

    def fn(h):
        f = sae.encode(h, feats)                      # [..., s] on the running model's activations
        delta = ((gains - 1.0) * f) @ W
        return h + delta.to(h.dtype)
    return fn


def mean_shift(bias: torch.Tensor):
    def fn(h):
        return h + bias.to(h.dtype)
    return fn


class TwinRunner:
    """Runs model_a on exactly the inputs model_b receives (prefill and each decode step) and exposes model_a's
    resid_post at `layer` for the current call in .hidden. Attach with `with TwinRunner(model_b, model_a, layer):`."""

    def __init__(self, model_b, model_a, layer: int):
        self.model_b, self.model_a, self.layer = model_b, model_a, layer
        self.cache = None
        self.hidden = None
        self._handle = None

    def _pre_hook(self, module, args, kwargs):
        from transformers import DynamicCache

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        past = kwargs.get("past_key_values")
        if past is None or self.cache is None or input_ids.shape[1] > 1:
            self.cache = DynamicCache()
        fwd = {k: v for k, v in kwargs.items() if k in ("attention_mask", "position_ids", "cache_position")}
        with torch.no_grad(), ResidualCapture(self.model_a, layers=[self.layer]) as cap:
            self.model_a(input_ids=input_ids, past_key_values=self.cache, use_cache=True, **fwd)
        self.hidden = cap.acts[self.layer]

    def __enter__(self):
        self._handle = self.model_b.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()


def oracle_patch(sae: JumpReLUSAE, feats: torch.Tensor, twin: TwinRunner):
    W = sae.W_dec[feats]

    def fn(h):
        ha = twin.hidden
        fa = sae.encode(ha, feats)
        fb = sae.encode(h, feats)
        return h + ((fa - fb) @ W).to(h.dtype)
    return fn


def full_residual_patch(twin: TwinRunner):
    def fn(h):
        return twin.hidden.to(h.dtype)
    return fn
