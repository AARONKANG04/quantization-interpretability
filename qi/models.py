"""Gemma 3 text-only loading plus the two hook primitives every intervention uses:
ResidualCapture (read resid_post at chosen layers) and ResidualPatch (replace or edit it)."""
from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn


def load_gemma3_text(name: str = "google/gemma-3-4b-pt", dtype=torch.bfloat16, device_map="cuda",
                     attn_implementation: str = "sdpa"):
    """Load the text-only causal LM from the (multimodal) Gemma 3 checkpoint. Needs transformers >= 4.53."""
    from transformers import AutoTokenizer, Gemma3ForCausalLM

    tok = AutoTokenizer.from_pretrained(name)
    model = Gemma3ForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=device_map,
                                              attn_implementation=attn_implementation)
    model.eval()
    return model, tok


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    inner = getattr(model, "model", model)
    return inner.layers


def global_layer_indices(config) -> list[int]:
    """Gemma 3 global-attention layers: every `sliding_window_pattern`-th layer (5, 11, 17, 23, 29 for 4B)."""
    types = getattr(config, "layer_types", None)
    if types:
        return [i for i, t in enumerate(types) if t == "full_attention"]
    pattern = getattr(config, "sliding_window_pattern", 6)
    return [i for i in range(config.num_hidden_layers) if (i + 1) % pattern == 0]


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _with_hidden(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class ResidualCapture:
    """with ResidualCapture(model, layers=[9, 17]) as cap: model(...); cap.acts[9] -> [B, L, d] resid_post."""

    def __init__(self, model: nn.Module, layers: Iterable[int] | None = None, to_cpu: bool = False):
        self.layers = list(layers) if layers is not None else list(range(len(decoder_layers(model))))
        self.model = model
        self.to_cpu = to_cpu
        self.acts: dict[int, torch.Tensor] = {}
        self._handles = []

    def _hook(self, li: int):
        def fn(_mod, _inp, out):
            h = _hidden(out).detach()
            self.acts[li] = h.cpu() if self.to_cpu else h
        return fn

    def __enter__(self):
        for li in self.layers:
            self._handles.append(decoder_layers(self.model)[li].register_forward_hook(self._hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class ResidualPatch:
    """Replace or edit resid_post at one layer: fn(h) -> h'. Use for activation patching, steering, feature patching."""

    def __init__(self, model: nn.Module, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.model, self.layer, self.fn = model, layer, fn
        self._handle = None

    def __enter__(self):
        def hook(_mod, _inp, out):
            return _with_hidden(out, self.fn(_hidden(out)))
        self._handle = decoder_layers(self.model)[self.layer].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()


def count_quantizable_params(model: nn.Module) -> int:
    from .quant.apply import iter_quantizable
    return sum(m.weight.numel() for _, m in iter_quantizable(model))
