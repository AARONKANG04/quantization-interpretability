"""Apply a QDQ scheme to the quantizable linear layers of a Gemma 3 model, with per-layer / per-matrix /
per-channel control. Everything is in place and in bf16, so hooks and interventions work unchanged."""
from __future__ import annotations

import json
import re
import zlib
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch import nn

from .fp8 import fp8_e4m3_qdq_per_channel
from .noise import matched_gaussian, permuted_error
from .nvfp4 import nvfp4_qdq

QUANTIZABLE = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SCHEMES = ("none", "nvfp4_rtn", "fp8_e4m3", "noise_matched", "error_permuted")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def iter_quantizable(model: nn.Module) -> Iterator[tuple[str, nn.Linear]]:
    """Yield (name, module) for every decoder-block linear projection, in model order."""
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in QUANTIZABLE and layer_index(name) is not None:
            yield name, mod


FUSED_GROUPS = (("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj"))


@torch.no_grad()
def fused_global_amax(model: nn.Module) -> dict[str, float]:
    """{module name: shared amax} for the matrices vLLM fuses (q/k/v and gate/up of the same layer)."""
    by_layer: dict[tuple[int, int], list[tuple[str, nn.Linear]]] = {}
    for name, mod in iter_quantizable(model):
        suffix = name.split(".")[-1]
        for gi, group in enumerate(FUSED_GROUPS):
            if suffix in group:
                by_layer.setdefault((layer_index(name), gi), []).append((name, mod))
    out: dict[str, float] = {}
    for members in by_layer.values():
        amax = max(float(m.weight.detach().abs().amax()) for _, m in members)
        for name, _ in members:
            out[name] = amax
    return out


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(name == p or name.endswith(p) for p in patterns)


@torch.no_grad()
def apply_scheme(
    model: nn.Module,
    scheme: str,
    *,
    layers: Iterable[int] | None = None,
    exclude_layers: Iterable[int] | None = None,
    matrices: Iterable[str] | None = None,
    protect: dict[str, dict] | None = None,
    seed: int = 0,
    block: int = 16,
    fuse_global_scales: bool = True,
) -> dict[str, dict]:
    """Quantize-dequantize the selected weights in place and return per-matrix stats keyed by module name.

    layers / exclude_layers: decoder-layer indices to include / leave untouched (None = all).
    matrices: module-name suffixes such as "down_proj" or full names; None = all seven projections.
    protect: {module name: {"rows": LongTensor, "cols": LongTensor}} kept at the original precision
             (rows = output channels, cols = input channels), the channel-level mixed-precision primitive.
    fuse_global_scales: NVFP4 per-tensor scale shared across q/k/v and across gate/up within a layer, matching
             llm-compressor's export for the GEMMs vLLM fuses (verified bit-exact in Phase 0).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    include = set(layers) if layers is not None else None
    exclude = set(exclude_layers) if exclude_layers else set()
    protect = protect or {}
    stats: dict[str, dict] = {}
    shared_amax = fused_global_amax(model) if (fuse_global_scales and scheme in ("nvfp4_rtn", "noise_matched", "error_permuted")) else {}
    for name, mod in iter_quantizable(model):
        li = layer_index(name)
        if include is not None and li not in include:
            continue
        if li in exclude:
            continue
        if matrices is not None and not _matches(name, matrices):
            continue
        if scheme == "none":
            continue
        w = mod.weight.data
        if scheme == "nvfp4_rtn":
            dq, st = nvfp4_qdq(w, block=block, global_amax=shared_amax.get(name))
            st = st.as_dict()
        elif scheme == "fp8_e4m3":
            dq, st = fp8_e4m3_qdq_per_channel(w)
        else:
            q1, _ = nvfp4_qdq(w, block=block, with_stats=False, global_amax=shared_amax.get(name))
            gen = torch.Generator(device=w.device)
            gen.manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            dq = matched_gaussian(w, q1, gen) if scheme == "noise_matched" else permuted_error(w, q1, gen)
            st = {"numel": w.numel(), "rel_fro_err": float((dq.float() - w.float()).norm() / w.float().norm())}
        if name in protect:
            p = protect[name]
            rows, cols = p.get("rows"), p.get("cols")
            if rows is not None and len(rows):
                dq[rows, :] = w[rows, :]
            if cols is not None and len(cols):
                dq[:, cols] = w[:, cols]
            st["protected_rows"] = int(len(rows)) if rows is not None else 0
            st["protected_cols"] = int(len(cols)) if cols is not None else 0
        mod.weight.data.copy_(dq)
        stats[name] = {"layer": li, "shape": list(w.shape), "scheme": scheme, **st}
    return stats


def summarize_stats(stats: dict[str, dict]) -> dict:
    """Weight-weighted aggregate of per-matrix stats, plus per-layer relative error."""
    numel = sum(s["numel"] for s in stats.values())
    if numel == 0:
        return {"numel": 0}

    def wavg(key: str) -> float | None:
        vals = [(s[key], s["numel"]) for s in stats.values() if key in s]
        return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None

    per_layer: dict[int, list] = {}
    for s in stats.values():
        per_layer.setdefault(s["layer"], []).append(s)
    layer_err = {
        li: sum(s["rel_fro_err"] ** 2 * s["numel"] for s in ss) / sum(s["numel"] for s in ss)
        for li, ss in per_layer.items()
    }
    return {
        "numel": numel,
        "n_matrices": len(stats),
        "rel_fro_err_wavg": wavg("rel_fro_err"),
        "frac_zeroed_wavg": wavg("frac_zeroed"),
        "frac_shadowed_wavg": wavg("frac_shadowed"),
        "clip_err_share_wavg": wavg("clip_err_share"),
        "layer_mse_rel": {int(k): float(v) for k, v in sorted(layer_err.items())},
    }


def save_qdq_checkpoint(model, tokenizer, path: str | Path, meta: dict) -> Path:
    """QDQ weights are ordinary bf16 safetensors, so vLLM serves them at full speed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
    (path / "qdq_meta.json").write_text(json.dumps(meta, indent=2))
    return path
