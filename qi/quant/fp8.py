"""FP8 E4M3 weight QDQ with one scale per output channel (llm-compressor's FP8 weight scheme)."""
from __future__ import annotations

import torch

E4M3_MAX = 448.0


@torch.no_grad()
def fp8_e4m3_qdq_per_channel(w: torch.Tensor, with_stats: bool = True):
    """w is [out, in]; one scale per row (output channel). Returns (dequantized in w.dtype, stats dict or None)."""
    w32 = w.detach().float()
    amax = w32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / E4M3_MAX, torch.ones_like(amax))
    q = (w32 / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()
    dq = q * scale
    out = dq.to(w.dtype)
    if not with_stats:
        return out, None
    stats = {
        "numel": w.numel(),
        "rel_fro_err": float((dq - w32).norm() / w32.norm().clamp(min=1e-30)),
        "frac_zeroed": float(((w32 != 0) & (dq == 0)).sum() / (w32 != 0).sum().clamp(min=1)),
    }
    return out, stats


def bits_per_weight() -> float:
    return 8.0
