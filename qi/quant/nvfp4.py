"""NVFP4 quantize-dequantize (QDQ) in plain PyTorch.

Format, following the compressed-tensors NVFP4 weight path (what llm-compressor exports and vLLM serves):
  elements     FP4 E2M1 on the grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, round to nearest, ties to the even code
  block scale  one FP8 E4M3 value per 16 consecutive elements along the input (last) dimension
  global scale one FP32 value per tensor, (448 * 6) / amax|W|, so the largest block scale lands exactly on 448
Bit-exactness against llm-compressor's own export is checked on the GPU box in Phase 0 (gate G0.2).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16
SHADOW_RATIO = 1.0 / 24.0  # |w| below block_max/24 rounds to zero under an ideal scale


def llmcompressor_global_scale(amax: torch.Tensor) -> torch.Tensor:
    """llm-compressor's per-tensor NVFP4 scale: (448 * 6) * (1 / amax) evaluated in bf16, returned as fp32.
    Reproduces the exported weight_global_scale bit-for-bit on every Gemma 3 4B matrix (Phase 0 cross-check)."""
    a = amax.to(torch.bfloat16)
    return (torch.tensor(E4M3_MAX * E2M1_MAX, dtype=torch.bfloat16, device=a.device) * (1.0 / a)).float()


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value with ties to the even code, saturating at +-6. fp32 math, any shape."""
    x32 = x.float()
    mag = x32.abs().clamp(max=E2M1_MAX)
    mids = torch.tensor(E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    grid = torch.tensor(E2M1_GRID, device=x.device, dtype=torch.float32)
    m = mag.unsqueeze(-1)
    ge = (m >= mids).sum(-1)  # the code when there is no tie; k+1 at a tie with midpoint k
    tie = (m == mids).any(-1)
    even = ge % 2 == 0
    code = torch.where(tie, torch.where(even, ge, ge - 1), ge)
    return torch.sign(x32) * grid[code]


@dataclass
class NVFP4Stats:
    numel: int
    rel_fro_err: float  # ||dq - w||_F / ||w||_F
    frac_zeroed: float  # nonzero weights that became exactly zero, as a share of nonzero weights
    frac_shadowed: float  # nonzero weights with |w| < block_max/24, as a share of nonzero weights
    clip_err_share: float  # share of squared error on elements saturated at |6| after scale rounding
    global_scale: float

    def as_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def nvfp4_qdq(w: torch.Tensor, block: int = BLOCK, with_stats: bool = True, global_amax: float | None = None,
              match_llmcompressor: bool = True):
    """Return (dequantized weight in w.dtype, NVFP4Stats or None). Blocks run along the last dimension.
    global_amax: use this instead of amax|w| for the per-tensor scale, as llm-compressor does for matrices that
    vLLM fuses into one GEMM (q/k/v share one global scale, gate/up share one), so the served checkpoint matches.
    match_llmcompressor: compute the per-tensor scale in bf16 arithmetic exactly as llm-compressor does
    (verified against its export on all 238 Gemma 3 4B matrices); False gives the ideal fp32 formula."""
    if w.shape[-1] % block:
        raise ValueError(f"last dim {w.shape[-1]} is not divisible by block {block}")
    w32 = w.detach().float()
    amax = w32.abs().amax()
    if global_amax is not None:
        amax = torch.as_tensor(float(global_amax), device=w32.device)
    if amax == 0:
        return w.detach().clone(), (NVFP4Stats(w.numel(), 0.0, 0.0, 0.0, 0.0, 0.0) if with_stats else None)
    global_scale = llmcompressor_global_scale(amax) if match_llmcompressor else (E4M3_MAX * E2M1_MAX) / amax
    g = w32.reshape(*w32.shape[:-1], w32.shape[-1] // block, block)
    bmax = g.abs().amax(dim=-1, keepdim=True)
    if match_llmcompressor:
        # llm-compressor: block max / 6 in the weight dtype (bf16), then times the fp32 global scale, then FP8
        local = (bmax.to(torch.bfloat16) / E2M1_MAX).float()
    else:
        local = bmax / E2M1_MAX
    scale_e4m3 = (local * global_scale).clamp(max=E4M3_MAX).to(torch.float8_e4m3fn)
    scale = scale_e4m3.float() / global_scale  # effective per-block scale after E4M3 rounding
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    x = g / safe
    q = round_to_e2m1(x)
    dq = torch.where(scale > 0, q * scale, torch.zeros_like(q)).reshape(w32.shape)
    out = dq.to(w.dtype)
    if not with_stats:
        return out, None
    err2 = (dq - w32).square()
    tot = err2.sum()
    nonzero = w32 != 0
    n_nonzero = nonzero.sum().clamp(min=1)
    zeroed = (nonzero & (dq == 0)).sum() / n_nonzero
    shadowed = ((g.abs() < bmax * SHADOW_RATIO) & (g != 0)).sum() / n_nonzero
    clipped = (x.abs() > E2M1_MAX).reshape(w32.shape)
    clip_share = err2[clipped].sum() / tot.clamp(min=1e-30)
    stats = NVFP4Stats(
        numel=w.numel(),
        rel_fro_err=float(tot.sqrt() / w32.norm().clamp(min=1e-30)),
        frac_zeroed=float(zeroed),
        frac_shadowed=float(shadowed),
        clip_err_share=float(clip_share),
        global_scale=float(global_scale),
    )
    return out, stats


def bits_per_weight(block: int = BLOCK) -> float:
    """Storage cost of NVFP4: 4-bit elements plus an 8-bit scale per block (per-tensor scale is negligible)."""
    return 4.0 + 8.0 / block
