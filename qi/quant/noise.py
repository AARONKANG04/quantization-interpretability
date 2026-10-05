"""Perturbation controls that share the size of the NVFP4 error but not its structure.

C2 matched_gaussian: isotropic Gaussian noise with the same per-matrix Frobenius error as the quantization error.
C3 permuted_error:   the quantization error entries themselves, randomly permuted within the matrix
                     (same histogram, destroyed placement).
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def matched_gaussian(w: torch.Tensor, w_q: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    delta = w_q.float() - w.float()
    sigma = delta.norm() / math.sqrt(delta.numel())
    noise = torch.randn(delta.shape, generator=generator, device=w.device, dtype=torch.float32) * sigma
    return (w.float() + noise).to(w.dtype)


@torch.no_grad()
def permuted_error(w: torch.Tensor, w_q: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    delta = (w_q.float() - w.float()).flatten()
    perm = torch.randperm(delta.numel(), generator=generator, device=delta.device)
    return (w.float() + delta[perm].reshape(w.shape)).to(w.dtype)
