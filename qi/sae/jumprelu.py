"""A minimal JumpReLU SAE container used for Gemma Scope 2 (loaded through sae-lens) and for our own trained
SAEs after a threshold is fitted. Keeps only what the analysis needs: encode, decode, decoder norms."""
from __future__ import annotations

import torch
from torch import nn


class JumpReLUSAE(nn.Module):
    def __init__(self, W_enc: torch.Tensor, b_enc: torch.Tensor, threshold: torch.Tensor, W_dec: torch.Tensor,
                 b_dec: torch.Tensor, apply_b_dec_to_input: bool = False):
        super().__init__()
        self.W_enc = nn.Parameter(W_enc, requires_grad=False)   # [d, m]
        self.b_enc = nn.Parameter(b_enc, requires_grad=False)   # [m]
        self.threshold = nn.Parameter(threshold, requires_grad=False)  # [m]
        self.W_dec = nn.Parameter(W_dec, requires_grad=False)   # [m, d]
        self.b_dec = nn.Parameter(b_dec, requires_grad=False)   # [d]
        self.apply_b_dec_to_input = apply_b_dec_to_input

    @property
    def d_in(self) -> int:
        return self.W_enc.shape[0]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    @property
    def dec_norms(self) -> torch.Tensor:
        return self.W_dec.norm(dim=1)

    def pre_acts(self, x: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        x = x.to(self.W_enc.dtype)
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        if feats is None:
            return x @ self.W_enc + self.b_enc
        return x @ self.W_enc[:, feats] + self.b_enc[feats]

    def encode(self, x: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        pre = self.pre_acts(x, feats)
        thr = self.threshold if feats is None else self.threshold[feats]
        return pre * (pre > thr)

    def decode(self, f: torch.Tensor, feats: torch.Tensor | None = None) -> torch.Tensor:
        if feats is None:
            return f @ self.W_dec + self.b_dec
        return f @ self.W_dec[feats] + self.b_dec

    @classmethod
    def from_sae_lens(cls, sae) -> "JumpReLUSAE":
        thr = getattr(sae, "threshold", None)
        if thr is None:
            thr = torch.zeros(sae.W_enc.shape[1], device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        apply = bool(getattr(getattr(sae, "cfg", None), "apply_b_dec_to_input", False))
        return cls(sae.W_enc.detach().clone(), sae.b_enc.detach().clone(), thr.detach().clone(),
                   sae.W_dec.detach().clone(), sae.b_dec.detach().clone(), apply)

    def save(self, path):
        torch.save({"W_enc": self.W_enc.data, "b_enc": self.b_enc.data, "threshold": self.threshold.data,
                    "W_dec": self.W_dec.data, "b_dec": self.b_dec.data, "apply_b_dec_to_input": self.apply_b_dec_to_input}, path)

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32) -> "JumpReLUSAE":
        d = torch.load(path, map_location=device)
        return cls(d["W_enc"].to(dtype), d["b_enc"].to(dtype), d["threshold"].to(dtype), d["W_dec"].to(dtype),
                   d["b_dec"].to(dtype), d.get("apply_b_dec_to_input", False)).to(device)


@torch.no_grad()
def reconstruction_stats(sae: JumpReLUSAE, h: torch.Tensor, chunk: int = 2048) -> dict:
    """Variance explained, mean L0, MSE and mean cosine between h and its reconstruction."""
    h = h.reshape(-1, h.shape[-1]).float()
    mean = h.mean(0, keepdim=True)
    ss_tot = (h - mean).square().sum()
    ss_res, l0, cos = 0.0, 0.0, 0.0
    for s in range(0, h.shape[0], chunk):
        x = h[s : s + chunk]
        f = sae.encode(x)
        xr = sae.decode(f).float()
        ss_res += (x - xr).square().sum().item()
        l0 += (f > 0).sum().item()
        cos += torch.nn.functional.cosine_similarity(x, xr, dim=-1).sum().item()
    n = h.shape[0]
    return {"variance_explained": 1.0 - ss_res / max(ss_tot.item(), 1e-12), "l0": l0 / n, "mse": ss_res / (n * h.shape[1]),
            "cosine": cos / n, "n_tokens": n}
