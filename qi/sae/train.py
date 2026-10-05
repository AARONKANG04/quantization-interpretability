"""BatchTopK SAE (Bussmann et al. 2024) trained on activations generated on the fly, then converted to a
JumpReLU SAE by fitting per-feature thresholds, so it plugs into the same analysis as Gemma Scope 2."""
from __future__ import annotations

import math

import torch
from torch import nn

from .jumprelu import JumpReLUSAE


class BatchTopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True))
            self.W_enc.copy_(self.W_dec.T)
        self.register_buffer("threshold", torch.zeros(d_sae))
        self.register_buffer("min_pos_act", torch.full((d_sae,), float("inf")))

    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x):
        pre = self.pre_acts(x)
        B = pre.shape[0]
        flat = pre.relu().flatten()
        kk = min(self.k * B, flat.numel())
        thr = flat.topk(kk).values[-1] if kk > 0 else torch.tensor(0.0, device=x.device)
        acts = pre * (pre >= thr) * (pre > 0)
        with torch.no_grad():
            pos = acts > 0
            mins = torch.where(pos, acts, torch.full_like(acts, float("inf"))).amin(0)
            self.min_pos_act = torch.minimum(self.min_pos_act, mins)
        return acts

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def to_jumprelu(self) -> JumpReLUSAE:
        thr = torch.where(torch.isfinite(self.min_pos_act), self.min_pos_act, torch.full_like(self.min_pos_act, float("inf")))
        return JumpReLUSAE(self.W_enc.data.clone(), self.b_enc.data.clone(), thr.clone(), self.W_dec.data.clone(),
                           self.b_dec.data.clone(), apply_b_dec_to_input=True)


def train_sae(sae: BatchTopKSAE, batches, steps: int, lr: float = 3e-4, aux_k_coef: float = 1 / 32,
              dead_after: int = 2000, log_every: int = 100, log=print) -> dict:
    """batches: iterator of [B, d_in] fp32 activation tensors on the SAE's device. AuxK revives dead latents."""
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    d_sae = sae.W_enc.shape[1]
    last_fired = torch.zeros(d_sae, dtype=torch.long, device=sae.W_enc.device)
    hist = []
    for step in range(1, steps + 1):
        x = next(batches)
        xr, f = sae(x)
        mse = (xr - x).square().sum(-1).mean()
        fired = (f > 0).any(0)
        last_fired[fired] = step
        dead = (step - last_fired) > dead_after
        aux = torch.tensor(0.0, device=x.device)
        if dead.any() and aux_k_coef > 0:
            resid = (x - xr).detach()
            pre = sae.pre_acts(x)[:, dead]
            kk = min(2 * sae.k, int(dead.sum()))
            top = pre.topk(kk, dim=1)
            aux_acts = torch.zeros_like(pre).scatter(1, top.indices, top.values.relu())
            aux = (aux_acts @ sae.W_dec[dead] - resid).square().sum(-1).mean()
        loss = mse + aux_k_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.normalize_decoder()
        if step % log_every == 0 or step == steps:
            var = x.var(0).sum()
            fvu = (mse / var.clamp(min=1e-12)).item()
            rec = {"step": step, "mse": mse.item(), "fvu": fvu, "dead": int(dead.sum()), "aux": aux.item()}
            hist.append(rec)
            log(rec)
    return {"history": hist}
