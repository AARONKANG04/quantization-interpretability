"""Streaming feature-shift accounting (plan E2.2). One pass over paired activations; nothing per-token is stored
except a top-K example heap per feature. The residual delta decomposes exactly as
    h_b - h_a = (f_b - f_a) @ W_dec + [(h_b - xhat_b) - (h_a - xhat_a)]  =  E + err,
so ||dh||^2 = ||E||^2 + ||err||^2 + 2<E, err>, and ||E||^2 = sum_i energy_i + cross terms between features."""
from __future__ import annotations

import torch


class ShiftAccumulator:
    def __init__(self, W_dec: torch.Tensor, n_examples: int = 20):
        self.W_dec = W_dec
        m = W_dec.shape[0]
        dev = W_dec.device
        z = lambda: torch.zeros(m, dtype=torch.float64, device=dev)  # noqa: E731
        self.sum_a, self.sum_b, self.sumsq_a, self.sumsq_b, self.prod = z(), z(), z(), z(), z()
        self.cnt_a, self.cnt_b, self.cnt_either, self.energy = z(), z(), z(), z()
        self.peak_a = torch.zeros(m, device=dev)
        self.peak_b = torch.zeros(m, device=dev)
        self.dec_norm2 = W_dec.float().square().sum(1).double()
        self.n = 0
        self.total_dh2 = self.explained2 = self.err2 = self.cross_e_err = 0.0
        d = W_dec.shape[1]
        self.sum_dh = torch.zeros(d, dtype=torch.float64, device=dev)       # mean shift direction
        self.dh_cov = torch.zeros(d, d, dtype=torch.float64, device=dev)   # second moment of dh, for its top direction
        self.K = n_examples
        self.ex_val = torch.full((m, n_examples), -1.0, device=dev)
        self.ex_seq = torch.full((m, n_examples), -1, dtype=torch.long, device=dev)
        self.ex_pos = torch.full((m, n_examples), -1, dtype=torch.long, device=dev)

    @torch.no_grad()
    def update(self, fa: torch.Tensor, fb: torch.Tensor, ha: torch.Tensor, hb: torch.Tensor,
               xa: torch.Tensor, xb: torch.Tensor, seq_ids: torch.Tensor, pos: torch.Tensor):
        fa, fb = fa.float(), fb.float()
        da = fb - fa
        self.sum_a += fa.sum(0).double(); self.sum_b += fb.sum(0).double()
        self.sumsq_a += fa.square().sum(0).double(); self.sumsq_b += fb.square().sum(0).double()
        self.prod += (fa * fb).sum(0).double()
        act_a, act_b = fa > 0, fb > 0
        self.cnt_a += act_a.sum(0).double(); self.cnt_b += act_b.sum(0).double(); self.cnt_either += (act_a | act_b).sum(0).double()
        self.energy += da.square().sum(0).double() * self.dec_norm2
        self.peak_a = torch.maximum(self.peak_a, fa.amax(0)); self.peak_b = torch.maximum(self.peak_b, fb.amax(0))
        E = da @ self.W_dec.float()
        err = (hb.float() - xb.float()) - (ha.float() - xa.float())
        dh = hb.float() - ha.float()
        self.total_dh2 += dh.square().sum().item(); self.explained2 += E.square().sum().item()
        self.sum_dh += dh.sum(0).double(); self.dh_cov += (dh.T @ dh).double()
        self.err2 += err.square().sum().item(); self.cross_e_err += (E * err).sum().item()
        self.n += fa.shape[0]
        k = min(self.K, fa.shape[0])
        vals, idx = fa.topk(k, dim=0)                      # [k, m]
        cand_val = torch.cat([self.ex_val, vals.T], dim=1)  # [m, K + k]
        cand_seq = torch.cat([self.ex_seq, seq_ids[idx].T], dim=1)
        cand_pos = torch.cat([self.ex_pos, pos[idx].T], dim=1)
        top, ti = cand_val.topk(self.K, dim=1)
        self.ex_val, self.ex_seq, self.ex_pos = top, cand_seq.gather(1, ti), cand_pos.gather(1, ti)

    @torch.no_grad()
    def finalize(self) -> dict:
        n = max(self.n, 1)
        mu_a, mu_b = self.sum_a / n, self.sum_b / n
        var_a = (self.sumsq_a / n - mu_a**2).clamp(min=0); var_b = (self.sumsq_b / n - mu_b**2).clamp(min=0)
        cov = self.prod / n - mu_a * mu_b
        corr = cov / (var_a.sqrt() * var_b.sqrt()).clamp(min=1e-12)
        corr[(var_a == 0) | (var_b == 0)] = float("nan")
        mean_act_a = self.sum_a / self.cnt_a.clamp(min=1); mean_act_b = self.sum_b / self.cnt_b.clamp(min=1)
        freq_ratio = (self.cnt_b / n) / (self.cnt_a / n).clamp(min=1e-12)
        active = self.cnt_either > 0
        label = torch.full_like(self.cnt_a, 4, dtype=torch.long)  # 0 suppressed, 1 amplified, 2 newly dead, 3 newly alive, 4 stable
        label[(self.cnt_a > 0) & (self.cnt_b == 0)] = 2
        label[(self.cnt_a == 0) & (self.cnt_b > 0)] = 3
        label[(self.cnt_a > 0) & (self.cnt_b > 0) & (freq_ratio < 0.5)] = 0
        label[(self.cnt_a > 0) & (self.cnt_b > 0) & (freq_ratio > 2.0)] = 1
        label[~active] = -1
        sum_energy = self.energy.sum().item()
        mean_dh = self.sum_dh / n
        bias_share = float(n * mean_dh.square().sum() / max(self.total_dh2, 1e-12))   # ||mean dh||^2 / mean ||dh||^2
        evals, evecs = torch.linalg.eigh(self.dh_cov)
        top_share = float(evals[-1] / max(self.total_dh2, 1e-12))                     # share of ||dh||^2 along its top direction
        top_dir = evecs[:, -1].float().cpu()
        out = {
            "n_tokens": self.n, "n_features": int(self.energy.numel()), "n_active_either": int(active.sum()),
            "shares": {"explained": self.explained2 / max(self.total_dh2, 1e-12), "error": self.err2 / max(self.total_dh2, 1e-12),
                       "cross_explained_error": 2 * self.cross_e_err / max(self.total_dh2, 1e-12),
                       "per_feature_sum_over_explained": sum_energy / max(self.explained2, 1e-12),
                       "mean_shift_bias": bias_share, "top_direction": top_share},
            "concentration": concentration(self.energy[active]),
            "concentration_per_activation": concentration((self.energy / self.cnt_either.clamp(min=1))[active]),
            "label_counts": {k: int((label == v).sum()) for k, v in
                             {"suppressed": 0, "amplified": 1, "newly_dead": 2, "newly_alive": 3, "stable": 4, "inactive": -1}.items()},
            "per_feature": {"energy": self.energy.float().cpu(), "cnt_a": self.cnt_a.float().cpu(), "cnt_b": self.cnt_b.float().cpu(),
                            "mean_act_a": mean_act_a.float().cpu(), "mean_act_b": mean_act_b.float().cpu(), "peak_a": self.peak_a.cpu(),
                            "peak_b": self.peak_b.cpu(), "corr": corr.float().cpu(), "freq_ratio": freq_ratio.float().cpu(),
                            "label": label.cpu(), "ex_val": self.ex_val.cpu(), "ex_seq": self.ex_seq.cpu(), "ex_pos": self.ex_pos.cpu(),
                            "mean_dh": mean_dh.float().cpu(), "top_dir": top_dir},
        }
        return out


def concentration(energy: torch.Tensor) -> dict:
    """Cumulative-share curve of a nonnegative vector sorted descending: fraction of items reaching 50% and 80%, Gini."""
    e = energy.double().sort(descending=True).values
    tot = e.sum()
    if e.numel() == 0 or tot <= 0:
        return {"n": int(e.numel()), "frac_for_50": None, "frac_for_80": None, "gini": None}
    cum = e.cumsum(0) / tot
    n = e.numel()
    f50 = (int((cum >= 0.5).nonzero()[0]) + 1) / n
    f80 = (int((cum >= 0.8).nonzero()[0]) + 1) / n
    asc = e.flip(0)
    idx = torch.arange(1, n + 1, dtype=torch.float64, device=e.device)
    gini = float((2 * (idx * asc).sum() / (n * asc.sum())) - (n + 1) / n)
    curve = {f"top_{p}pct": float(cum[max(int(n * p / 100) - 1, 0)]) for p in (0.1, 0.5, 1, 2, 5, 10, 20, 50)}
    return {"n": n, "frac_for_50": f50, "frac_for_80": f80, "gini": gini, "share_at": curve}
