import torch

from qi.metrics.kl import bucketed_kl_from_reference, token_metrics_from_logits, topk_reference


def _logits(T=50, V=300, seed=0, scale=3.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, V, generator=g) * scale


def test_kl_zero_for_identical_and_nonnegative():
    a = _logits()
    m = token_metrics_from_logits(a, a, chunk=7)
    assert torch.allclose(m["kl"], torch.zeros_like(m["kl"]), atol=1e-6)
    assert m["top1_agree"].all()
    b = a + _logits(seed=1, scale=0.5)
    m2 = token_metrics_from_logits(a, b, chunk=7)
    assert (m2["kl"] >= -1e-6).all() and m2["kl"].mean() > 0
    assert m2["entropy_a"].shape == (50,)


def test_bucketed_kl_is_lower_bound_and_exact_with_full_support():
    a = _logits()
    b = a + _logits(seed=2, scale=0.5)
    full = token_metrics_from_logits(a, b)["kl"]
    logp, idx, tail = topk_reference(a, k=32, chunk=16)
    assert logp.shape == (50, 32) and idx.dtype == torch.int32
    bucketed = bucketed_kl_from_reference(logp, idx, tail, b, chunk=16)["kl"]
    assert (bucketed <= full + 1e-5).all()
    assert bucketed.mean() > 0.5 * full.mean()  # top-32 of 300 captures most of the mass at this scale
    logp_all, idx_all, tail_all = topk_reference(a, k=300)
    exact = bucketed_kl_from_reference(logp_all, idx_all, tail_all, b)["kl"]
    assert torch.allclose(exact, full, atol=1e-4)


def test_bucketed_top1_agreement_matches_full():
    a = _logits(); b = a + _logits(seed=3, scale=1.0)
    full = token_metrics_from_logits(a, b)["top1_agree"]
    logp, idx, tail = topk_reference(a, k=8)
    assert torch.equal(bucketed_kl_from_reference(logp, idx, tail, b)["top1_agree"], full)
