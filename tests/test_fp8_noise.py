import torch

from qi.quant.fp8 import fp8_e4m3_qdq_per_channel
from qi.quant.noise import matched_gaussian, permuted_error
from qi.quant.nvfp4 import nvfp4_qdq


def test_fp8_error_small_and_per_channel_scaling():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w[3] *= 50  # one hot row must not affect the others
    dq, st = fp8_e4m3_qdq_per_channel(w)
    assert st["rel_fro_err"] < 0.05
    row_err = (dq[5] - w[5]).norm() / w[5].norm()
    assert row_err < 0.07


def test_fp8_exact_on_representable_values():
    w = torch.tensor([[1.0, 2.0, 0.5, -0.25, 448.0, 0.0]])
    dq, _ = fp8_e4m3_qdq_per_channel(w)
    assert torch.equal(dq, w)


def test_matched_gaussian_matches_frobenius_error():
    torch.manual_seed(0)
    w = torch.randn(128, 256) * 0.02
    q, _ = nvfp4_qdq(w)
    g = torch.Generator().manual_seed(0)
    n = matched_gaussian(w, q, g)
    assert abs((n - w).norm() / (q - w).norm() - 1.0) < 0.05


def test_permuted_error_keeps_histogram_and_norm():
    torch.manual_seed(0)
    w = torch.randn(32, 64) * 0.02
    q, _ = nvfp4_qdq(w)
    g = torch.Generator().manual_seed(0)
    p = permuted_error(w, q, g)
    d_q = (q - w).flatten().sort().values
    d_p = (p - w).flatten().sort().values
    assert torch.allclose(d_q, d_p, atol=1e-6)
    assert not torch.equal(q, p)
