import pytest
import torch

from qi.sae.jumprelu import JumpReLUSAE, reconstruction_stats
from qi.sae.shift import ShiftAccumulator, concentration
from qi.sae.steer import gain_correction, mean_shift
from qi.sae.train import BatchTopKSAE, train_sae


def _toy_sae(d=16, m=48, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_dec = torch.randn(m, d, generator=g); W_dec /= W_dec.norm(dim=1, keepdim=True)
    return JumpReLUSAE(W_dec.T.clone(), torch.zeros(m), torch.full((m,), 0.5), W_dec, torch.zeros(d))


def test_jumprelu_encode_decode_and_subset():
    sae = _toy_sae()
    x = torch.randn(10, 16)
    f = sae.encode(x)
    assert f.shape == (10, 48) and (f[f > 0] > 0.5).all()
    feats = torch.tensor([1, 5, 7])
    assert torch.allclose(sae.encode(x, feats), f[:, feats])
    assert torch.allclose(sae.decode(f), f @ sae.W_dec + sae.b_dec)
    st = reconstruction_stats(sae, x)
    assert set(st) >= {"variance_explained", "l0", "mse", "cosine"} and st["n_tokens"] == 10


def test_shift_accumulator_identities():
    sae = _toy_sae()
    g = torch.Generator().manual_seed(1)
    ha = torch.randn(64, 16, generator=g); hb = ha + 0.3 * torch.randn(64, 16, generator=g)
    fa, fb = sae.encode(ha), sae.encode(hb)
    xa, xb = sae.decode(fa), sae.decode(fb)
    acc = ShiftAccumulator(sae.W_dec, n_examples=3)
    seq = torch.arange(64) // 8; pos = torch.arange(64) % 8
    acc.update(fa[:32], fb[:32], ha[:32], hb[:32], xa[:32], xb[:32], seq[:32], pos[:32])
    acc.update(fa[32:], fb[32:], ha[32:], hb[32:], xa[32:], xb[32:], seq[32:], pos[32:])
    out = acc.finalize()
    s = out["shares"]
    assert abs(s["explained"] + s["error"] + s["cross_explained_error"] - 1.0) < 1e-5  # exact decomposition
    assert 0 <= s["mean_shift_bias"] <= 1 and s["mean_shift_bias"] <= s["top_direction"] <= 1 + 1e-6
    assert out["n_tokens"] == 64
    pf = out["per_feature"]
    assert torch.allclose(pf["energy"], ((fb - fa).square().sum(0) * sae.W_dec.square().sum(1)), atol=1e-4)
    assert (pf["ex_val"][:, 0] >= pf["ex_val"][:, 1]).all()  # heap sorted
    top_feat = int(pf["ex_val"][:, 0].argmax())
    assert pf["ex_val"][top_feat, 0] == fa[:, top_feat].max()


def test_concentration_curve():
    c = concentration(torch.tensor([10.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert c["frac_for_50"] == 0.1 and 0 < c["gini"] < 1 and c["share_at"]["top_10pct"] == pytest.approx(10 / 19)


def test_batchtopk_trains_and_converts():
    torch.manual_seed(0)
    d, m, k = 16, 64, 4
    basis = torch.randn(m, d); basis /= basis.norm(dim=1, keepdim=True)

    def batches():
        while True:
            coef = torch.zeros(256, m)
            idx = torch.randint(0, m, (256, k))
            coef.scatter_(1, idx, torch.rand(256, k) + 0.5)
            yield coef @ basis + 0.01 * torch.randn(256, d)

    sae = BatchTopKSAE(d, m, k)
    hist = train_sae(sae, batches(), steps=300, lr=3e-3, log_every=100, log=lambda r: None)["history"]
    assert hist[-1]["fvu"] < hist[0]["fvu"]
    j = sae.to_jumprelu()
    x = next(batches())
    f = j.encode(x)
    assert f.shape == (256, m) and (f > 0).sum(1).float().mean() < 3 * k


def test_steer_hooks():
    sae = _toy_sae()
    x = torch.randn(5, 16)
    feats = torch.tensor([0, 2]); gains = torch.tensor([2.0, 0.5])
    y = gain_correction(sae, feats, gains)(x)
    f = sae.encode(x, feats)
    expect = x + ((gains - 1) * f) @ sae.W_dec[feats]
    assert torch.allclose(y, expect, atol=1e-6)
    assert torch.allclose(mean_shift(torch.ones(16))(x), x + 1)
