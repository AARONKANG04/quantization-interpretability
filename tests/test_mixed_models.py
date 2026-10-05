import pytest
import torch

pytest.importorskip("transformers")
from qi.mixed.saliency import InputStats, fg_proj_scores, kl_gradients, klg_scores, mag_scores, norm_gains, quant_errors  # noqa: E402
from qi.metrics.kl import topk_reference  # noqa: E402
from qi.quant import apply_scheme  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=128,
                           sliding_window=16, sliding_window_pattern=2)
    torch.manual_seed(0)
    return Gemma3ForCausalLM(cfg).eval()


def test_fg_mag_klg_pipeline(tiny):
    errs = quant_errors(tiny)
    assert len(errs) == 21 and all(v.abs().sum() > 0 for v in errs.values())
    gains = norm_gains(tiny)
    assert set(gains) == set(errs)
    dirs = torch.randn(5, 64); w = torch.ones(5)
    fg = fg_proj_scores(tiny, errs, dirs, w, sae_layer=1)
    assert fg["model.layers.0.mlp.down_proj"]["rows"].sum() > 0          # writer at/below the SAE layer
    assert fg["model.layers.2.mlp.down_proj"]["rows"].sum() == 0         # writer above it: not a candidate
    assert fg["model.layers.2.mlp.up_proj"]["cols"].sum() > 0            # reader above it
    assert fg["model.layers.0.mlp.up_proj"]["cols"].sum() == 0
    ids = torch.randint(0, 512, (2, 12))
    stats = InputStats(tiny)
    with torch.no_grad():
        ref = tiny(input_ids=ids).logits[:, :-1].reshape(-1, 512)
    stats.remove()
    mag = mag_scores(errs, stats.mean_abs())
    assert mag["model.layers.1.self_attn.q_proj"]["cols"].shape == (64,)
    logp, idx, tail = topk_reference(ref, k=32)
    apply_scheme(tiny, "nvfp4_rtn")
    grads = kl_gradients(tiny, [ids], lambda bi: (logp, idx, tail))
    assert all(torch.isfinite(g).all() for g in grads.values())
    klg = klg_scores(tiny, errs, grads)
    assert klg["model.layers.0.mlp.down_proj"]["rows"].shape == (64,)
