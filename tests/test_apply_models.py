"""Needs transformers; skipped on machines without it. Uses a tiny random Gemma 3 text model."""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from qi.models import ResidualCapture, ResidualPatch, count_quantizable_params, global_layer_indices  # noqa: E402
from qi.quant import apply_scheme, iter_quantizable  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=128,
                           sliding_window=16, sliding_window_pattern=2)
    torch.manual_seed(0)
    return Gemma3ForCausalLM(cfg).eval()


def test_global_layers(tiny):
    assert global_layer_indices(tiny.config) == [1, 3]


def test_apply_only_selected_layers_and_protect(tiny):
    before = {n: m.weight.detach().clone() for n, m in iter_quantizable(tiny)}
    assert len(before) == 4 * 7
    down0 = "model.layers.0.mlp.down_proj"
    stats = apply_scheme(tiny, "nvfp4_rtn", layers=[0], protect={down0: {"rows": torch.tensor([0, 1])}})
    assert set(s["layer"] for s in stats.values()) == {0}
    for n, m in iter_quantizable(tiny):
        if n.endswith("layers.0.self_attn.q_proj"):
            assert not torch.equal(m.weight, before[n])
        if ".layers.1." in n:
            assert torch.equal(m.weight, before[n])
        if n == down0:
            assert torch.equal(m.weight[:2], before[n][:2]) and not torch.equal(m.weight[2:], before[n][2:])
    assert count_quantizable_params(tiny) == sum(w.numel() for w in before.values())


def test_capture_and_patch(tiny):
    ids = torch.randint(0, 512, (2, 12))
    with ResidualCapture(tiny, layers=[1, 2]) as cap:
        base = tiny(input_ids=ids).logits
    assert cap.acts[1].shape == (2, 12, 64) and 2 in cap.acts
    with ResidualPatch(tiny, 1, lambda h: h * 0):
        patched = tiny(input_ids=ids).logits
    assert not torch.allclose(base, patched)
    with ResidualPatch(tiny, 1, lambda h: h):
        same = tiny(input_ids=ids).logits
    assert torch.allclose(base, same)


def test_fused_global_amax_groups(tiny):
    from qi.quant.apply import fused_global_amax
    shared = fused_global_amax(tiny)
    l0 = {k: v for k, v in shared.items() if ".layers.0." in k}
    qkv = {v for k, v in l0.items() if k.split(".")[-1] in ("q_proj", "k_proj", "v_proj")}
    gu = {v for k, v in l0.items() if k.split(".")[-1] in ("gate_proj", "up_proj")}
    assert len(qkv) == 1 and len(gu) == 1 and not any(k.endswith(("o_proj", "down_proj")) for k in shared)
