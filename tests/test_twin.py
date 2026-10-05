"""TwinRunner must feed model_a exactly what model_b sees during generation (needs transformers)."""
import pytest
import torch

pytest.importorskip("transformers")
from qi.models import ResidualCapture, ResidualPatch  # noqa: E402
from qi.sae.steer import TwinRunner, full_residual_patch  # noqa: E402


@pytest.fixture(scope="module")
def pair():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    cfg = Gemma3TextConfig(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                           num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=128,
                           sliding_window=16, sliding_window_pattern=2)
    torch.manual_seed(0); a = Gemma3ForCausalLM(cfg).eval()
    torch.manual_seed(1); b = Gemma3ForCausalLM(cfg).eval()
    return a, b


def test_twin_tracks_generation(pair):
    a, b = pair
    ids = torch.randint(0, 512, (2, 7))
    with TwinRunner(b, a, layer=2) as twin:
        out = b.generate(input_ids=ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
        last = twin.hidden  # model_a's layer-2 residual for the last decode step
    assert out.shape == (2, 11) and last.shape[0] == 2 and last.shape[1] == 1
    with ResidualCapture(a, layers=[2]) as cap:
        a(input_ids=out[:, :-1])
    assert torch.allclose(cap.acts[2][:, -1], last[:, 0], atol=1e-4)


def test_full_residual_patch_turns_b_into_a_downstream(pair):
    a, b = pair
    ids = torch.randint(0, 512, (1, 9))
    with TwinRunner(b, a, layer=1) as twin, ResidualPatch(b, 1, full_residual_patch(twin)):
        patched = b(input_ids=ids).logits
    # from layer 2 on, b computes with its own weights, so logits differ from a's; the patch itself must apply
    with ResidualCapture(b, layers=[1]) as cap_b:
        plain = b(input_ids=ids).logits
    assert not torch.allclose(patched, plain)
