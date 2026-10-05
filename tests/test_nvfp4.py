import pytest
import torch

from qi.quant.nvfp4 import E2M1_GRID, nvfp4_qdq, round_to_e2m1


def test_round_to_e2m1_nearest_and_ties_to_even_code():
    x = torch.tensor([0.0, 0.2, 0.3, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, 7.0, -0.75, -5.5, -9.0])
    want = torch.tensor([0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -1.0, -6.0, -6.0])
    assert torch.equal(round_to_e2m1(x), want)


def test_grid_values_are_fixed_points():
    g = torch.tensor(E2M1_GRID)
    assert torch.equal(round_to_e2m1(g), g)
    assert torch.equal(round_to_e2m1(-g), -g)


def test_exact_reconstruction_when_representable():
    # one block whose max is 6 * 2^-3: ideal scale 2^-3 is exactly representable in E4M3 after global scaling
    block = torch.tensor(E2M1_GRID + E2M1_GRID) * 2.0**-3  # 16 values
    w = torch.stack([block, -block])
    dq, st = nvfp4_qdq(w)
    assert torch.equal(dq, w)
    assert st.rel_fro_err == 0.0 and st.frac_zeroed == 0.0


def test_shadowed_small_weights_round_to_zero():
    block = torch.full((16,), 0.03)
    block[0] = 1.0  # block max; 0.03 < 1/24 so every other element must vanish
    w = block.unsqueeze(0)
    dq, st = nvfp4_qdq(w)
    assert dq[0, 0] == 1.0 and torch.all(dq[0, 1:] == 0)
    assert st.frac_shadowed == pytest.approx(15 / 16)
    assert st.frac_zeroed == pytest.approx(15 / 16)


def test_random_matrix_error_range_and_dtype():
    torch.manual_seed(0)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16)
    dq, st = nvfp4_qdq(w)
    assert dq.dtype == torch.bfloat16 and dq.shape == w.shape
    assert 0.05 < st.rel_fro_err < 0.35
    for v in (st.frac_zeroed, st.frac_shadowed, st.clip_err_share):
        assert 0.0 <= v <= 1.0
    assert st.global_scale > 0


def test_block_max_is_never_amplified_beyond_six_scale():
    torch.manual_seed(1)
    w = torch.randn(64, 160)
    dq, _ = nvfp4_qdq(w)
    assert dq.abs().max() <= w.abs().max() * 1.07  # at most one E4M3 rounding step of the block scale


def test_bad_block_shape_raises():
    with pytest.raises(ValueError):
        nvfp4_qdq(torch.randn(4, 30))


def test_stats_optional():
    dq, st = nvfp4_qdq(torch.randn(8, 32), with_stats=False)
    assert st is None and dq.shape == (8, 32)
