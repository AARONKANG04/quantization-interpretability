import torch

from qi.mixed.saliency import memory_bytes, random_scores, select_budget


def test_select_budget_respects_budget_and_prefers_high_scores():
    shapes = {"m1": (4, 8), "m2": (6, 3)}
    scores = {"m1": {"rows": torch.tensor([0.1, 5.0, 0.2, 0.0]), "cols": torch.zeros(8)},
              "m2": {"rows": torch.zeros(6), "cols": torch.tensor([3.0, 0.0, 9.0])}}
    sel, used = select_budget(scores, shapes, budget_weights=14, per_weight=False)
    assert used <= 14
    assert 2 in sel["m2"]["cols"] and 1 in sel["m1"]["rows"]  # the two best candidates (6 + 8 weights)
    assert 1 not in sel["m2"]["cols"]  # zero score never selected


def test_memory_accounting_matches_plan_numbers():
    m = memory_bytes(3_208_642_560, int(0.01 * 3_208_642_560))
    assert abs(m["extra_bytes"] / 1e6 - 46.1) < 0.5           # 1% of weights adds about 46 MB
    assert abs(m["bytes_nvfp4"] / 1e9 - 1.805) < 0.01
    assert abs(m["bytes_bf16"] / 1e9 - 6.417) < 0.01


def test_random_scores_shapes():
    errs = {"a": torch.zeros(3, 5), "b": torch.zeros(2, 4)}
    s = random_scores(errs, seed=1)
    assert s["a"]["rows"].shape == (3,) and s["a"]["cols"].shape == (5,) and s["b"]["cols"].shape == (4,)
