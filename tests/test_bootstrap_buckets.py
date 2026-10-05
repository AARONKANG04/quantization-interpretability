import numpy as np

from qi.metrics.bootstrap import cluster_bootstrap_mean, flips, paired_bootstrap_delta
from qi.metrics.token_buckets import freq_bucket, position_bucket, token_class


def test_paired_bootstrap_contains_true_delta_and_is_tighter_than_naive():
    rng = np.random.default_rng(0)
    base = rng.random(1319) < 0.6
    flip = rng.random(1319) < 0.05
    other = np.where(flip, ~base, base)
    r = paired_bootstrap_delta(base, other, n_boot=2000)
    assert r["lo"] <= r["delta"] <= r["hi"]
    assert (r["hi"] - r["lo"]) < 0.05  # paired design: half-width around 1 point at 5% flips


def test_flips_accounting():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 1, 0, 1], bool)
    f = flips(a, b)
    assert f["correct_to_incorrect"] == 0.2 and f["incorrect_to_correct"] == 0.2 and f["total"] == 0.4


def test_cluster_bootstrap():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(40), 50)
    values = rng.exponential(0.1, size=clusters.size) + np.repeat(rng.normal(0, 0.02, 40), 50)
    r = cluster_bootstrap_mean(values, clusters, n_boot=500)
    assert r["n_clusters"] == 40 and r["lo"] <= r["mean"] <= r["hi"]


def test_token_classes():
    cases = {"▁123": "numeral", "3.14": "numeral", ",": "punctuation", "▁{": "code_symbol", "();": "code_symbol",
             "▁the": "word", "ing": "word", "日本": "non_latin", "\n\n": "whitespace", "▁": "whitespace",
             "café": "word", "=24": "numeral_mixed", "x2": "numeral_mixed", "▁$5": "numeral_mixed"}
    for tok, want in cases.items():
        assert token_class(tok) == want, (tok, token_class(tok), want)


def test_buckets():
    assert position_bucket(0) == "0-256" and position_bucket(300) == "256-1024" and position_bucket(5000) == "4096+"
    assert freq_bucket(3) == "<10" and freq_bucket(50_000) == "10k+"
