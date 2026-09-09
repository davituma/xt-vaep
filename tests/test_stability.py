"""Testes das métricas de estabilidade."""
import numpy as np, pandas as pd
import sys; sys.path.insert(0, ".")
from src.evaluation.stability import spearman, top_k_overlap, median_rank_shift, bootstrap_ci


def test_identical_rankings():
    a = pd.Series(np.arange(50, dtype=float), index=range(50))
    assert abs(spearman(a, a) - 1.0) < 1e-9 and top_k_overlap(a, a, 20) == 1.0 and median_rank_shift(a, a) == 0.0


def test_reversed_rankings():
    a = pd.Series(np.arange(50, dtype=float), index=range(50))
    assert abs(spearman(a, a[::-1].set_axis(a.index)) + 1.0) < 1e-9


def test_topk_insensitive_to_internal_permutation():
    """Spearman cai com permutação dentro do top-k, mas a sobreposição não."""
    rng = np.random.default_rng(0)
    a = pd.Series(np.arange(100, dtype=float)[::-1], index=range(100))
    b = a.copy(); top = b.index[:20]; b.loc[top] = rng.permutation(b.loc[top].values)
    assert top_k_overlap(a, b, 20) == 1.0 and spearman(a, b) < 1.0


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    a = pd.Series(rng.random(80), index=range(80)); b = a + rng.normal(0, .2, 80)
    ci = bootstrap_ci(a, b, n_boot=200)
    assert ci["spearman_lo"] <= spearman(a, b) <= ci["spearman_hi"]
