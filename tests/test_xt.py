"""Testes do xT: convergência, monotonicidade, equivalência com a referência."""
import numpy as np, pandas as pd, pytest
import sys; sys.path.insert(0, ".")
from socceraction.spadl import config as c
from src.models.xt import ExpectedThreat


def synth(n=4000, seed=0):
    """Ações sintéticas: passes progressivos e chutes perto do gol."""
    rng = np.random.default_rng(seed)
    sx = rng.uniform(0, 105, n); sy = rng.uniform(0, 68, n)
    ex = np.clip(sx + rng.normal(8, 10, n), 0, 105); ey = np.clip(sy + rng.normal(0, 8, n), 0, 68)
    is_shot = (sx > 85) & (rng.random(n) < .3)
    type_id = np.where(is_shot, c.actiontypes.index("shot"), c.actiontypes.index("pass"))
    goal = is_shot & (rng.random(n) < (sx - 85) / 20 * .4)
    res = np.where(goal | (~is_shot & (rng.random(n) < .8)), c.results.index("success"), c.results.index("fail"))
    return pd.DataFrame(dict(start_x=sx, start_y=sy, end_x=ex, end_y=ey, type_id=type_id, result_id=res))


def test_converges_and_monotonic():
    m = ExpectedThreat().fit(synth())
    assert m.diag.converged and m.diag.n_iterations < 200
    assert m.xT.min() >= 0 and m.xT.max() <= 1
    assert m.is_monotonic_towards_goal()


def test_matches_reference_without_smoothing():
    import socceraction.xthreat as ref
    a = synth()
    # Mesma tolerância da referência (eps=1e-5) para comparação justa;
    # com tolerâncias distintas a diferença é ~1e-5 (artefato de convergência).
    own = ExpectedThreat(smoothing="none", tolerance=1e-5).fit(a)
    r = ref.ExpectedThreat(l=12, w=8, eps=1e-5).fit(a)
    assert np.allclose(own.xT, r.xT, atol=1e-4)
    assert np.corrcoef(own.xT.ravel(), r.xT.ravel())[0, 1] > 0.99999


def test_rate_only_successful_moves():
    a = synth(); m = ExpectedThreat().fit(a)
    v = m.rate(a)
    mask = (a.type_id == c.actiontypes.index("pass")) & (a.result_id == c.results.index("success"))
    assert np.isfinite(v[mask]).all() and np.isnan(v[~mask]).all()


def test_smoothing_reduces_zero_cells():
    a = synth(n=300)  # poucas ações → células vazias
    raw = ExpectedThreat(smoothing="none").fit(a); sm = ExpectedThreat(alpha=1.0).fit(a)
    assert (sm.goal_prob > 0).sum() >= (raw.goal_prob > 0).sum()
