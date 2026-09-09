"""
Métricas de estabilidade de rankings (seção 2.2.6) com IC por bootstrap.

Três métricas, todas entre um ranking "reduzido" e um "de referência"
sobre o MESMO conjunto de jogadores:
  - spearman: correlação de postos (deslocamento geral)
  - top_k_overlap: |top_k(a) ∩ top_k(b)| / k (estabilidade dos destaques)
  - median_rank_shift: mediana de |posto_a - posto_b| (magnitude típica)

O termo aqui é ESTABILIDADE, não acurácia: o ranking de referência é
ele próprio uma estimativa.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx = a.index.intersection(b.index)
    return a.loc[idx], b.loc[idx]


def spearman(a: pd.Series, b: pd.Series) -> float:
    a, b = _align(a, b)
    if len(a) < 3:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def top_k_overlap(a: pd.Series, b: pd.Series, k: int = 20) -> float:
    a, b = _align(a, b)
    k = min(k, len(a))
    if k == 0:
        return float("nan")
    ta = set(a.nlargest(k).index); tb = set(b.nlargest(k).index)
    return len(ta & tb) / k


def median_rank_shift(a: pd.Series, b: pd.Series) -> float:
    a, b = _align(a, b)
    if len(a) == 0:
        return float("nan")
    ra = a.rank(ascending=False); rb = b.rank(ascending=False)
    return float((ra - rb).abs().median())


def all_metrics(a: pd.Series, b: pd.Series, k: int = 20) -> dict:
    return {"spearman": spearman(a, b), "top_k_overlap": top_k_overlap(a, b, k),
            "median_rank_shift": median_rank_shift(a, b), "n_players": len(_align(a, b)[0])}


def bootstrap_ci(a: pd.Series, b: pd.Series, k: int = 20, n_boot: int = 500, seed: int = 42,
                 ci: float = 0.95) -> dict:
    """
    IC por bootstrap sobre jogadores (reamostra jogadores com reposição,
    recalcula as três métricas). Retorna dict com *_lo e *_hi.
    """
    a, b = _align(a, b)
    rng = np.random.default_rng(seed)
    idx = a.index.to_numpy()
    res = {"spearman": [], "top_k_overlap": [], "median_rank_shift": []}
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        s = pd.Index(s).unique()          # nlargest/rank precisam de índice único
        if len(s) < 3:
            continue
        aa, bb = a.loc[s], b.loc[s]
        res["spearman"].append(spearman(aa, bb))
        res["top_k_overlap"].append(top_k_overlap(aa, bb, k))
        res["median_rank_shift"].append(median_rank_shift(aa, bb))
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for m, vals in res.items():
        v = np.array(vals, dtype=float); v = v[np.isfinite(v)]
        out[f"{m}_lo"] = float(np.quantile(v, lo)) if len(v) else float("nan")
        out[f"{m}_hi"] = float(np.quantile(v, hi)) if len(v) else float("nan")
    return out


def compare(a: pd.Series, b: pd.Series, k=20, n_boot=500, seed=42) -> dict:
    d = all_metrics(a, b, k)
    d.update(bootstrap_ci(a, b, k, n_boot, seed))
    return d
