"""
Partição por partida, estratificada por competição (seção 2.2.5).
NUNCA por amostragem aleatória de eventos.
"""
import numpy as np
import pandas as pd


def split_games(meta: pd.DataFrame, eval_fraction: float, seed: int,
                strat_col: str = "label") -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (fit_game_ids, eval_game_ids). Dentro de cada competição, uma
    fração eval_fraction das partidas vai para avaliação.
    """
    rng = np.random.default_rng(seed)
    fit, ev = [], []
    for _, grp in meta.groupby(strat_col):
        ids = grp.game_id.to_numpy().copy()
        rng.shuffle(ids)
        n_ev = max(1, int(round(len(ids) * eval_fraction)))
        ev.extend(ids[:n_ev])
        fit.extend(ids[n_ev:])
    return np.array(fit), np.array(ev)


def subset(actions: pd.DataFrame, game_ids) -> pd.DataFrame:
    return actions[actions.game_id.isin(set(game_ids))]


def sample_games(pool: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Sorteia n partidas do pool sem reposição, com semente fixa."""
    rng = np.random.default_rng(seed)
    n = min(n, len(pool))
    return rng.choice(pool, size=n, replace=False)
