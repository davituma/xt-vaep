"""
VAEP — Decroos et al. (2019), via socceraction + XGBoost (seção 2.2.5).

Features e rótulos vêm das funções de referência da biblioteca; a
classificação usa XGBoost. Features são geradas uma vez e cacheadas,
porque não dependem do split — só o ajuste dos classificadores depende.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import socceraction.vaep.formula as vf

from src.config import path

XFNS = [fs.actiontype_onehot, fs.result_onehot, fs.bodypart_onehot,
        fs.startlocation, fs.endlocation, fs.movement, fs.space_delta,
        fs.startpolar, fs.endpolar, fs.team, fs.time_delta, fs.time]

FEATURES = path("data", "processed", "vaep_features.parquet")
LABELS = path("data", "processed", "vaep_labels.parquet")


def build_features(actions: pd.DataFrame, meta: pd.DataFrame, nb_prev: int, horizon: int
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera X (features) e Y (rótulos) para todas as ações, partida a partida.
    Índice alinhado com `actions` (mesma ordem de linhas).
    """
    home = meta.set_index("game_id")["home_team_id"].to_dict()
    Xs, Ys = [], []
    for gid, ga in tqdm(actions.groupby("game_id", sort=False), desc="vaep features"):
        ga = ga.sort_values(["period_id", "time_seconds", "action_id"])
        gs = fs.gamestates(ga, nb_prev_actions=nb_prev)
        gs = fs.play_left_to_right(gs, home[gid])
        X = pd.concat([f(gs) for f in XFNS], axis=1)
        Y = pd.concat([lab.scores(ga, horizon), lab.concedes(ga, horizon)], axis=1)
        X.index, Y.index = ga.index, ga.index
        Xs.append(X); Ys.append(Y)
    X = pd.concat(Xs).loc[actions.index]
    Y = pd.concat(Ys).loc[actions.index]
    return X, Y


def save_features(X, Y):
    X.to_parquet(FEATURES); Y.to_parquet(LABELS)


def load_features():
    return pd.read_parquet(FEATURES), pd.read_parquet(LABELS)


def features_are_stale(actions: pd.DataFrame) -> bool:
    """
    True se o cache de features não corresponde ao corpus atual.
    Acontece quando configs/data.yaml muda e o corpus é remontado.
    """
    if not FEATURES.exists():
        return True
    try:
        X = pd.read_parquet(FEATURES, columns=[])
    except Exception:
        return True
    return not X.index.equals(actions.index)


def require_fresh_features(actions: pd.DataFrame):
    """
    Carrega as features garantindo que batem com o corpus. Se não baterem,
    levanta erro com instrução clara em vez do KeyError gigante do pandas.
    """
    if features_are_stale(actions):
        raise RuntimeError(
            "O cache de features do VAEP não corresponde ao corpus atual "
            "(provavelmente configs/data.yaml mudou desde a última geração).\n"
            "Regenere com:\n"
            "    python scripts/04_fit_vaep.py --rebuild-features\n"
            "ou apague data/processed/vaep_features.parquet e rode de novo.")
    return load_features()


class VAEP:
    def __init__(self, xgb_params: dict):
        self.params = dict(xgb_params)
        self.models = {}
        self.metrics = {}

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, X_eval=None, Y_eval=None) -> "VAEP":
        for target in ("scores", "concedes"):
            m = xgb.XGBClassifier(**self.params)
            m.fit(X, Y[target])
            self.models[target] = m
            if X_eval is not None:
                p = m.predict_proba(X_eval)[:, 1]
                y = Y_eval[target]
                self.metrics[target] = {
                    "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
                    "brier": float(brier_score_loss(y, p)),
                    "logloss": float(log_loss(y, p, labels=[0, 1])),
                    "pos_rate": float(y.mean()),
                }
        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return (self.models["scores"].predict_proba(X)[:, 1],
                self.models["concedes"].predict_proba(X)[:, 1])

    def rate(self, actions: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica a fórmula VAEP por partida (a diferença de estado é intra-partida)."""
        ps, pc = self.predict(X)
        ps = pd.Series(ps, index=actions.index); pc = pd.Series(pc, index=actions.index)
        out = []
        for gid, ga in actions.groupby("game_id", sort=False):
            ga = ga.sort_values(["period_id", "time_seconds", "action_id"])
            v = vf.value(ga, ps.loc[ga.index], pc.loc[ga.index])
            v.index = ga.index
            out.append(v)
        return pd.concat(out).loc[actions.index]

    def flat_metrics(self) -> dict:
        return {f"vaep_{t}_{k}": v for t, d in self.metrics.items() for k, v in d.items()}
