"""Funções compartilhadas pelos experimentos: ajustar e ranquear ambos os modelos."""
import numpy as np
import pandas as pd

from src.config import load
from src.data.splits import subset
from src.models import xt as xtmod
from src.models.vaep import VAEP
from src.evaluation.rankings import aggregate


def fit_and_rank(actions_fit, actions_eval, X_fit, Y_fit, X_eval, Y_eval, players_eval,
                 min_minutes: int, xt_cfg: dict, vaep_cfg: dict, with_vaep_metrics=True):
    """
    Ajusta xT e VAEP em `actions_fit`, valora `actions_eval`, agrega por
    jogador. Retorna (rank_xt_p90, rank_vaep_p90, diagnostics_dict).
    """
    # ---- xT ----
    xt = xtmod.from_config(xt_cfg).fit(actions_fit)
    r_xt = aggregate(actions_eval, xt.rate(actions_eval), players_eval, min_minutes, "xt")
    diag = xt.diag.summary(xt_cfg["sparse_cells"]["min_count_report"])
    diag["xt_monotonic"] = xt.is_monotonic_towards_goal()

    # ---- VAEP ----
    v = VAEP(vaep_cfg["xgboost"]).fit(X_fit, Y_fit, X_eval if with_vaep_metrics else None,
                                       Y_eval if with_vaep_metrics else None)
    vals = v.rate(actions_eval, X_eval)
    r_v = aggregate(actions_eval, vals["vaep_value"], players_eval, min_minutes, "vaep")
    diag.update(v.flat_metrics())
    return r_xt["xt_p90"], r_v["vaep_p90"], diag, xt, v


def restrict_players(players: pd.DataFrame, game_ids) -> pd.DataFrame:
    return players[players.game_id.isin(set(game_ids))]
