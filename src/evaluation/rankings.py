"""
Agregação de valorações por jogador e linhas de base (seções 2.2.6 e 2.2.9).
"""
import numpy as np
import pandas as pd
from socceraction.spadl import config as spadlcfg

from src.data.corpus import player_minutes, position_group

SHOT_ID = spadlcfg.actiontypes.index("shot")
PEN_ID = spadlcfg.actiontypes.index("shot_penalty")
FK_ID = spadlcfg.actiontypes.index("shot_freekick")
SUCCESS = spadlcfg.results.index("success")


def aggregate(actions: pd.DataFrame, values: np.ndarray | pd.Series, players: pd.DataFrame,
              min_minutes: int, name: str) -> pd.DataFrame:
    """
    Soma o valor por jogador e normaliza por 90 min.
    Retorna DataFrame indexado por player_id com colunas <name>_total, <name>_p90,
    minutes, games, position, pos_group, player_name.
    """
    v = pd.Series(np.asarray(values, dtype=float), index=actions.index).fillna(0.0)
    tot = v.groupby(actions.player_id).sum().rename(f"{name}_total")
    pm = player_minutes(players).set_index("player_id")
    df = pm.join(tot, how="left").fillna({f"{name}_total": 0.0})
    df = df[df.minutes >= min_minutes].copy()
    df[f"{name}_p90"] = df[f"{name}_total"] / df.minutes * 90
    df["pos_group"] = df.position.map(position_group)
    if _exclude_gk():
        df = df[df.pos_group != "GK"]
    return df


def _exclude_gk() -> bool:
    from src.config import load
    return bool(load("data")["preprocessing"].get("exclude_goalkeepers", False))


def baselines(actions: pd.DataFrame, players: pd.DataFrame, min_minutes: int) -> pd.DataFrame:
    """Gols, assistências (aproximação), xG-proxy por chutes, volume de ações."""
    is_shot = actions.type_id.isin([SHOT_ID, PEN_ID, FK_ID])
    goals = (is_shot & (actions.result_id == SUCCESS)).astype(float)
    # Assistência: passe bem-sucedido imediatamente anterior a um gol, mesma equipe
    nxt_goal = goals.shift(-1, fill_value=0.0)
    same_team = actions.team_id.eq(actions.team_id.shift(-1))
    is_pass = actions.type_id.isin([spadlcfg.actiontypes.index(t) for t in ("pass", "cross")])
    assists = (is_pass & (actions.result_id == SUCCESS) & same_team & (nxt_goal == 1)).astype(float)
    shots = is_shot.astype(float)
    volume = pd.Series(1.0, index=actions.index)

    ga = aggregate(actions, goals + assists, players, min_minutes, "ga")
    sh = aggregate(actions, shots, players, min_minutes, "shots")
    vol = aggregate(actions, volume, players, min_minutes, "volume")
    out = ga[["player_name", "minutes", "games", "position", "pos_group", "ga_total", "ga_p90"]]
    out = out.join(sh[["shots_total", "shots_p90"]]).join(vol[["volume_total", "volume_p90"]])
    return out


def attach_xg_baseline(actions: pd.DataFrame, xg: pd.Series, players, min_minutes) -> pd.DataFrame:
    """xG acumulado usando os valores do próprio StatsBomb (quando disponíveis)."""
    return aggregate(actions, xg, players, min_minutes, "xg")[["xg_total", "xg_p90"]]


def rank(df: pd.DataFrame, col: str) -> pd.Series:
    """Posto (1 = melhor) com empates pela média."""
    return df[col].rank(ascending=False, method="average")
