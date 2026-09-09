"""
Montagem do corpus: junta as partidas cacheadas, aplica filtros e adiciona
metadados de jogador (nome, posição, minutos). Seção 2.2.2.
"""
import pandas as pd
from tqdm import tqdm

import socceraction.spadl as spadl
from socceraction.spadl import config as spadlcfg

from src.config import load, path
from src.data.loader import _raw_path, _players_path

FIELD_L, FIELD_W = spadlcfg.field_length, spadlcfg.field_width


def _load_game(game_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_parquet(_raw_path(game_id)), pd.read_parquet(_players_path(game_id))


def clean_actions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filtros de pré-processamento declarados na seção 2.2.2."""
    n0 = len(df)
    if cfg["drop_missing_coords"]:
        df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y"])
    if cfg["drop_out_of_bounds"]:
        df = df[df.start_x.between(0, FIELD_L) & df.start_y.between(0, FIELD_W)
                & df.end_x.between(0, FIELD_L) & df.end_y.between(0, FIELD_W)]
    if cfg["drop_missing_player"]:
        df = df.dropna(subset=["player_id"])
    # Disputa de pênaltis (period_id=5) não é jogo aberto: não existe "próximas
    # 10 ações" para rotular, e distorce o VAEP. Prorrogação (3-4) é mantida.
    excl = cfg.get("exclude_periods", [5])
    if excl:
        df = df[~df.period_id.isin(excl)]
    df = df.copy()
    df["player_id"] = df["player_id"].astype(int)
    df["_dropped"] = n0 - len(df)
    return df.reset_index(drop=True)


def build_corpus(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna (actions, players, games_meta).
    - actions: SPADL de todas as partidas, com colunas de metadado da competição
    - players: uma linha por (game_id, player_id) com minutos e posição
    - games_meta: partidas com label/category/gender
    """
    cfg = load("data")["preprocessing"]
    acts, pls, dropped = [], [], 0
    for _, g in tqdm(games.iterrows(), total=len(games), desc="corpus"):
        try:
            a, p = _load_game(int(g.game_id))
        except FileNotFoundError:
            continue
        a = clean_actions(a, cfg)
        dropped += int(a["_dropped"].iloc[0]) if len(a) else 0
        a = a.drop(columns="_dropped")
        a["label"], a["category"], a["gender"] = g.label, g.category, g.gender
        acts.append(a)
        pls.append(p)
    actions = pd.concat(acts, ignore_index=True)
    actions = spadl.add_names(actions)
    players = pd.concat(pls, ignore_index=True)
    print(f"partidas: {actions.game_id.nunique()} | ações: {len(actions):,} | "
          f"descartadas no filtro: {dropped:,}")
    meta = games[["game_id", "home_team_id", "away_team_id", "label", "category", "gender"]].copy()
    return actions, players, meta


def player_minutes(players: pd.DataFrame) -> pd.DataFrame:
    """Minutos totais e posição mais frequente por jogador."""
    agg = players.groupby("player_id").agg(
        player_name=("player_name", "first"),
        minutes=("minutes_played", "sum"),
        games=("game_id", "nunique"),
        position=("starting_position_name", lambda s: s.mode().iat[0] if s.notna().any() else "Unknown"),
        team_id=("team_id", lambda s: s.mode().iat[0]),
    ).reset_index()
    return agg


def position_group(pos: str) -> str:
    """Agrupa posições do StatsBomb em 4 grupos para a análise da seção 2.2.9."""
    if pd.isna(pos):
        return "Unknown"
    p = pos.lower()
    if "goalkeeper" in p:
        return "GK"
    if "back" in p:
        return "DEF"
    if "midfield" in p:
        return "MID"
    if any(k in p for k in ("forward", "wing", "striker", "attacking")):
        return "FWD"
    return "Unknown"


def describe(actions: pd.DataFrame, players: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Tabela 1 do TCC: composição do corpus por competição."""
    g = actions.groupby("label").agg(games=("game_id", "nunique"), actions=("action_id", "size"))
    g = g.join(meta.groupby("label")[["category", "gender"]].first())
    g["actions_per_game"] = (g["actions"] / g["games"]).round(0).astype(int)
    total = pd.DataFrame({"games": [g.games.sum()], "actions": [g.actions.sum()],
                          "category": ["—"], "gender": ["—"],
                          "actions_per_game": [int(g.actions.sum() / g.games.sum())]},
                         index=["TOTAL"])
    return pd.concat([g, total])


# --- cache do corpus montado ---
CORPUS = path("data", "processed", "actions.parquet")
PLAYERS = path("data", "processed", "players.parquet")
META = path("data", "processed", "games_meta.parquet")


def save_corpus(actions, players, meta):
    actions.to_parquet(CORPUS, index=False)
    players.to_parquet(PLAYERS, index=False)
    meta.to_parquet(META, index=False)


def load_corpus():
    return pd.read_parquet(CORPUS), pd.read_parquet(PLAYERS), pd.read_parquet(META)
