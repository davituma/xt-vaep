"""
Aquisição do StatsBomb Open Data e conversão para SPADL (seções 2.2.2-2.2.3).

Usa o StatsBombLoader do socceraction em modo remoto (lê direto do GitHub).
Cada partida é convertida e cacheada individualmente em data/raw/<game_id>.parquet,
de modo que uma interrupção não obriga a recomeçar.
"""
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

from src.config import load, path

warnings.filterwarnings("ignore")

_LOADER = None


def loader() -> StatsBombLoader:
    global _LOADER
    if _LOADER is None:
        _LOADER = StatsBombLoader(getter="remote", creds={"user": None, "passwd": None})
    return _LOADER


def discover() -> pd.DataFrame:
    """
    Relatório de cobertura: quantas partidas cada competição-temporada tem
    no acervo aberto. Alimenta a decisão de composição do corpus.
    """
    L = loader()
    comps = L.competitions()
    rows = []
    for _, c in tqdm(comps.iterrows(), total=len(comps), desc="discover"):
        try:
            g = L.games(c.competition_id, c.season_id)
            n = len(g)
        except Exception as e:  # noqa: BLE001
            n = -1
        rows.append({
            "competition_id": c.competition_id, "season_id": c.season_id,
            "competition_name": c.competition_name, "season_name": c.season_name,
            "country": c.country_name, "gender": c.competition_gender, "n_games": n,
        })
    return pd.DataFrame(rows).sort_values(["competition_name", "season_name"])


def list_games() -> pd.DataFrame:
    """Lista todas as partidas das competições configuradas, com metadados."""
    cfg = load("data")
    L = loader()
    parts = []
    for c in cfg["competitions"]:
        g = L.games(c["competition_id"], c["season_id"])
        g = g.assign(label=c["label"], category=c["category"], gender=c["gender"])
        parts.append(g)
    games = pd.concat(parts, ignore_index=True)
    return games


def _raw_path(game_id: int) -> Path:
    return path("data", "raw", f"{game_id}.parquet")


def _players_path(game_id: int) -> Path:
    return path("data", "raw", f"{game_id}_players.parquet")


def fetch_game(game_id: int, home_team_id: int, force: bool = False) -> bool:
    """
    Baixa eventos de uma partida, converte para SPADL orientado (ataque da
    esquerda para a direita) e cacheia. Retorna True se processou.
    """
    p = _raw_path(game_id)
    pp = _players_path(game_id)
    if p.exists() and pp.exists() and not force:
        return False
    L = loader()
    events = L.events(game_id)
    actions = spadl.statsbomb.convert_to_actions(events, home_team_id)
    actions = spadl.play_left_to_right(actions, home_team_id)
    actions.to_parquet(p, index=False)
    L.players(game_id).to_parquet(pp, index=False)
    return True


def fetch_all(games: pd.DataFrame, force: bool = False) -> None:
    """Baixa todas as partidas listadas, com retomada automática."""
    done, err = 0, []
    for _, g in tqdm(games.iterrows(), total=len(games), desc="download"):
        try:
            if fetch_game(int(g.game_id), int(g.home_team_id), force):
                done += 1
        except Exception as e:  # noqa: BLE001
            err.append((g.game_id, str(e)[:120]))
    print(f"processadas: {done} | já em cache: {len(games) - done - len(err)} | erros: {len(err)}")
    if err:
        pd.DataFrame(err, columns=["game_id", "error"]).to_csv(
            path("outputs", "logs", "download_errors.csv"), index=False)
        print("erros salvos em outputs/logs/download_errors.csv")
