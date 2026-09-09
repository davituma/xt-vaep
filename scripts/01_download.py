"""
Etapa A.2 — baixa e converte para SPADL todas as partidas de configs/data.yaml.
Rodar: python scripts/01_download.py [--force]
Retoma automaticamente; partidas já em cache são puladas.
"""
import argparse
import _bootstrap  # noqa: F401
from src.cost import CostTracker
from src.data.loader import list_games, fetch_all
from src.config import path

ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
args = ap.parse_args()

games = list_games()
games.to_parquet(path("data", "processed", "games_list.parquet"), index=False)
print(games.groupby("label").size().to_string())
print(f"\ntotal: {len(games)} partidas")
with CostTracker("download", n_games=len(games)):
    fetch_all(games, force=args.force)
