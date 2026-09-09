"""
Etapa A.2/A.3 — monta o corpus, aplica filtros, gera Tabela 1 e o split.
Rodar: python scripts/02_build_corpus.py
"""
import _bootstrap  # noqa: F401
import pandas as pd
from src.config import load, path
from src.cost import CostTracker
from src.data.corpus import build_corpus, describe, save_corpus
from src.data.splits import split_games

games = pd.read_parquet(path("data", "processed", "games_list.parquet"))
with CostTracker("build_corpus"):
    actions, players, meta = build_corpus(games)
    save_corpus(actions, players, meta)

tab = describe(actions, players, meta)
tab.to_csv(path("outputs", "tables", "tab01_corpus.csv"))
print("\n== Tabela 1 — composição do corpus ==")
print(tab.to_string())

cfg = load("data")["split"]
fit_ids, eval_ids = split_games(meta, cfg["eval_fraction"], cfg["seed"])
pd.DataFrame({"game_id": fit_ids, "split": "fit"}).to_parquet(path("data", "processed", "split_fit.parquet"))
pd.DataFrame({"game_id": eval_ids, "split": "eval"}).to_parquet(path("data", "processed", "split_eval.parquet"))
print(f"\nsplit: {len(fit_ids)} partidas de ajuste | {len(eval_ids)} de avaliação")
print(f"→ ajuste configs/stability.yaml: sample_sizes deve ir até ~{len(fit_ids)}")

# validação: gols no SPADL vs placar oficial (etapa A.3)
from socceraction.spadl import config as c
goals = actions[(actions.type_id.isin([c.actiontypes.index(t) for t in ("shot","shot_penalty","shot_freekick")]))
                & (actions.result_id == c.results.index("success"))]
og = actions[actions.result_id == c.results.index("owngoal")]
spadl_goals = goals.groupby("game_id").size().add(og.groupby("game_id").size(), fill_value=0)
official = (games.set_index("game_id").home_score + games.set_index("game_id").away_score)
chk = pd.DataFrame({"spadl": spadl_goals, "official": official}).dropna()
mism = chk[chk.spadl != chk.official]
print(f"\nvalidação de gols: {len(chk)-len(mism)}/{len(chk)} partidas batem com o placar oficial")
if len(mism):
    mism.to_csv(path("outputs", "logs", "goal_mismatch.csv"))
    print(f"  {len(mism)} divergências salvas em outputs/logs/goal_mismatch.csv (normal: gols contra e pênaltis em disputa)")
