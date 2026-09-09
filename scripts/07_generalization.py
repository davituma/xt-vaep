"""
Etapa A.8 — generalização entre competições. Gera Tabela 6.
Rodar: python scripts/07_generalization.py
"""
import _bootstrap  # noqa: F401
from src.config import path
from src.data.corpus import load_corpus
from src.models.vaep import require_fresh_features
from src.experiments import generalization as gen

actions, players, meta = load_corpus(); X, Y = require_fresh_features(actions)
df = gen.run(actions, X, Y, players, meta)
cols = ["partition", "model", "n_fit_external", "n_fit_internal", "n_eval", "n_players",
        "spearman", "spearman_lo", "spearman_hi", "top_k_overlap", "median_rank_shift"]
df[cols].to_csv(path("outputs", "tables", "tab06_generalization.csv"), index=False)
print("== Tabela 6 — generalização entre competições ==\n", df[cols].round(3).to_string(index=False))
