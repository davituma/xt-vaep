"""
Etapa A.5 — gera features VAEP (cacheadas), ajusta os classificadores no
pool de ajuste, reporta AUC/Brier no conjunto de avaliação, gera Figura 4.
Rodar: python scripts/04_fit_vaep.py [--rebuild-features]
"""
import argparse
import _bootstrap  # noqa: F401
import pandas as pd
from src.config import load, path
from src.cost import CostTracker
from src.data.corpus import load_corpus
from src.data.splits import subset
from src.models.vaep import (VAEP, build_features, save_features, load_features,
                             features_are_stale, FEATURES)
from src.evaluation import plots

ap = argparse.ArgumentParser(); ap.add_argument("--rebuild-features", action="store_true")
args = ap.parse_args()

cfg = load("vaep")
actions, players, meta = load_corpus()

stale = features_are_stale(actions)
if stale and FEATURES.exists():
    print("cache de features não bate com o corpus atual (data.yaml mudou?) — regenerando.")
if FEATURES.exists() and not args.rebuild_features and not stale:
    X, Y = load_features(); print(f"features do cache: {X.shape}")
else:
    with CostTracker("vaep_features", n_actions=len(actions)):
        X, Y = build_features(actions, meta, cfg["gamestate"]["nb_prev_actions"], cfg["gamestate"]["horizon"])
        save_features(X, Y)
    print(f"features geradas: {X.shape} | rótulos positivos: scores={Y.scores.mean():.4f} concedes={Y.concedes.mean():.4f}")

fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id
ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id
a_fit, a_ev = subset(actions, fit_ids), subset(actions, ev_ids)

with CostTracker("fit_vaep", n_games=len(fit_ids), n_actions=len(a_fit)):
    model = VAEP(cfg["xgboost"]).fit(X.loc[a_fit.index], Y.loc[a_fit.index],
                                     X.loc[a_ev.index], Y.loc[a_ev.index])

print("\n== Tabela 2 — desempenho dos classificadores (conjunto de avaliação) ==")
tab = pd.DataFrame(model.metrics).T
print(tab.round(4).to_string())
tab.to_csv(path("outputs", "tables", "tab02_vaep_classifiers.csv"))

for t in ("scores", "concedes"):
    model.models[t].save_model(str(path("outputs", "models", f"vaep_{t}.json")))
    p = model.models[t].predict_proba(X.loc[a_ev.index])[:, 1]
    plots.calibration(Y.loc[a_ev.index, t], p, f"fig04_calibration_{t}.png", title=f"Calibração — P({t})")

vals = model.rate(a_ev, X.loc[a_ev.index])
print(f"\nVAEP no conjunto de avaliação: média={vals.vaep_value.mean():.4f} | "
      f"p99={vals.vaep_value.quantile(.99):.4f} | máx={vals.vaep_value.max():.4f}")
