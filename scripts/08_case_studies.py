"""
Etapa A.6 — estudos de caso: jogadores com maior divergência de posto entre
xT e VAEP. Gera Figura 11 (uma por jogador).
Rodar: python scripts/08_case_studies.py [--n 3]
"""
import argparse
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd
from src.config import load, path
from src.data.corpus import load_corpus
from src.data.splits import subset
from src.models import xt as xtmod
from src.models.vaep import VAEP, require_fresh_features
from src.evaluation import plots

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()

df = pd.read_csv(path("outputs", "tables", "rankings_full.csv"), index_col=0)
df["rank_xt"] = df.xt_p90.rank(ascending=False); df["rank_vaep"] = df.vaep_p90.rank(ascending=False)
df["rank_gap"] = (df.rank_xt - df.rank_vaep)
cases = pd.concat([df.nlargest(args.n, "rank_gap"), df.nsmallest(args.n, "rank_gap")])
cases[["player_name", "position", "minutes", "xt_p90", "vaep_p90", "rank_xt", "rank_vaep", "rank_gap"]] \
    .to_csv(path("outputs", "tables", "tab_case_studies.csv"))
print("candidatos a estudo de caso (rank_gap>0: VAEP avalia melhor; <0: xT avalia melhor)")
print(cases[["player_name", "position", "xt_p90", "vaep_p90", "rank_xt", "rank_vaep", "rank_gap"]].round(3).to_string())

# valorar de novo para desenhar as ações
actions, players, meta = load_corpus(); X, Y = require_fresh_features(actions)
fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id
ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id
a_fit, a_ev = subset(actions, fit_ids), subset(actions, ev_ids)
xt = xtmod.from_config(load("xt")).fit(a_fit)
v = VAEP(load("vaep")["xgboost"]).fit(X.loc[a_fit.index], Y.loc[a_fit.index])
xt_v = pd.Series(xt.rate(a_ev), index=a_ev.index); v_v = v.rate(a_ev, X.loc[a_ev.index]).vaep_value
for i, (pid, r) in enumerate(cases.iterrows()):
    pa = a_ev[a_ev.player_id == pid]
    safe = "".join(c for c in str(r.player_name) if c.isalnum())[:20]
    plots.player_actions(pa, xt_v.loc[pa.index], r.player_name, f"fig11_{i}_{safe}_xt.png")
    plots.player_actions(pa, v_v.loc[pa.index], r.player_name, f"fig11_{i}_{safe}_vaep.png")
print(f"figuras fig11_* em outputs/figures/")
