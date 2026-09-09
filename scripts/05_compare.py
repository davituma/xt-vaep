"""
Etapa A.6 — rankings sobre o conjunto de avaliação, concordância entre
modelos, análise por posição/tipo de ação, linhas de base.
Gera Tabelas 3 e 4 e Figura 5.
Rodar: python scripts/05_compare.py
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd
from src.config import load, path
from src.cost import CostTracker
from src.data.corpus import load_corpus
from src.data.splits import subset
from src.models import xt as xtmod
from src.models.vaep import VAEP, require_fresh_features
from src.evaluation.rankings import aggregate, baselines
from src.evaluation.stability import compare, spearman
from src.evaluation import plots

dcfg, xcfg, vcfg, scfg = load("data"), load("xt"), load("vaep"), load("stability")
mm, k, nb, bs = dcfg["preprocessing"]["min_minutes"], scfg["metrics"]["top_k"], \
    scfg["metrics"]["bootstrap_samples"], scfg["metrics"]["bootstrap_seed"]
actions, players, meta = load_corpus(); X, Y = require_fresh_features(actions)
fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id
ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id
a_fit, a_ev = subset(actions, fit_ids), subset(actions, ev_ids)
pl_ev = players[players.game_id.isin(set(ev_ids))]

with CostTracker("compare"):
    xt = xtmod.from_config(xcfg).fit(a_fit)
    v = VAEP(vcfg["xgboost"]).fit(X.loc[a_fit.index], Y.loc[a_fit.index])
    xt_vals = xt.rate(a_ev)
    v_vals = v.rate(a_ev, X.loc[a_ev.index])

r_xt = aggregate(a_ev, xt_vals, pl_ev, mm, "xt")
r_v = aggregate(a_ev, v_vals["vaep_value"], pl_ev, mm, "vaep")
bl = baselines(a_ev, pl_ev, mm)
df = bl.join(r_xt[["xt_total", "xt_p90"]]).join(r_v[["vaep_total", "vaep_p90"]])
df.to_csv(path("outputs", "tables", "rankings_full.csv"))

# Tabela 3 — top-20 por cada modelo
cols = ["player_name", "position", "minutes", "games", "xt_total", "xt_p90", "vaep_total", "vaep_p90", "ga_total"]
top = pd.concat({"xT": df.nlargest(k, "xt_p90")[cols], "VAEP": df.nlargest(k, "vaep_p90")[cols]})
top.to_csv(path("outputs", "tables", "tab03_top20.csv"))
print(f"== Tabela 3 — top {k} por VAEP/90 ==\n", df.nlargest(k, "vaep_p90")[cols].round(3).to_string())

# Tabela 4 — concordância + linhas de base
rows = {"xT vs VAEP": compare(df.xt_p90, df.vaep_p90, k, nb, bs)}
for name, col in (("gols+assist", "ga_p90"), ("chutes", "shots_p90"), ("volume", "volume_p90")):
    rows[f"xT vs {name}"] = {"spearman": spearman(df.xt_p90, df[col])}
    rows[f"VAEP vs {name}"] = {"spearman": spearman(df.vaep_p90, df[col])}
tab4 = pd.DataFrame(rows).T
tab4.to_csv(path("outputs", "tables", "tab04_concordance.csv"))
print("\n== Tabela 4 — concordância e linhas de base ==\n", tab4.round(3).to_string())

# por posição
pos = df.groupby("pos_group").apply(lambda g: pd.Series({"n": len(g), "spearman": spearman(g.xt_p90, g.vaep_p90)}))
pos.to_csv(path("outputs", "tables", "tab04b_by_position.csv"))
print("\n== concordância por posição ==\n", pos.round(3).to_string())

# por tipo de ação: distribuição de valores em cada modelo
vt = pd.DataFrame({"type": a_ev.type_name.values, "xt": xt_vals, "vaep": v_vals.vaep_value.values})
by_type = vt.groupby("type").agg(n=("xt", "size"), xt_mean=("xt", "mean"), xt_nonnull=("xt", lambda s: s.notna().mean()),
                                 vaep_mean=("vaep", "mean"), vaep_abs_mean=("vaep", lambda s: s.abs().mean()))
by_type.to_csv(path("outputs", "tables", "tab04c_by_action_type.csv"))
print("\n== por tipo de ação ==\n", by_type.round(4).to_string())

plots.scatter_models(df)
print("\nfigura: outputs/figures/fig05_scatter_xt_vaep.png")
