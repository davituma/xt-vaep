"""
Etapa A.4 — ajusta o xT no pool de ajuste, valida contra a referência
do socceraction, gera Figuras 2 e 3.
Rodar: python scripts/03_fit_xt.py
"""
import _bootstrap  # noqa: F401
import numpy as np, pandas as pd
import socceraction.xthreat as ref
from src.config import load, path
from src.cost import CostTracker
from src.data.corpus import load_corpus
from src.data.splits import subset
from src.models import xt as xtmod
from src.evaluation import plots

cfg = load("xt")
actions, players, meta = load_corpus()
fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id
a_fit = subset(actions, fit_ids)

with CostTracker("fit_xt", n_games=len(fit_ids)):
    model = xtmod.from_config(cfg).fit(a_fit)

d = model.diag.summary(cfg["sparse_cells"]["min_count_report"])
print(f"convergiu: {model.diag.converged} em {model.diag.n_iterations} iterações (delta={model.diag.final_delta:.2e})")
print(f"monotônico em direção ao gol: {model.is_monotonic_towards_goal()}")
print(f"xT: min={model.xT.min():.4f} max={model.xT.max():.4f} | coluna do gol média={model.xT[:, -1].mean():.4f}")
print(f"N(z): mediana={d['nz_median']:.0f} | células esparsas (<{cfg['sparse_cells']['min_count_report']}): {d['nz_frac_sparse']:.1%}")

# validação contra referência (sem suavização, para comparação justa)
r = ref.ExpectedThreat(l=cfg["grid"]["n_cols"], w=cfg["grid"]["n_rows"]).fit(a_fit)
own_raw = xtmod.ExpectedThreat(cfg["grid"]["n_cols"], cfg["grid"]["n_rows"], smoothing="none").fit(a_fit)
corr = np.corrcoef(own_raw.xT.ravel(), r.xT.ravel())[0, 1]
mad = np.abs(own_raw.xT - r.xT).max()
print(f"validação vs socceraction (sem suavização): corr={corr:.5f} | diff máx={mad:.2e}")
assert corr > 0.999, "implementação diverge da referência — investigar antes de prosseguir"

np.save(path("outputs", "models", "xt_grid.npy"), model.xT)
np.save(path("outputs", "models", "xt_counts.npy"), model.diag.counts_total)
plots.xt_heatmap(model.xT)
plots.nz_heatmap(model.diag.counts_total)
print("figuras: outputs/figures/fig02_xt_heatmap.png, fig03_nz_counts.png")
