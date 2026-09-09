"""
Etapa A.7 — curvas de estabilidade (resultado central).
Rodar primeiro:  python scripts/06_stability_curves.py --quick   (estima tempo)
Depois:          python scripts/06_stability_curves.py
Retoma automaticamente. Gera Tabela 5 e Figuras 6-10.
"""
import argparse
import _bootstrap  # noqa: F401
import pandas as pd
from src.config import path
from src.data.corpus import load_corpus
from src.models.vaep import require_fresh_features
from src.experiments import stability_curves as sc
from src.evaluation import plots

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
ap.add_argument("--plots-only", action="store_true")
args = ap.parse_args()

if not args.plots_only:
    actions, players, meta = load_corpus(); X, Y = require_fresh_features(actions)
    fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id.to_numpy()
    ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id.to_numpy()
    out = path("outputs", "curves", "stability_quick.jsonl") if args.quick else sc.OUT
    df = sc.run(actions, X, Y, players, fit_ids, ev_ids, quick=args.quick, out=out)
else:
    df = sc.load_results()

tab = sc.summarize(df)
tab.to_csv(path("outputs", "tables", "tab05_stability.csv"))
print("\n== Tabela 5 — estabilidade por tamanho de amostra ==\n", tab.to_string())

if not args.quick:
    plots.stability_curves(df, "spearman", "fig06_stability_spearman.png", "Spearman vs. referência")
    plots.stability_curves(df, "top_k_overlap", "fig07_stability_topk.png", "sobreposição top-20")
    plots.stability_curves(df, "median_rank_shift", "fig08_stability_rankshift.png", "deslocamento mediano de posto")
    plots.diagnostic_curve(df, "vaep_scores_auc", "fig09a_vaep_auc.png", "AUC-ROC (P_scores)", model="vaep")
    plots.diagnostic_curve(df, "vaep_scores_brier", "fig09b_vaep_brier.png", "Brier (P_scores)", model="vaep")
    plots.diagnostic_curve(df, "nz_frac_sparse", "fig10_xt_sparse_cells.png", "fração de células esparsas", model="xt")
    print("figuras 6-10 em outputs/figures/")
