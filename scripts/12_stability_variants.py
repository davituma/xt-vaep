"""
Etapa A.7b — sensibilidade a hiperparâmetros e encolhimento (seção 2.2.7b).
Responde: a instabilidade do VAEP sob amostra pequena é intrínseca ou corrigível?

    python scripts/12_stability_variants.py --quick          # estima o tempo (2 tamanhos × 2 reps)
    python scripts/12_stability_variants.py                  # modo 'schedule' (tabela por tamanho)
    python scripts/12_stability_variants.py --mode cv        # modo CV por partida (mais lento, mais rigoroso)
    python scripts/12_stability_variants.py --plots-only

Usa as MESMAS sementes do experimento principal, logo os subconjuntos
sorteados são idênticos aos das Figuras 6-8. Gera Tabela 9 e Figuras 13-14.
"""
import argparse
import _bootstrap  # noqa: F401
import pandas as pd
from src.config import path
from src.data.corpus import load_corpus
from src.models.vaep import require_fresh_features
from src.experiments import stability_variants as sv
from src.evaluation import plots

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["schedule", "cv"], default="schedule")
ap.add_argument("--quick", action="store_true")
ap.add_argument("--plots-only", action="store_true")
a = ap.parse_args()

if not a.plots_only:
    actions, players, meta = load_corpus(); X, Y = require_fresh_features(actions)
    fit_ids = pd.read_parquet(path("data", "processed", "split_fit.parquet")).game_id.to_numpy()
    ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id.to_numpy()
    out = path("outputs", "curves", f"stability_variants{'_quick' if a.quick else ''}_{a.mode}.jsonl")
    df = sv.run(actions, X, Y, players, fit_ids, ev_ids, mode=a.mode, quick=a.quick, out=out)
else:
    out = path("outputs", "curves", f"stability_variants_{a.mode}.jsonl")
    df = sv.load_results(out)

tab = sv.summarize(df)
tab.to_csv(path("outputs", "tables", f"tab09_variants_{a.mode}.csv"))
print(f"\n== Tabela 9 — estabilidade das variantes ({a.mode}) ==")
print(tab[["spearman_mean", "spearman_std", "top_k_overlap_mean", "spearman_gain_vs_vaep"]].to_string())

if not a.quick:
    order = ["xt", "vaep", "vaep_reg", "vaep_shrink", "vaep_reg_shrink"]
    plots.stability_curves(df[df.model.isin(order)], "spearman",
                           f"fig13_variants_spearman_{a.mode}.png", "Spearman vs. referência")
    plots.stability_curves(df[df.model.isin(order)], "top_k_overlap",
                           f"fig14_variants_topk_{a.mode}.png", "sobreposição top-20")
    print(f"figuras: outputs/figures/fig13_variants_spearman_{a.mode}.png, fig14_variants_topk_{a.mode}.png")
    hp = df[df.model == "vaep_reg"].groupby("n_games")[["reg_n_estimators", "reg_max_depth", "reg_min_child_weight"]].agg(lambda s: s.mode().iat[0])
    print("\nhiperparâmetros efetivamente usados pelo vaep_reg, por tamanho:")
    print(hp.to_string())
