"""
Curvas de estabilidade (seção 2.2.7) — o experimento central.

Para cada tamanho n e cada repetição r:
  1. sorteia n partidas do pool de ajuste (semente = seed + hash(n, r))
  2. ajusta xT e VAEP só nessas partidas
  3. valora o conjunto de AVALIAÇÃO (fixo) e ranqueia
  4. compara com o ranking de referência (modelo ajustado no pool inteiro,
     mesmo conjunto de avaliação) pelas três métricas de estabilidade
  5. coleta diagnósticos: N(z) do xT, AUC/Brier do VAEP

Resultados são gravados linha a linha em outputs/curves/stability.jsonl;
o script retoma de onde parou se interrompido.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load, path
from src.cost import CostTracker
from src.data.splits import subset, sample_games
from src.evaluation.stability import all_metrics
from src.experiments.common import fit_and_rank, restrict_players

OUT = path("outputs", "curves", "stability.jsonl")


def _done_keys(out=OUT) -> set:
    if not out.exists():
        return set()
    with open(out, encoding="utf-8") as f:
        return {(json.loads(l)["n_games"], json.loads(l)["rep"]) for l in f if l.strip()}


def run(actions, X, Y, players, fit_ids, eval_ids, quick=False, out=OUT):
    scfg, dcfg = load("stability"), load("data")
    xt_cfg, v_cfg = load("xt"), load("vaep")
    min_min = dcfg["preprocessing"]["min_minutes"]
    k = scfg["metrics"]["top_k"]
    sizes = scfg["sample_sizes"]; reps = scfg["n_repetitions"]
    if quick:
        sizes, reps = sizes[:2], 2
    sizes = [s for s in sizes if s <= len(fit_ids)]
    if len(sizes) < len(scfg["sample_sizes"]) and not quick:
        print(f"AVISO: pool de ajuste tem {len(fit_ids)} partidas; tamanhos acima disso foram removidos.")

    a_eval = subset(actions, eval_ids); pl_eval = restrict_players(players, eval_ids)
    X_eval, Y_eval = X.loc[a_eval.index], Y.loc[a_eval.index]

    # ---- referência: ajuste no pool completo ----
    with CostTracker("stability_reference", n_games=len(fit_ids)):
        a_ref = subset(actions, fit_ids)
        ref_xt, ref_v, ref_diag, _, _ = fit_and_rank(
            a_ref, a_eval, X.loc[a_ref.index], Y.loc[a_ref.index], X_eval, Y_eval,
            pl_eval, min_min, xt_cfg, v_cfg)
    ref_diag.update({"n_games": len(fit_ids), "rep": -1, "model": "reference"})
    pd.Series(ref_diag).to_json(path("outputs", "curves", "reference_diag.json"))

    done = _done_keys(out)
    total = sum(1 for n in sizes for r in range(reps) if (n, r) not in done)
    pbar = tqdm(total=total, desc="stability curves")
    with open(out, "a", encoding="utf-8") as f:
        for n in sizes:
            for r in range(reps):
                if (n, r) in done:
                    continue
                seed = scfg["seed"] * 1000 + n * 10 + r
                ids = sample_games(fit_ids, n, seed)
                a_fit = subset(actions, ids)
                with CostTracker("stability_point", n_games=n, rep=r):
                    rk_xt, rk_v, diag, _, _ = fit_and_rank(
                        a_fit, a_eval, X.loc[a_fit.index], Y.loc[a_fit.index],
                        X_eval, Y_eval, pl_eval, min_min, xt_cfg, v_cfg)
                base = {"n_games": n, "rep": r, "seed": int(seed), **diag}
                for model, rk, ref in (("xt", rk_xt, ref_xt), ("vaep", rk_v, ref_v)):
                    row = {**base, "model": model, **all_metrics(rk, ref, k)}
                    f.write(json.dumps(row) + "\n"); f.flush()
                pbar.update(1)
    pbar.close()
    return load_results(out)


def load_results(out=OUT) -> pd.DataFrame:
    if not out.exists() or out.stat().st_size == 0:
        raise FileNotFoundError(
            f"{out} não existe ou está vazio. Rode as curvas antes de usar --plots-only:\n"
            f"  python scripts/06_stability_curves.py")
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela 5 do TCC: média e desvio por (model, n_games)."""
    m = ["spearman", "top_k_overlap", "median_rank_shift"]
    g = df.groupby(["model", "n_games"])[m].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    d = df[df.model == "xt"].groupby("n_games")[["nz_frac_sparse", "nz_median", "xt_iterations"]].mean()
    v = df[df.model == "vaep"].groupby("n_games")[["vaep_scores_auc", "vaep_scores_brier",
                                                    "vaep_concedes_auc", "vaep_concedes_brier"]].mean()
    return g.join(d, on="n_games").join(v, on="n_games").round(4)
