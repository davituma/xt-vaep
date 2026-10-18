"""
Curvas de estabilidade com variantes do VAEP (seção 2.2.7b).

Para cada tamanho n e repetição r — com as MESMAS sementes do experimento
principal, de modo que os subconjuntos sorteados sejam idênticos —
ajusta e compara:

  xt            Expected Threat (referência de robustez)
  vaep          VAEP com hiperparâmetros fixos (idêntico ao experimento principal)
  vaep_reg      VAEP com capacidade ajustada ao volume (tabela ou CV por partida)
  vaep_shrink   VAEP fixo + encolhimento das notas por minutos
  vaep_reg_shrink  as duas correções juntas

A pergunta que responde: quanto da instabilidade do VAEP sob amostra
pequena é corrigível por regularização e/ou encolhimento?

Grava em outputs/curves/stability_variants.jsonl; retoma se interrompido.
"""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import load, path
from src.cost import CostTracker
from src.data.splits import subset, sample_games
from src.models import xt as xtmod
from src.models.vaep import VAEP, params_for_n, select_params_cv
from src.evaluation.rankings import aggregate, shrink_by_minutes
from src.evaluation.stability import all_metrics
from src.experiments.common import restrict_players

OUT = path("outputs", "curves", "stability_variants.jsonl")
MODELS = ["xt", "vaep", "vaep_reg", "vaep_shrink", "vaep_reg_shrink"]


def _done(out) -> set:
    if not out.exists():
        return set()
    return {(json.loads(l)["n_games"], json.loads(l)["rep"]) for l in open(out, encoding="utf-8") if l.strip()}


def _fit_all(a_fit, a_eval, X, Y, X_eval, pl_eval, min_min, xt_cfg, v_cfg, var_cfg, mode, K):
    """Ajusta xT e as variantes de VAEP; devolve dict nome -> Series (p90) e diag."""
    n_games = a_fit.game_id.nunique()
    diag = {}
    # xT
    xt = xtmod.from_config(xt_cfg).fit(a_fit)
    r_xt = aggregate(a_eval, xt.rate(a_eval), pl_eval, min_min, "xt")
    diag.update(xt.diag.summary(xt_cfg["sparse_cells"]["min_count_report"]))

    Xf, Yf = X.loc[a_fit.index], Y.loc[a_fit.index]
    base = v_cfg["xgboost"]

    # VAEP fixo
    v0 = VAEP(base).fit(Xf, Yf)
    r_v0 = aggregate(a_eval, v0.rate(a_eval, X_eval)["vaep_value"], pl_eval, min_min, "vaep")

    # VAEP regularizado por tamanho
    if mode == "cv":
        params = select_params_cv(base, var_cfg["schedule"], Xf, Yf, a_fit.game_id,
                                  var_cfg["cv"]["n_folds"], var_cfg["cv"]["select_by"])
    else:
        params = params_for_n(base, var_cfg["schedule"], n_games)
    diag.update({f"reg_{k}": params[k] for k in ("n_estimators", "max_depth", "min_child_weight")})
    v1 = VAEP(params).fit(Xf, Yf)
    r_v1 = aggregate(a_eval, v1.rate(a_eval, X_eval)["vaep_value"], pl_eval, min_min, "vaep")

    out = {
        "xt": r_xt["xt_p90"],
        "vaep": r_v0["vaep_p90"],
        "vaep_reg": r_v1["vaep_p90"],
        "vaep_shrink": shrink_by_minutes(r_v0, "vaep_p90", K),
        "vaep_reg_shrink": shrink_by_minutes(r_v1, "vaep_p90", K),
    }
    return out, diag


def run(actions, X, Y, players, fit_ids, eval_ids, mode="schedule", quick=False, out=OUT):
    scfg, dcfg = load("stability"), load("data")
    xt_cfg, v_cfg, var_cfg = load("xt"), load("vaep"), load("vaep_variants")
    min_min = dcfg["preprocessing"]["min_minutes"]
    k = scfg["metrics"]["top_k"]; K = var_cfg["shrinkage"]["K_minutes"]
    sizes = [s for s in scfg["sample_sizes"] if s <= len(fit_ids)]
    reps = scfg["n_repetitions"]
    if quick:
        sizes, reps = sizes[:2], 2

    a_eval = subset(actions, eval_ids); pl_eval = restrict_players(players, eval_ids)
    X_eval = X.loc[a_eval.index]

    with CostTracker("variants_reference", n_games=len(fit_ids), mode=mode):
        ref, _ = _fit_all(subset(actions, fit_ids), a_eval, X, Y, X_eval, pl_eval,
                          min_min, xt_cfg, v_cfg, var_cfg, mode, K)

    done = _done(out)
    todo = [(n, r) for n in sizes for r in range(reps) if (n, r) not in done]
    pbar = tqdm(total=len(todo), desc=f"variants ({mode})")
    with open(out, "a", encoding="utf-8") as f:
        for n, r in todo:
            seed = scfg["seed"] * 1000 + n * 10 + r          # MESMA semente do experimento principal
            a_fit = subset(actions, sample_games(fit_ids, n, seed))
            with CostTracker("variants_point", n_games=n, rep=r, mode=mode):
                rk, diag = _fit_all(a_fit, a_eval, X, Y, X_eval, pl_eval,
                                    min_min, xt_cfg, v_cfg, var_cfg, mode, K)
            base = {"n_games": n, "rep": r, "seed": int(seed), "mode": mode, **diag}
            for m in MODELS:
                f.write(json.dumps({**base, "model": m, **all_metrics(rk[m], ref[m], k)}) + "\n")
            f.flush(); pbar.update(1)
    pbar.close()
    return load_results(out)


def load_results(out=OUT) -> pd.DataFrame:
    if not out.exists() or out.stat().st_size == 0:
        raise FileNotFoundError(f"{out} vazio — rode: python scripts/12_stability_variants.py")
    return pd.DataFrame([json.loads(l) for l in open(out, encoding="utf-8") if l.strip()])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela 9: estabilidade por modelo e tamanho, com o gap em relação ao VAEP fixo."""
    m = ["spearman", "top_k_overlap", "median_rank_shift"]
    g = df.groupby(["model", "n_games"])[m].agg(["mean", "std"])
    g.columns = ["_".join(c) for c in g.columns]
    # ganho de Spearman de cada variante sobre o VAEP fixo, no mesmo n
    base = g.xs("vaep", level="model")["spearman_mean"]
    g["spearman_gain_vs_vaep"] = [
        g.loc[(mo, n), "spearman_mean"] - base.loc[n] for mo, n in g.index]
    return g.round(4)
