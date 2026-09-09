"""
Generalização entre competições (seção 2.2.8).

Para cada partição {fit: filtro, eval: filtro}:
  - modelo EXTERNO: ajusta em todas as partidas de `fit`, valora `eval`
  - modelo INTERNO (referência): ajusta em parte de `eval`, valora o resto de `eval`
  - o ranking externo é comparado ao interno sobre o mesmo subconjunto de avaliação
"""
import json
import pandas as pd

from src.config import load, path
from src.cost import CostTracker
from src.data.splits import subset, split_games
from src.evaluation.stability import compare
from src.experiments.common import fit_and_rank, restrict_players

OUT = path("outputs", "curves", "generalization.jsonl")


def _filter(meta: pd.DataFrame, f: dict):
    m = pd.Series(True, index=meta.index)
    for k, v in f.items():
        m &= meta[k] == v
    return meta[m].game_id.to_numpy()


def run(actions, X, Y, players, meta, out=OUT):
    gcfg, dcfg = load("generalization"), load("data")
    xt_cfg, v_cfg = load("xt"), load("vaep")
    min_min = dcfg["preprocessing"]["min_minutes"]
    k, nb, bs = (load("stability")["metrics"][x] for x in ("top_k", "bootstrap_samples", "bootstrap_seed"))
    rows = []
    for part in gcfg["partitions"]:
        fit_ids = _filter(meta, part["fit"]); ev_all = _filter(meta, part["eval"])
        if len(fit_ids) == 0 or len(ev_all) == 0:
            print(f"pulando {part['name']}: partição vazia"); continue
        # split interno dentro de eval
        ev_meta = meta[meta.game_id.isin(ev_all)]
        int_fit, int_eval = split_games(ev_meta, gcfg["internal_eval_fraction"], gcfg["seed"])
        a_ev = subset(actions, int_eval); pl_ev = restrict_players(players, int_eval)
        Xe, Ye = X.loc[a_ev.index], Y.loc[a_ev.index]
        with CostTracker("generalization", partition=part["name"]):
            a_ext = subset(actions, fit_ids)
            ext_xt, ext_v, d_ext, _, _ = fit_and_rank(a_ext, a_ev, X.loc[a_ext.index], Y.loc[a_ext.index],
                                                     Xe, Ye, pl_ev, min_min, xt_cfg, v_cfg)
            a_int = subset(actions, int_fit)
            int_xt, int_v, d_int, _, _ = fit_and_rank(a_int, a_ev, X.loc[a_int.index], Y.loc[a_int.index],
                                                     Xe, Ye, pl_ev, min_min, xt_cfg, v_cfg)
        for model, ext, ref in (("xt", ext_xt, int_xt), ("vaep", ext_v, int_v)):
            rows.append({"partition": part["name"], "model": model,
                         "n_fit_external": len(fit_ids), "n_fit_internal": len(int_fit),
                         "n_eval": len(int_eval), **compare(ext, ref, k, nb, bs),
                         **{f"ext_{kk}": vv for kk, vv in d_ext.items() if kk.startswith("vaep_")},
                         **{f"int_{kk}": vv for kk, vv in d_int.items() if kk.startswith("vaep_")}})
    df = pd.DataFrame(rows)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return df
