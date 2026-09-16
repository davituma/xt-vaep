"""
Figuras de ações para jogadores escolhidos pelo nome.

    python scripts/11_player_figures.py Messi Neymar Higuain
    python scripts/11_player_figures.py --famous
    python scripts/11_player_figures.py Messi --top-n 40 --prefix fig12

Carrega os modelos já treinados de outputs/models/ em vez de reajustar,
então roda em cerca de um minuto. Se os modelos não existirem, rode antes:
    python run_all.py --only xt
    python run_all.py --only vaep
"""
import argparse, sys, unicodedata
from pathlib import Path

import numpy as np, pandas as pd
import xgboost as xgb

import _bootstrap  # noqa: F401
from src.config import load, path
from src.data.corpus import load_corpus
from src.data.splits import subset
from src.models import xt as xtmod
from src.models.vaep import VAEP, require_fresh_features
from src.evaluation import plots

FAMOUS = ["Messi", "Neymar", "Modric", "Cristiano Ronaldo", "Iniesta", "Higuain",
          "Hazard", "Aguero", "Kroos", "Mbappe"]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower()


def load_models():
    """Carrega xT e VAEP já treinados. Devolve (xt, vaep)."""
    grid = path("outputs", "models", "xt_grid.npy")
    ms = path("outputs", "models", "vaep_scores.json")
    mc = path("outputs", "models", "vaep_concedes.json")
    faltam = [p.name for p in (grid, ms, mc) if not p.exists()]
    if faltam:
        sys.exit(f"modelos ausentes em outputs/models/: {', '.join(faltam)}\n"
                 "rode antes:  python run_all.py --only xt   e   python run_all.py --only vaep")

    xt = xtmod.from_config(load("xt"))
    xt.xT = np.load(grid)
    if xt.xT.shape != (xt.w, xt.l):
        sys.exit(f"grade salva {xt.xT.shape} não bate com configs/xt.yaml ({xt.w},{xt.l}) — "
                 "reajuste o xT ou alinhe a config.")

    v = VAEP(load("vaep")["xgboost"])
    for alvo, arq in (("scores", ms), ("concedes", mc)):
        m = xgb.XGBClassifier()
        m.load_model(str(arq))
        v.models[alvo] = m
    return xt, v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="trechos de nome (sem acento, sem caixa)")
    ap.add_argument("--famous", action="store_true")
    ap.add_argument("--top-n", type=int, default=25, help="ações desenhadas por figura")
    ap.add_argument("--prefix", default="fig12", help="prefixo dos arquivos gerados")
    ap.add_argument("--min-minutes", type=int, default=0)
    a = ap.parse_args()

    alvos = a.names or (FAMOUS if a.famous else [])
    if not alvos:
        ap.print_help(); return

    rk = pd.read_csv(path("outputs", "tables", "rankings_full.csv"), index_col=0)
    rk["rank_xt"] = rk.xt_p90.rank(ascending=False).astype(int)
    rk["rank_vaep"] = rk.vaep_p90.rank(ascending=False).astype(int)
    rk["_key"] = rk.player_name.map(strip_accents)
    keys = [strip_accents(n) for n in alvos]
    sel = rk[rk._key.str.contains("|".join(keys), na=False)]
    if a.min_minutes:
        sel = sel[sel.minutes >= a.min_minutes]
    if sel.empty:
        sys.exit("nenhum jogador encontrado com esses termos.")

    print(f"{len(sel)} jogador(es):")
    print(sel[["player_name", "position", "minutes", "rank_xt", "rank_vaep"]].to_string())

    actions, players, meta = load_corpus()
    X, _ = require_fresh_features(actions)
    ev_ids = pd.read_parquet(path("data", "processed", "split_eval.parquet")).game_id
    a_ev = subset(actions, ev_ids)

    xt, v = load_models()
    xt_v = pd.Series(xt.rate(a_ev), index=a_ev.index)
    v_v = v.rate(a_ev, X.loc[a_ev.index]).vaep_value

    feitas = []
    for pid, r in sel.iterrows():
        pa = a_ev[a_ev.player_id == pid]
        if pa.empty:
            print(f"  ! {r.player_name}: sem ações no conjunto de avaliação"); continue
        safe = "".join(c for c in strip_accents(r.player_name) if c.isalnum())[:22]
        for tag, vals in (("xt", xt_v), ("vaep", v_v)):
            nome = f"{a.prefix}_{safe}_{tag}.png"
            titulo = f"{r.player_name} — {tag.upper()} (xT {r.rank_xt}º / VAEP {r.rank_vaep}º)"
            plots.player_actions(pa, vals.loc[pa.index], titulo, nome, top_n=a.top_n)
            feitas.append(nome)
    print(f"\n{len(feitas)} figuras em outputs/figures/:")
    for f in feitas:
        print("  ", f)


if __name__ == "__main__":
    main()
