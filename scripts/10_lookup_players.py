"""
Consulta jogadores nos rankings — sanity check qualitativo.

    python scripts/10_lookup_players.py Messi Neymar Modric
    python scripts/10_lookup_players.py --pos FWD --top 15
    python scripts/10_lookup_players.py --famous
    python scripts/10_lookup_players.py --divergent 15

Busca é por substring, sem acento e sem diferenciar maiúsculas,
então "Modric" acha "Luka Modrić" e "ibrahimo" acha "Zlatan Ibrahimović".
"""
import argparse, sys, unicodedata
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import path

FAMOUS = ["Messi", "Neymar", "Modric", "Cristiano Ronaldo", "Iniesta", "Griezmann",
          "Bale", "Kroos", "Busquets", "Ibrahimovic", "Higuain", "Dybala", "Pogba",
          "Hazard", "Aguero", "Kante", "Sergio Ramos", "Jordi Alba", "Mbappe",
          "Lewandowski", "Suarez Diaz", "Alarcon"]

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower()

def load():
    df = pd.read_csv(path("outputs", "tables", "rankings_full.csv"), index_col=0)
    df["rank_xt"] = df.xt_p90.rank(ascending=False).astype(int)
    df["rank_vaep"] = df.vaep_p90.rank(ascending=False).astype(int)
    df["gap"] = df.rank_xt - df.rank_vaep      # >0: VAEP avalia melhor
    df["_key"] = df.player_name.map(strip_accents)
    return df

COLS = ["player_name", "position", "games", "minutes",
        "xt_p90", "rank_xt", "vaep_p90", "rank_vaep", "gap", "ga_total"]

def show(d, title):
    if d.empty:
        print(f"\n{title}: nenhum jogador encontrado."); return
    print(f"\n=== {title} (n={len(d)}) ===")
    print(d[COLS].round(3).to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="trechos de nome a procurar")
    ap.add_argument("--famous", action="store_true", help="lista pré-definida de craques")
    ap.add_argument("--pos", help="filtra por grupo: GK, DEF, MID, FWD")
    ap.add_argument("--top", type=int, help="mostra os N melhores por VAEP/90")
    ap.add_argument("--by", default="vaep_p90", help="ordenar por (default vaep_p90)")
    ap.add_argument("--divergent", type=int, metavar="N",
                    help="os N com maior divergência de posto entre os modelos")
    ap.add_argument("--min-minutes", type=int, default=0)
    ap.add_argument("--report", action="store_true",
                    help="modo pipeline: gera outputs/tables/tab08_lookup.csv")
    a = ap.parse_args()

    df = load()
    if a.min_minutes:
        df = df[df.minutes >= a.min_minutes]
    if a.pos:
        df = df[df.pos_group == a.pos.upper()]

    if a.report:
        keys = [strip_accents(n) for n in FAMOUS]
        fam = df[df._key.str.contains("|".join(keys), na=False)].copy()
        fam["grupo"] = "referência conhecida"
        div = pd.concat([df.nlargest(10, "gap"), df.nsmallest(10, "gap")]).copy()
        div["grupo"] = "maior divergência"
        out = pd.concat([fam, div]).drop_duplicates(subset="player_name")
        out = out[["grupo"] + COLS].sort_values(["grupo", "rank_vaep"])
        dest = path("outputs", "tables", "tab08_lookup.csv")
        out.to_csv(dest, index=False)
        show(fam.sort_values("rank_vaep"), "jogadores de referência")
        show(div.sort_values("gap", ascending=False), "maior divergência entre modelos")
        print(f"\nsalvo em {dest}")
        return

    if a.divergent:
        d = pd.concat([df.nlargest(a.divergent, "gap"), df.nsmallest(a.divergent, "gap")])
        show(d.sort_values("gap", ascending=False),
             "maior divergência (gap>0: VAEP avalia melhor; gap<0: xT avalia melhor)")
    elif a.top:
        show(df.nlargest(a.top, a.by), f"top {a.top} por {a.by}"
             + (f" — {a.pos.upper()}" if a.pos else ""))
    else:
        alvos = a.names or (FAMOUS if a.famous else [])
        if not alvos:
            ap.print_help(); return
        keys = [strip_accents(n) for n in alvos]
        d = df[df._key.str.contains("|".join(keys), na=False)]
        show(d.sort_values("rank_vaep"), "jogadores encontrados")
        if not d.empty:
            print("\nlembrete: rankings são calculados sobre o conjunto de AVALIAÇÃO "
                  "(25% das partidas), por isso o número de jogos por atleta é baixo.")

if __name__ == "__main__":
    main()
