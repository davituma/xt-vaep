#!/usr/bin/env python
"""
Executa o pipeline completo em um único comando.

    python run_all.py                 # tudo, do download ao relatório de custo
    python run_all.py --from corpus   # retoma a partir de uma etapa
    python run_all.py --only curves   # roda só uma etapa
    python run_all.py --quick         # corpus de teste (data.test.yaml) + curvas reduzidas
    python run_all.py --list          # mostra as etapas

Funciona em Windows, macOS e Linux. Cada etapa é um script em scripts/;
se uma falhar, o pipeline para e mostra o erro. Etapas já concluídas
(download, features) são detectadas e puladas automaticamente pelos
próprios scripts, então rodar de novo é seguro.

Antes das curvas, ajusta configs/stability.yaml automaticamente para que
sample_sizes chegue até o tamanho real do pool de ajuste (desligue com
--no-auto-sizes se quiser controlar manualmente).
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("test",     "python -m pytest tests -q",                  "testes unitários"),
    ("discover", "scripts/00_discover.py",                     "relatório de cobertura do acervo"),
    ("download", "scripts/01_download.py",                     "download + conversão SPADL (retomável)"),
    ("corpus",   "scripts/02_build_corpus.py",                 "montar corpus, split, Tabela 1"),
    ("xt",       "scripts/03_fit_xt.py",                       "ajustar xT, validar, Fig. 2-3"),
    ("vaep",     "scripts/04_fit_vaep.py",                     "features + classificadores, Tabela 2, Fig. 4"),
    ("compare",  "scripts/05_compare.py",                      "rankings e concordância, Tabelas 3-4, Fig. 5"),
    ("quick",    "scripts/06_stability_curves.py --quick",     "estimativa de tempo das curvas"),
    ("curves",   "scripts/06_stability_curves.py",             "curvas de estabilidade, Tabela 5, Fig. 6-10"),
    ("gen",      "scripts/07_generalization.py",               "generalização entre competições, Tabela 6"),
    ("cases",    "scripts/08_case_studies.py",                 "estudos de caso, Fig. 11"),
    ("lookup",   "scripts/10_lookup_players.py --report",      "validação de construto por jogador, Tabela 8"),
    ("cost",     "scripts/09_cost_report.py",                  "Tabela 7 + ambiente"),
]
NAMES = [s[0] for s in STEPS]


def banner(msg):
    print("\n" + "=" * 78 + f"\n  {msg}\n" + "=" * 78, flush=True)


def run(cmd: str) -> None:
    if cmd.startswith("python "):
        argv = [PY] + cmd.split()[1:]
    else:
        argv = [PY] + cmd.split()
    r = subprocess.run(argv, cwd=ROOT)
    if r.returncode != 0:
        print(f"\n!! etapa falhou (código {r.returncode}).", file=sys.stderr)
        print("   O erro está nas linhas acima. Se a saída for muito longa, rode de novo com:",
              file=sys.stderr)
        print("     python run_all.py 2>&1 | Tee-Object -FilePath log.txt   (PowerShell)",
              file=sys.stderr)
        print("   Depois de corrigir, retome com: python run_all.py --from <etapa>", file=sys.stderr)
        sys.exit(r.returncode)


def check_features() -> None:
    """Avisa cedo se o cache de features não bate com o corpus."""
    try:
        sys.path.insert(0, str(ROOT))
        from src.data.corpus import load_corpus, CORPUS
        from src.models.vaep import features_are_stale, FEATURES
        if not CORPUS.exists() or not FEATURES.exists():
            return
        actions, _, _ = load_corpus()
        if features_are_stale(actions):
            print("[aviso] cache de features desatualizado; será regenerado na etapa vaep.")
    except Exception:
        pass


def auto_sizes() -> None:
    """Lê o pool de ajuste e ajusta sample_sizes para cobrir até ele."""
    split = ROOT / "data" / "processed" / "split_fit.parquet"
    if not split.exists():
        return
    import pandas as pd
    n_fit = len(pd.read_parquet(split))
    cfg_path = ROOT / "configs" / "stability.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    sizes = [8]
    while sizes[-1] * 2 <= n_fit:
        sizes.append(sizes[-1] * 2)
    # inclui o pool inteiro se o último ponto ficou longe dele (>25% abaixo)
    if sizes[-1] < 0.75 * n_fit:
        sizes.append(n_fit)
    if cfg.get("sample_sizes") != sizes:
        cfg["sample_sizes"] = sizes
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        print(f"[auto] pool de ajuste = {n_fit} partidas → sample_sizes = {sizes}")
    else:
        print(f"[auto] sample_sizes já cobre o pool ({n_fit}): {sizes}")


def swap_config(test: bool) -> None:
    """--quick usa data.test.yaml; guarda o data.yaml original em data.yaml.bak."""
    cfg, tst, bak = (ROOT / "configs" / f for f in ("data.yaml", "data.test.yaml", "data.yaml.bak"))
    if test:
        if not bak.exists():
            shutil.copy(cfg, bak)
        shutil.copy(tst, cfg)
        print("[quick] usando configs/data.test.yaml (original salvo em data.yaml.bak)")
    elif bak.exists():
        shutil.move(bak, cfg)
        print("[quick] configs/data.yaml restaurado")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=NAMES, help="começar a partir desta etapa")
    ap.add_argument("--to", dest="end", choices=NAMES, help="parar nesta etapa (inclusive)")
    ap.add_argument("--only", choices=NAMES, help="rodar só esta etapa")
    ap.add_argument("--skip", nargs="*", default=[], choices=NAMES, help="etapas a pular")
    ap.add_argument("--quick", action="store_true", help="corpus de teste + curvas reduzidas")
    ap.add_argument("--no-auto-sizes", action="store_true", help="não ajustar sample_sizes automaticamente")
    ap.add_argument("--list", action="store_true", help="listar etapas e sair")
    a = ap.parse_args()

    if a.list:
        for n, _, d in STEPS:
            print(f"  {n:9s} {d}")
        return

    names = NAMES[:]
    if a.only:
        names = [a.only]
    else:
        if a.start:
            names = names[names.index(a.start):]
        if a.end:
            names = names[: names.index(a.end) + 1]
    # discover é informativo; quick é só estimativa. Por padrão ambos pulam no
    # fluxo completo (rode explicitamente com --only se quiser).
    default_skip = {"discover", "quick"} if not a.only else set()
    if a.quick:
        default_skip.discard("quick"); default_skip.add("curves")
    names = [n for n in names if n not in set(a.skip) | default_skip]

    if a.quick:
        swap_config(True)
    t0 = time.time()
    try:
        for name, cmd, desc in STEPS:
            if name not in names:
                continue
            if name == "vaep":
                check_features()
            if name in ("curves", "quick") and not a.no_auto_sizes and not a.quick:
                auto_sizes()
            banner(f"[{name}] {desc}")
            t = time.time()
            run(cmd)
            print(f"\n  ✓ {name} concluído em {(time.time()-t)/60:.1f} min", flush=True)
    finally:
        if a.quick:
            swap_config(False)
    banner(f"pipeline concluído em {(time.time()-t0)/60:.1f} min — saídas em outputs/")


if __name__ == "__main__":
    main()
