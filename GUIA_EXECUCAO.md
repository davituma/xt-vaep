# Guia de execução — passo a passo

Instruções para rodar o projeto do zero em Windows, macOS ou Linux.
Não precisa de GPU. Precisa de internet só no download dos dados.

---

## 0. Pré-requisitos

| O quê | Versão | Como verificar |
|---|---|---|
| Python | 3.10, 3.11 ou 3.12 | `python --version` (Windows) / `python3 --version` (mac/Linux) |
| pip | qualquer recente | `python -m pip --version` |
| Git | opcional | só se for clonar em vez de descompactar |
| Espaço em disco | ~3 GB | dados brutos + features |
| RAM | 4 GB mínimo, 8 GB confortável | — |

**Python 3.13 não funciona** (socceraction 1.5.3 depende de numpy<2).
Se sua máquina só tem 3.13, instale 3.12 em paralelo: https://www.python.org/downloads/

---

## 1. Descompactar e entrar na pasta

```bash
# descompacte tcc-xt-vaep.zip onde quiser, depois:
cd tcc-xt-vaep
```

Todos os comandos abaixo são executados **de dentro dessa pasta**.

---

## 2. Criar o ambiente virtual

Isolar as dependências evita conflito com outros projetos.

**Windows (PowerShell ou cmd):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Você sabe que deu certo quando o prompt mostra `(.venv)` no início.
**Repita o `activate` toda vez que abrir um terminal novo.**

---

## 3. Instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Leva 2–5 minutos. Se der erro em `xgboost` no macOS Apple Silicon:
```bash
brew install libomp
python -m pip install -r requirements.txt
```

---

## 4a. Atalho: rodar tudo de uma vez

Se não quiser seguir passo a passo, depois do item 3:

```bash
python run_all.py --quick   # primeiro: validação com corpus pequeno (~10 min)
python run_all.py           # depois: pipeline completo
```

O `run_all.py` chama os scripts na ordem, para se algum falhar, e ajusta
`sample_sizes` sozinho. Se preferir controlar cada etapa, siga os passos
abaixo — o resultado é o mesmo.

**Atenção no Windows:** sempre use `python scripts/xxx.py`. Digitar só
`scripts/xxx.py` não executa nada (o PowerShell tenta "abrir" o arquivo).

## 4. Verificar a instalação

```bash
python -m pytest tests -q
```

Esperado: `8 passed`. Se falhar, não siga — cole o erro e resolva antes.

---

## 5. Ver o que existe no acervo aberto

```bash
python scripts/00_discover.py
```

Leva ~2 min. Gera `outputs/tables/coverage_report.csv` com o número de
partidas de cada competição-temporada disponível no StatsBomb Open Data.

**Abra esse CSV** e compare com a lista em `configs/data.yaml`:
- Copas do Mundo, Euros, Copa América, AFCON → devem ter 32–64 partidas cada (completas).
- Ligas de clubes → verifique se o número bate com uma temporada inteira
  (Bundesliga 306, FA WSL 132, NWSL ~180). Se vier bem menor, a cobertura é
  parcial: **remova a linha** de `data.yaml` ou mantenha e declare como parcial
  na seção 2.2.2 do TCC.

---

## 6. Baixar os dados

```bash
python scripts/01_download.py
```

~1,2 s por partida. Mil partidas ≈ 20 min. Pode interromper (Ctrl+C) e
rodar de novo: ele pula o que já baixou. Erros ficam em
`outputs/logs/download_errors.csv`.

---

## 7. Montar o corpus

```bash
python scripts/02_build_corpus.py
```

Imprime a **Tabela 1** e uma linha assim:

```
split: 830 partidas de ajuste | 277 de avaliação
→ ajuste configs/stability.yaml: sample_sizes deve ir até ~830
```

**Abra `configs/stability.yaml`** e ajuste `sample_sizes` para terminar perto
desse número. Ex.: com 830, use `[8, 16, 32, 64, 128, 256, 512, 800]`.

Também confira a linha `validação de gols: X/Y partidas batem`. Se muitas não
baterem, há problema na conversão — investigue antes de seguir.

---

## 8. Ajustar os modelos

```bash
python scripts/03_fit_xt.py
python scripts/04_fit_vaep.py
```

O `03` valida sua implementação contra a do socceraction (deve imprimir
`corr=1.00000`) e gera as Figuras 2 e 3.
O `04` gera as features (demora alguns minutos na primeira vez, depois usa
cache), treina os classificadores e imprime a **Tabela 2** (AUC, Brier).

Faixa esperada de AUC: 0,70–0,85. Muito acima de 0,90 indica vazamento —
verifique o split.

---

## 9. Comparar os modelos

```bash
python scripts/05_compare.py
```

Gera **Tabelas 3, 4** e **Figura 5**. Rankings completos em
`outputs/tables/rankings_full.csv`.

---

## 10. Curvas de estabilidade (resultado central)

**Primeiro estime o tempo:**
```bash
python scripts/06_stability_curves.py --quick
```

Roda só 2 tamanhos × 2 repetições. Olhe a linha `[cost] stability_point`
em `outputs/logs/cost.csv` para saber os segundos por ponto.

Custo total ≈ (número de tamanhos) × (repetições) × (segundos por ponto).
Se der mais de 2 h, reduza `n_repetitions` para 3 em `configs/stability.yaml`.

**Depois rode completo:**
```bash
python scripts/06_stability_curves.py
```

Grava linha a linha em `outputs/curves/stability.jsonl` — pode interromper e
retomar. Gera **Tabela 5** e **Figuras 6 a 10**.

Se quiser só refazer as figuras sem recalcular:
```bash
python scripts/06_stability_curves.py --plots-only
```

---

## 11. Generalização, estudos de caso, custo

```bash
python scripts/07_generalization.py    # Tabela 6
python scripts/08_case_studies.py      # Figura 11 (uma por jogador)
python scripts/09_cost_report.py       # Tabela 7 + environment.csv
```

---

## Atalho: Makefile

Se tiver `make` instalado (Linux/mac nativo; Windows via Git Bash ou WSL):

```bash
make test discover
# edite data.yaml
make download corpus
# edite stability.yaml
make xt vaep compare quick
make curves gen cases cost
```

---

## Onde ficam as saídas

```
outputs/figures/   fig02 … fig11  (PNG 300 dpi, prontas para o Word)
outputs/tables/    tab01 … tab07  (CSV)
outputs/logs/      cost.csv, download_errors.csv, goal_mismatch.csv
outputs/curves/    stability.jsonl, generalization.jsonl
```

---

## Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `ImportError: cannot import name 'overload' from 'multimethod'` | multimethod 2.x | `pip install "multimethod<2"` |
| `AttributeError: module 'numpy' has no attribute 'NaN'` | numpy 2.x | `pip install "numpy<2"` |
| `ModuleNotFoundError: No module named 'src'` | rodou de fora da pasta | `cd tcc-xt-vaep` antes |
| Download muito lento / timeouts | rede | rode de novo; ele retoma |
| `06` travou no meio | qualquer coisa | rode de novo; retoma do `.jsonl` |
| AUC > 0,95 | vazamento | verifique se o split está por partida (`split_fit.parquet`) |
| Figura em branco | matplotlib sem backend | já forçado `Agg` no código; reinstale matplotlib |

---

## Reprodutibilidade

Todas as sementes estão em `configs/*.yaml` (`seed: 42`). Rodando com as
mesmas configs, mesma versão das bibliotecas (`requirements.txt`) e mesmo
acervo, os resultados são idênticos. O `09_cost_report.py` grava as versões
em `outputs/tables/environment.csv` — inclua no Apêndice A.
