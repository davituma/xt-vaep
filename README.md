# Complexidade Amostral em Modelos de Valoração de Ações no Futebol

Implementação do TCC — Bacharelado em Ciência da Computação, CESUPA, 2026.
Compara a **estabilidade** de Expected Threat (xT) e VAEP sob variação do volume
de partidas de ajuste, com StatsBomb Open Data e bibliotecas de código aberto.

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                            # 8 testes devem passar
```

Requer Python 3.10–3.12. `numpy<2` é obrigatório (socceraction 1.5.3).

## Comando único

```bash
python run_all.py            # tudo: download → corpus → modelos → curvas → relatórios
python run_all.py --quick    # validação rápida com corpus de teste (~10 min)
python run_all.py --from curves     # retomar de uma etapa
python run_all.py --only compare    # rodar só uma etapa
python run_all.py --list            # ver as etapas
```

Funciona em Windows (PowerShell), macOS e Linux. Ajusta `sample_sizes`
automaticamente ao tamanho do pool de ajuste antes das curvas. Se uma
etapa falhar, para e mostra onde; corrija e retome com `--from`.

## Pipeline (etapa a etapa)

| # | Script | Etapa (Apêndice A) | Produz |
|---|---|---|---|
| 00 | `00_discover.py` | A.2 | `coverage_report.csv` — partidas por competição no acervo |
| 01 | `01_download.py` | A.2 | `data/raw/<game>.parquet` (SPADL por partida, com retomada) |
| 02 | `02_build_corpus.py` | A.2/A.3 | corpus, split, **Tabela 1**, validação de gols |
| 03 | `03_fit_xt.py` | A.4 | xT + validação vs. referência, **Fig. 2–3** |
| 04 | `04_fit_vaep.py` | A.5 | features (cache), classificadores, **Tabela 2, Fig. 4** |
| 05 | `05_compare.py` | A.6 | rankings, concordância, linhas de base, **Tabelas 3–4, Fig. 5** |
| 06 | `06_stability_curves.py` | A.7 | **curvas de estabilidade — Tabela 5, Fig. 6–10** |
| 07 | `07_generalization.py` | A.8 | **Tabela 6** |
| 08 | `08_case_studies.py` | A.6 | **Fig. 11** (uma por jogador) |
| 09 | `09_cost_report.py` | A.7 | **Tabela 7** + `environment.csv` |

```bash
make discover          # 1. veja o relatório, ajuste configs/data.yaml
make download corpus   # 2. baixa (~1,2 s/partida) e monta o corpus
make xt vaep compare   # 3. modelos e comparação
make quick             # 4. estima o tempo das curvas
make curves gen cases cost   # 5. experimentos e relatórios
```

Ou `make all` depois de ajustar as configs. Todos os scripts retomam de onde
pararam; `06` grava linha a linha em `outputs/curves/stability.jsonl`.

## Configuração

Tudo que é decisão metodológica está em `configs/`, nunca no código:

- `data.yaml` — competições, filtros, limiar de minutos, split. **Marcado `[AJUSTAR]`.**
- `xt.yaml` — grade, solver, tratamento de células esparsas (suavização aditiva).
- `vaep.yaml` — horizonte, hiperparâmetros do XGBoost.
- `stability.yaml` — tamanhos de amostra e repetições. **Ajuste após `02`.**
- `generalization.yaml` — partições fit/eval.
- `data.test.yaml` — corpus reduzido (3 torneios, 114 partidas) usado para
  validar o pipeline. Copie sobre `data.yaml` para um teste rápido.

## Teste de referência (corpus reduzido)

Com `data.test.yaml` (114 partidas), o pipeline completo roda em **~8 min** e
**<1 GB RAM** em CPU. Números abaixo são só de validação do pipeline, não
resultados do TCC:

| Etapa | Tempo |
|---|---|
| download (114 jogos) | 142 s |
| features VAEP (229k ações) | 14 s |
| ajuste xT | 0,3 s |
| ajuste VAEP | 21 s |
| ponto da curva (xT+VAEP) | ~8 s |

## Estrutura

```
src/
  data/       loader (StatsBomb→SPADL), corpus, splits
  models/     xt.py (implementação própria + diagnósticos N(z)), vaep.py (socceraction+XGBoost)
  evaluation/ rankings, stability (3 métricas + bootstrap), plots
  experiments/ stability_curves, generalization
scripts/      00–09, na ordem
tests/        xT (convergência, monotonicidade, equivalência com referência), métricas
outputs/      figures/ tables/ curves/ logs/cost.csv
```

## Decisões de implementação que importam para o texto

- **Matriz de transição do xT** normalizada pelo total de movimentos *tentados*
  (sucessos + falhas), como em Singh (2019) e na referência. Linhas somam <1;
  o valor "perdido" em passes interceptados é modelado implicitamente.
- **Split por partida, estratificado por competição.** Nunca por evento.
- **Disputa de pênaltis (period 5) excluída**; prorrogação mantida.
- **Estabilidade ≠ acurácia.** O ranking de referência é uma estimativa. O
  código e as figuras usam "estabilidade" e "vs. referência", nunca "acurácia".
- **Assistência** é aproximada como passe/cruzamento bem-sucedido imediatamente
  anterior a um gol da mesma equipe (SPADL não tem o rótulo nativo).
- **xG** do StatsBomb não é carregado no SPADL; a linha de base de finalização
  usa contagem de chutes. Se quiser xG real, é preciso extrair do JSON bruto
  (`shot.statsbomb_xg`) — pendência opcional.

## Licença

MIT. Dados: StatsBomb Open Data (github.com/statsbomb/open-data), uso
não comercial.
