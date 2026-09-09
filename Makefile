# Pipeline completo. Rode os alvos na ordem, ou `make all`.
PY=python
.PHONY: all setup discover download corpus xt vaep compare quick curves gen cases cost test clean

setup:
	pip install -r requirements.txt

discover:
	$(PY) scripts/00_discover.py

download:
	$(PY) scripts/01_download.py

corpus:
	$(PY) scripts/02_build_corpus.py

xt:
	$(PY) scripts/03_fit_xt.py

vaep:
	$(PY) scripts/04_fit_vaep.py

compare:
	$(PY) scripts/05_compare.py

quick:            # estima tempo das curvas antes da execução completa
	$(PY) scripts/06_stability_curves.py --quick

curves:
	$(PY) scripts/06_stability_curves.py

gen:
	$(PY) scripts/07_generalization.py

cases:
	$(PY) scripts/08_case_studies.py

cost:
	$(PY) scripts/09_cost_report.py

test:
	$(PY) -m pytest tests -q

all: download corpus xt vaep compare curves gen cases cost

clean:
	rm -rf data/processed/* outputs/models/* outputs/curves/*.jsonl
