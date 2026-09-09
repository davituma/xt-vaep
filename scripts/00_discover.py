"""
Etapa A.2 — relatório de cobertura do acervo aberto.
Rodar: python scripts/00_discover.py
Saída: outputs/tables/coverage_report.csv
Use o relatório para decidir a lista em configs/data.yaml.
"""
import _bootstrap  # noqa: F401
from src.data.loader import discover
from src.config import path

df = discover()
out = path("outputs", "tables", "coverage_report.csv")
df.to_csv(out, index=False)
print(df.to_string(index=False))
print(f"\nsalvo em {out}")
