"""
Etapa A.7 — Tabela 7: custo de execução por estágio.
Rodar: python scripts/09_cost_report.py
"""
import _bootstrap  # noqa: F401
import platform, sys, pandas as pd
from src.config import path
from src.cost import LOG

df = pd.read_csv(LOG)
tab = df.groupby("stage").agg(runs=("wall_seconds", "size"), total_s=("wall_seconds", "sum"),
                              mean_s=("wall_seconds", "mean"), max_rss_mb=("rss_mb", "max")).round(1)
tab.loc["TOTAL"] = [tab.runs.sum(), tab.total_s.sum(), float("nan"), tab.max_rss_mb.max()]
tab.to_csv(path("outputs", "tables", "tab07_cost.csv"))
print(tab.to_string())
print(f"\ntempo total: {tab.loc['TOTAL','total_s']/3600:.2f} h")
env = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu": platform.processor() or "n/a"}
for m in ("numpy", "pandas", "xgboost", "socceraction", "sklearn"):
    try:
        env[m] = __import__(m).__version__
    except Exception:
        pass
pd.Series(env).to_csv(path("outputs", "tables", "environment.csv"))
print("\nambiente:", env)
