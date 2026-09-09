"""Mantém no log de custo apenas a execução final e regenera a Tabela 7."""
import sys, shutil, pandas as pd
sys.path.insert(0, ".")
from src.cost import LOG

df = pd.read_csv(LOG.with_suffix(".csv.bak"))   # parte do backup íntegro
df["ts"] = pd.to_datetime(df.timestamp)

# A execução boa começa quando as features foram regeneradas pela última vez.
T = df[df.stage == "vaep_features"].ts.max()
print(f"corte: {T}  (última regeneração das features)")

bom = df[df.ts >= T].copy()
# build_corpus é anterior ao corte: pega o último antes dele
corpus = df[(df.stage == "build_corpus") & (df.ts < T)].tail(1)
bom = pd.concat([corpus, bom]).sort_values("ts")

# download é aquisição única, não custo recorrente: reportar à parte
dl = bom[bom.stage == "download"]
bom = bom[bom.stage != "download"]

print("\nlinhas mantidas por estágio:")
print(bom.groupby("stage").size().to_string())
print(f"\ntotal recorrente: {bom.wall_seconds.sum()/3600:.2f} h")
if len(dl):
    print(f"download (à parte): {dl.wall_seconds.sum()/3600:.2f} h")

bom.drop(columns="ts").to_csv(LOG, index=False)
print(f"\nlog reescrito: {len(df)} → {len(bom)} linhas")