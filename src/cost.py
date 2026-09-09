"""
Registro de custo computacional (seção 2.2.10 / 2.3.9 do TCC).

Uso:
    from src.cost import CostTracker
    with CostTracker("fit_xt", n_games=128):
        ...

Cada bloco grava uma linha em outputs/logs/cost.csv com tempo de parede,
pico de memória residente (se psutil disponível) e metadados livres.
Coletar isso desde o início; não dá para reconstituir depois.
"""
import csv
import time
from datetime import datetime
from src.config import path

try:
    import psutil
    _PROC = psutil.Process()
except ImportError:  # psutil é opcional
    _PROC = None

LOG = path("outputs", "logs", "cost.csv")
_FIELDS = ["timestamp", "stage", "wall_seconds", "rss_mb", "meta"]


class CostTracker:
    def __init__(self, stage: str, **meta):
        self.stage = stage
        self.meta = meta

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        wall = time.perf_counter() - self.t0
        rss = _PROC.memory_info().rss / 1e6 if _PROC else float("nan")
        new = not LOG.exists()
        with open(LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "stage": self.stage,
                "wall_seconds": round(wall, 3),
                "rss_mb": round(rss, 1),
                "meta": ";".join(f"{k}={v}" for k, v in self.meta.items()),
            })
        print(f"[cost] {self.stage}: {wall:.1f}s")
        return False
