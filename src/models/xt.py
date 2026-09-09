"""
Expected Threat — implementação própria (seção 2.2.4).

Segue Singh (2019). A convenção de indexação da grade (eixo y invertido)
é a mesma da implementação de referência do socceraction, para que os
mapas sejam diretamente comparáveis no teste de validação.

Diferenças em relação à referência:
  - suavização aditiva opcional para células esparsas (A.0 item 4)
  - diagnósticos N(z): distribuição das contagens por célula, essencial
    para as curvas de estabilidade (seção 2.2.7)
  - registro do número de iterações e do delta final
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadlcfg


@dataclass
class XTDiagnostics:
    n_iterations: int = 0
    final_delta: float = float("nan")
    converged: bool = False
    counts_total: np.ndarray = field(default=None, repr=False)   # N(z) = chutes + movimentos
    counts_shot: np.ndarray = field(default=None, repr=False)
    counts_move: np.ndarray = field(default=None, repr=False)

    def summary(self, min_count: int) -> dict:
        """Quantis de N(z) e proporção de células esparsas — vai para a Tabela 5."""
        n = self.counts_total.ravel()
        return {
            "xt_iterations": self.n_iterations,
            "xt_converged": self.converged,
            "nz_min": int(n.min()), "nz_p10": float(np.percentile(n, 10)),
            "nz_median": float(np.median(n)), "nz_p90": float(np.percentile(n, 90)),
            "nz_max": int(n.max()),
            "nz_frac_sparse": float((n < min_count).mean()),
            "nz_zero_cells": int((n == 0).sum()),
        }


class ExpectedThreat:
    def __init__(self, n_cols=12, n_rows=8, max_iterations=200, tolerance=1e-6,
                 smoothing="additive", alpha=1.0,
                 move_types=("pass", "cross", "dribble"), shot_types=("shot",)):
        self.l, self.w = n_cols, n_rows
        self.max_iterations, self.tolerance = max_iterations, tolerance
        self.smoothing, self.alpha = smoothing, alpha
        self.move_ids = [spadlcfg.actiontypes.index(t) for t in move_types]
        self.shot_ids = [spadlcfg.actiontypes.index(t) for t in shot_types]
        self.success_id = spadlcfg.results.index("success")
        self.xT = None
        self.diag = XTDiagnostics()

    # ------------------------------------------------------------------
    def _cells(self, x, y):
        """(x, y) em metros → (coluna, linha). Linha invertida como na referência."""
        xi = np.clip((np.asarray(x) / spadlcfg.field_length * self.l).astype(int), 0, self.l - 1)
        yj = np.clip((np.asarray(y) / spadlcfg.field_width * self.w).astype(int), 0, self.w - 1)
        return xi, (self.w - 1) - yj

    def _count(self, x, y):
        m = np.zeros((self.w, self.l))
        xi, yj = self._cells(x, y)
        np.add.at(m, (yj, xi), 1)
        return m

    def _smooth_ratio(self, num, den, prior):
        """(num + α·prior) / (den + α): suavização aditiva com prior global."""
        if self.smoothing == "additive" and self.alpha > 0:
            return (num + self.alpha * prior) / (den + self.alpha)
        return np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    # ------------------------------------------------------------------
    def fit(self, actions: pd.DataFrame) -> "ExpectedThreat":
        shots = actions[actions.type_id.isin(self.shot_ids)]
        goals = shots[shots.result_id == self.success_id]
        moves = actions[actions.type_id.isin(self.move_ids)]
        ok_moves = moves[moves.result_id == self.success_id]

        c_shot = self._count(shots.start_x, shots.start_y)
        c_goal = self._count(goals.start_x, goals.start_y)
        c_move = self._count(moves.start_x, moves.start_y)
        c_tot = c_shot + c_move

        # priors globais para a suavização
        p_shot_g = c_shot.sum() / max(c_tot.sum(), 1)
        p_goal_g = c_goal.sum() / max(c_shot.sum(), 1)

        self.shot_prob = self._smooth_ratio(c_shot, c_tot, p_shot_g)
        self.move_prob = 1.0 - self.shot_prob
        self.goal_prob = self._smooth_ratio(c_goal, c_shot, p_goal_g)

        # Matriz de transição T(z, z') = P(chegar com sucesso em z' | tentou mover de z).
        # IMPORTANTE: o denominador é o total de movimentos TENTADOS a partir de z
        # (sucessos + falhas), não só os bem-sucedidos. Assim as linhas somam
        # menos de 1 e o valor "perdido" em passes interceptados é modelado
        # implicitamente. É a formulação de Singh (2019) e a da referência.
        n = self.w * self.l
        T = np.zeros((n, n))
        sx, sy = self._cells(ok_moves.start_x, ok_moves.start_y)
        ex, ey = self._cells(ok_moves.end_x, ok_moves.end_y)
        np.add.at(T, (sy * self.l + sx, ey * self.l + ex), 1)
        all_sx, all_sy = self._cells(moves.start_x, moves.start_y)
        attempts = np.zeros(n)
        np.add.at(attempts, all_sy * self.l + all_sx, 1)
        if self.smoothing == "additive" and self.alpha > 0:
            # prior: distribui α tentativas fictícias uniformemente entre destinos,
            # com a taxa de sucesso global — evita linhas nulas em células vazias
            succ_rate = len(ok_moves) / max(len(moves), 1)
            T = T + self.alpha * succ_rate / n
            attempts = attempts + self.alpha
        self.transition = np.divide(T, attempts[:, None], out=np.zeros_like(T), where=attempts[:, None] > 0)

        self.diag = XTDiagnostics(counts_total=c_tot, counts_shot=c_shot, counts_move=c_move)
        self._solve()
        return self

    def _solve(self):
        """Iteração de valor sobre a equação de Bellman (seção 1.5.3)."""
        s, m, g = self.shot_prob.ravel(), self.move_prob.ravel(), self.goal_prob.ravel()
        xT = np.zeros(self.w * self.l)
        delta = float("inf")
        for it in range(1, self.max_iterations + 1):
            new = s * g + m * (self.transition @ xT)
            delta = float(np.abs(new - xT).max())
            xT = new
            if delta < self.tolerance:
                break
        self.diag.n_iterations, self.diag.final_delta = it, delta
        self.diag.converged = delta < self.tolerance
        self.xT = xT.reshape(self.w, self.l)

    # ------------------------------------------------------------------
    def rate(self, actions: pd.DataFrame) -> np.ndarray:
        """ΔxT para movimentos bem-sucedidos; NaN para as demais ações."""
        out = np.full(len(actions), np.nan)
        mask = (actions.type_id.isin(self.move_ids) & (actions.result_id == self.success_id)).to_numpy()
        a = actions[mask]
        sx, sy = self._cells(a.start_x, a.start_y)
        ex, ey = self._cells(a.end_x, a.end_y)
        out[mask] = self.xT[ey, ex] - self.xT[sy, sx]
        return out

    def is_monotonic_towards_goal(self) -> bool:
        """Teste de sanidade: média por coluna deve crescer em direção ao gol."""
        col_means = self.xT.mean(axis=0)
        return bool(np.all(np.diff(col_means) >= -1e-9))


def from_config(cfg: dict) -> ExpectedThreat:
    return ExpectedThreat(
        n_cols=cfg["grid"]["n_cols"], n_rows=cfg["grid"]["n_rows"],
        max_iterations=cfg["solver"]["max_iterations"], tolerance=cfg["solver"]["tolerance"],
        smoothing=cfg["sparse_cells"]["strategy"], alpha=cfg["sparse_cells"]["alpha"],
        move_types=cfg["move_types"], shot_types=cfg["shot_types"],
    )
