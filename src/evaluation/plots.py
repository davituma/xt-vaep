"""
Figuras do TCC. Todas salvas em outputs/figures/ em 300 dpi.
Legenda vai ACIMA da figura no documento (padrão do template), então os
títulos aqui são curtos e a descrição completa fica no texto.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch
from socceraction.spadl import config as spadlcfg

from src.config import path

FIG = lambda name: path("outputs", "figures", name)  # noqa: E731
L, W = spadlcfg.field_length, spadlcfg.field_width


def xt_heatmap(xt: np.ndarray, fname="fig02_xt_heatmap.png", annotate=True):
    pitch = Pitch(pitch_type="custom", pitch_length=L, pitch_width=W,
                  line_color="black", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11, 7))
    # xt tem linha 0 = topo (y invertido); imshow com origin='upper' desenha certo
    im = ax.imshow(xt, extent=[0, L, 0, W], origin="upper", cmap="Reds", alpha=.85, zorder=1)
    if annotate:
        w, l = xt.shape
        for i in range(w):
            for j in range(l):
                ax.text((j + .5) * L / l, W - (i + .5) * W / w, f"{xt[i, j]:.3f}",
                        ha="center", va="center", fontsize=6.5, zorder=3)
    plt.colorbar(im, ax=ax, fraction=.03, pad=.02, label="xT")
    ax.set_title("Expected Threat por zona (ataque →)", fontsize=12)
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def nz_heatmap(counts: np.ndarray, fname="fig03_nz_counts.png"):
    pitch = Pitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="black", line_zorder=2)
    fig, ax = pitch.draw(figsize=(11, 7))
    im = ax.imshow(np.log10(counts + 1), extent=[0, L, 0, W], origin="upper",
                   cmap="Blues", alpha=.85, zorder=1)
    plt.colorbar(im, ax=ax, fraction=.03, pad=.02, label="log10(N(z)+1)")
    ax.set_title("Contagem de observações por célula, N(z)", fontsize=12)
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def scatter_models(df: pd.DataFrame, xcol="xt_p90", ycol="vaep_p90",
                   fname="fig05_scatter_xt_vaep.png", label_top=8):
    fig, ax = plt.subplots(figsize=(8, 7))
    groups = df.pos_group.fillna("Unknown")
    for g, sub in df.groupby(groups):
        ax.scatter(sub[xcol], sub[ycol], s=22, alpha=.7, label=g)
    top = df.nlargest(label_top, ycol)
    for pid, r in top.iterrows():
        ax.annotate(str(r.player_name)[:18], (r[xcol], r[ycol]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("xT por 90 min"); ax.set_ylabel("VAEP por 90 min")
    ax.axhline(0, c="gray", lw=.5); ax.axvline(0, c="gray", lw=.5)
    ax.legend(title="Posição", fontsize=8); ax.grid(alpha=.3)
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def calibration(y_true, y_prob, fname, n_bins=10, title=""):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(6, 6))
    frac, mean = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot(mean, frac, "o-", label="modelo")
    ax.plot([0, max(mean.max(), frac.max())], [0, max(mean.max(), frac.max())], "--", c="gray", label="ideal")
    ax.set_xlabel("probabilidade prevista"); ax.set_ylabel("frequência observada")
    ax.set_title(title); ax.legend(); ax.grid(alpha=.3)
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def stability_curves(curves: pd.DataFrame, metric: str, fname: str, ylabel: str):
    """
    curves: colunas [model, n_games, rep, <metric>]. Plota média ± IC (via quantis
    das repetições) por modelo em função de n_games. Eixo x em log2.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model, sub in curves.groupby("model"):
        g = sub.groupby("n_games")[metric]
        mean, lo, hi = g.mean(), g.quantile(.1), g.quantile(.9)
        ax.plot(mean.index, mean.values, "o-", label=model.upper())
        ax.fill_between(mean.index, lo.values, hi.values, alpha=.2)
    ax.set_xscale("log", base=2); ax.set_xlabel("partidas de ajuste")
    ax.set_ylabel(ylabel); ax.grid(alpha=.3, which="both"); ax.legend()
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def diagnostic_curve(curves: pd.DataFrame, col: str, fname: str, ylabel: str, model=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = curves if model is None else curves[curves.model == model]
    g = sub.groupby("n_games")[col]
    ax.plot(g.mean().index, g.mean().values, "o-")
    ax.fill_between(g.mean().index, g.quantile(.1).values, g.quantile(.9).values, alpha=.2)
    ax.set_xscale("log", base=2); ax.set_xlabel("partidas de ajuste"); ax.set_ylabel(ylabel)
    ax.grid(alpha=.3, which="both")
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)


def player_actions(actions: pd.DataFrame, values: pd.Series, player_name: str, fname: str,
                   top_n=25):
    """Estudo de caso: as top_n ações (por |valor|) de um jogador sobre o campo."""
    v = values.reindex(actions.index)
    a = actions.assign(v=v).dropna(subset=["v"])
    a = a.reindex(a.v.abs().nlargest(top_n).index)
    pitch = Pitch(pitch_type="custom", pitch_length=L, pitch_width=W, line_color="black")
    fig, ax = pitch.draw(figsize=(11, 7))
    pos, neg = a[a.v >= 0], a[a.v < 0]
    pitch.arrows(pos.start_x, pos.start_y, pos.end_x, pos.end_y, ax=ax, color="green", width=1.5, alpha=.8)
    pitch.arrows(neg.start_x, neg.start_y, neg.end_x, neg.end_y, ax=ax, color="red", width=1.5, alpha=.8)
    ax.set_title(f"{player_name} — {top_n} ações de maior |valor| (verde +, vermelho −)", fontsize=11)
    fig.savefig(FIG(fname), dpi=300, bbox_inches="tight"); plt.close(fig)
