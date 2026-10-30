"""Charts for the README (PNG, written to docs/)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, ACCENT, GOOD, BAD = "#1f2933", "#7b8794", "#3b6fd8", "#2f9e5b", "#d64545"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis="x", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def eval_chart(out: dict, path: Path) -> None:
    rows = [(m, r) for m, r in out["results"].items() if m != "naive"]
    rows.sort(key=lambda mr: mr[1]["mae_vs_naive"])
    names = [m for m, _ in rows]
    ratio = np.array([r["mae_vs_naive"] for _, r in rows])
    lo = np.array([r["mae_vs_naive_ci95"][0] for _, r in rows])
    hi = np.array([r["mae_vs_naive_ci95"][1] for _, r in rows])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rows) + 1.6))
    y = np.arange(len(rows))
    ax.errorbar(
        ratio,
        y,
        xerr=[ratio - lo, hi - ratio],
        fmt="o",
        color=ACCENT,
        ecolor=MUTED,
        elinewidth=2,
        capsize=4,
        markersize=7,
    )
    ax.axvline(1.0, color=INK, linewidth=1.2, linestyle="--")
    ax.text(1.0, -0.55, " naive = 1.0", color=INK, va="bottom", fontsize=9)
    ax.set_yticks(y, names)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.set_xlabel("MAE of the price forecast ÷ MAE of “tomorrow = today”   (< 1 is better)")
    t = out["test"]
    ax.set_title(
        f"Next-day BTC forecasts, walk-forward {t['from']} – {t['to']} ({t['days']} days)",
        loc="left",
        fontsize=11,
        color=INK,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def leakage_chart(leak: dict, path: Path) -> None:
    s = leak["same_day_features"]
    labels = [
        "Same-day features,\nshuffled split",
        "Yesterday's data only,\nchronological split",
        "Always predict\nthe majority class",
    ]
    vals = [s["leaky_accuracy"], s["honest_accuracy"], s["majority_baseline"]]
    colors = [BAD, ACCENT, MUTED]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.barh(labels, vals, color=colors, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.1%}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy: did BTC close lower than the day before?")
    ax.set_title(
        "The same question, with and without data leakage", loc="left", fontsize=11, color=INK
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def s2f_chart(s2f_out: dict, path: Path) -> None:
    import pandas as pd

    m = s2f_out["_monthly"]
    cut = pd.Timestamp(s2f_out["fit_until"])
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.semilogy(m.index, m["price"], color=INK, linewidth=1.6, label="BTC price (month end)")
    ax.semilogy(
        m.index[m.index <= cut],
        m["model_price"][m.index <= cut],
        color=ACCENT,
        linewidth=1.6,
        label="S2F model, fitted to 2019",
    )
    ax.semilogy(
        m.index[m.index > cut],
        m["model_price"][m.index > cut],
        color=BAD,
        linewidth=1.6,
        linestyle="--",
        label="S2F model, out of sample",
    )
    ax.axvline(cut, color=MUTED, linewidth=1)
    ax.set_ylabel("USD (log scale)")
    ax.set_title(
        "Stock-to-Flow fits the past, overshoots the future", loc="left", fontsize=11, color=INK
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
