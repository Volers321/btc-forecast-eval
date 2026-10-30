"""Walk-forward evaluation: for every test day, models see only the past.

Loop: for each day t in the test period, the model (refitted every `refit` days on all
returns before t) predicts r[t]. Nothing from day t or later is ever used — no random
splits, no scaler fitted on the whole series, no features computed from the future.

Metrics compare each model with the naive random walk on the same days:
* MAE of the price forecast and its ratio to the naive MAE (< 1 = better than naive)
* Diebold–Mariano test on squared return errors vs naive (two-sided p-value)
* direction hit rate vs "always up" (the share of up days in the test period)
* a long/flat rule (hold BTC when the forecast is positive), before costs, vs buy & hold
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd


def walk_forward(
    returns: np.ndarray,
    model,
    test_start: int,
    refit: int = 21,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Predictions for returns[test_start:], each made from returns[:t] only."""
    preds = np.empty(len(returns) - test_start)
    for k, t in enumerate(range(test_start, len(returns))):
        history = returns[:t]
        if k % refit == 0:
            model.fit(history)
        preds[k] = model.predict(history)
        if progress and k % 100 == 0:
            progress(f"  {model.name}: {k}/{len(preds)}")
    return preds


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray, lag: int = 5) -> tuple[float, float]:
    """DM statistic on squared-error differences (Newey–West variance). Negative = the
    model's errors are smaller. Returns (statistic, two-sided normal p-value)."""
    d = e_model**2 - e_base**2
    n = len(d)
    dc = d - d.mean()
    var = dc @ dc / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * (dc[k:] @ dc[:-k]) / n
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / math.sqrt(var / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def block_bootstrap_ratio(
    a: np.ndarray, b: np.ndarray, n: int = 2000, block: int = 10, seed: int = 0
) -> tuple[float, float]:
    """95% interval for mean(a) / mean(b) with a moving-block bootstrap (errors of
    consecutive days are not independent)."""
    rng = np.random.default_rng(seed)
    m = len(a)
    starts_n = max(1, m - block + 1)
    ratios = []
    for _ in range(n):
        starts = rng.integers(0, starts_n, size=math.ceil(m / block))
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:m]
        ratios.append(a[idx].mean() / b[idx].mean())
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(lo), float(hi)


def score(df_test: pd.DataFrame, pred_ret: np.ndarray, naive_err: np.ndarray | None) -> dict:
    """Metrics for one model. df_test holds the test days with `close`, `ret` and
    `prev_close` (the last close the forecast could use)."""
    r = df_test["ret"].to_numpy()
    price_pred = df_test["prev_close"].to_numpy() * np.exp(pred_ret)
    abs_err = np.abs(df_test["close"].to_numpy() - price_pred)
    out = {
        "mae_usd": float(abs_err.mean()),
        "mape_pct": float((abs_err / df_test["close"].to_numpy()).mean() * 100),
        "rmse_ret": float(np.sqrt(np.mean((r - pred_ret) ** 2))),
    }
    nonzero = pred_ret != 0
    if nonzero.any():
        hits = np.sign(pred_ret[nonzero]) == np.sign(r[nonzero])
        out["direction_hit"] = float(hits.mean())
        position = (pred_ret > 0).astype(float)  # long when the forecast is positive, else flat
        out["long_flat_return_pct"] = float((np.exp((position * r).sum()) - 1) * 100)
    if naive_err is not None:
        out["mae_vs_naive"] = out["mae_usd"] / float(naive_err.mean())
        out["mae_vs_naive_ci95"] = block_bootstrap_ratio(abs_err, naive_err)
        stat, p = diebold_mariano(r - pred_ret, r)  # naive return error = r - 0
        out["dm_stat"], out["dm_p"] = stat, p
    return out, abs_err


def run(
    df: pd.DataFrame,
    model_names: list[str],
    test_from: str,
    refit: int = 21,
    progress: Callable[[str], None] | None = print,
) -> dict:
    from .models import MODELS

    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    test_start = int((df["date"] < pd.Timestamp(test_from)).sum())
    if test_start < 300:
        raise ValueError("need at least 300 training days before the test period")
    returns = df["ret"].to_numpy()
    df_test = df.iloc[test_start:]
    r_test = df_test["ret"].to_numpy()

    results, preds = {}, {}
    naive_pred = np.zeros(len(df_test))
    _, naive_err = score(df_test, naive_pred, None)
    names = ["naive"] + [m for m in model_names if m != "naive"]
    for name in names:
        model = MODELS[name]()
        p = (
            naive_pred
            if name == "naive"
            else walk_forward(returns, model, test_start, refit, progress)
        )
        preds[name] = p
        results[name], _ = score(df_test, p, None if name == "naive" else naive_err)
    up_share = float((r_test > 0).mean())
    return {
        "train": {
            "from": str(df["date"].iloc[0].date()),
            "to": str(df["date"].iloc[test_start - 1].date()),
        },
        "test": {
            "from": str(df_test["date"].iloc[0].date()),
            "to": str(df_test["date"].iloc[-1].date()),
            "days": len(df_test),
        },
        "refit_every_days": refit,
        "always_up_hit": up_share,
        "buy_and_hold_return_pct": float((np.exp(r_test.sum()) - 1) * 100),
        "results": results,
        "_preds": preds,
        "_dates": df_test["date"].dt.strftime("%Y-%m-%d").tolist(),
    }
