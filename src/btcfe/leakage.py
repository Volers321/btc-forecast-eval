"""Three ways a Bitcoin forecast looks good when it is not — each next to the honest version.

All three appeared in the course version of this project (and in many online tutorials):

1. **Same-day features + shuffled split.** Predict whether today's close is below
   yesterday's from today's open/high/low/close, with `train_test_split(shuffle=True)`.
   Today's close is in the features, so the model "predicts" what it already sees.
   Honest version: only information available yesterday, chronological split.
2. **R² on price levels.** Regressing tomorrow's price on today's gives R² ≈ 0.99,
   because prices are persistent — the naive "tomorrow = today" scores the same.
   Honest version: compare with the naive forecast on the same days.
3. **One long-horizon forecast scored as if it were a model.** ARIMA(5,1,0) on prices,
   fitted on 80% and forecasting the remaining 20% in one go: the forecast is almost a
   flat line, i.e. the last training price. Its RMSE equals that of "last price
   forever"; it says nothing about the model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def same_day_features(df: pd.DataFrame, seed: int = 42) -> dict:
    d = df.copy()
    d["median"] = d[["open", "high", "close"]].median(axis=1)
    d["fell"] = (d["close"].shift(1) > d["close"]).astype(int)  # the course target: 1 = fell
    d = d.iloc[1:]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    leaky = ["open", "high", "low", "close", "median"]
    Xtr, Xte, ytr, yte = train_test_split(
        d[leaky], d["fell"], test_size=0.2, random_state=seed, stratify=d["fell"]
    )
    acc_leaky = accuracy_score(yte, clf.fit(Xtr, ytr).predict(Xte))

    # Honest: yesterday's returns and range only, chronological 80/20 split.
    h = pd.DataFrame(
        {
            "ret_1": d["ret"].shift(1),
            "ret_2": d["ret"].shift(2),
            "range_1": ((d["high"] - d["low"]) / d["close"]).shift(1),
        }
    )
    h["fell"] = d["fell"]
    h = h.dropna()
    cut = int(len(h) * 0.8)
    tr, te = h.iloc[:cut], h.iloc[cut:]
    cols = ["ret_1", "ret_2", "range_1"]
    acc_honest = accuracy_score(te["fell"], clf.fit(tr[cols], tr["fell"]).predict(te[cols]))
    majority = float(max(te["fell"].mean(), 1 - te["fell"].mean()))
    return {
        "leaky_accuracy": float(acc_leaky),
        "honest_accuracy": float(acc_honest),
        "majority_baseline": majority,
    }


def price_level_r2(df: pd.DataFrame, seed: int = 42) -> dict:
    d = pd.DataFrame({"x": df["close"].shift(1), "y": df["close"]}).dropna()
    Xtr, Xte, ytr, yte = train_test_split(d[["x"]], d["y"], test_size=0.2, random_state=seed)
    pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
    return {
        "model_r2": float(r2_score(yte, pred)),
        "naive_r2": float(r2_score(yte, Xte["x"])),
        "model_mae": float(mean_absolute_error(yte, pred)),
        "naive_mae": float(mean_absolute_error(yte, Xte["x"])),
    }


def one_shot_arima(df: pd.DataFrame) -> dict:
    from statsmodels.tsa.arima.model import ARIMA

    y = df["close"].to_numpy()
    cut = int(len(y) * 0.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = ARIMA(y[:cut], order=(5, 1, 0)).fit().forecast(len(y) - cut)
    flat = np.full(len(y) - cut, y[cut - 1])
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))  # noqa: E731
    return {
        "arima_rmse": rmse(y[cut:], fc),
        "last_price_rmse": rmse(y[cut:], flat),
        "forecast_range": [float(fc.min()), float(fc.max())],
        "test_days": len(y) - cut,
    }


def run(df: pd.DataFrame) -> dict:
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    return {
        "same_day_features": same_day_features(df),
        "price_level_r2": price_level_r2(df),
        "one_shot_arima": one_shot_arima(df),
    }
