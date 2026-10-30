"""One-step-ahead forecasters of the next daily log return.

Every model predicts r[t] = log(close[t] / close[t-1]) using only data up to t-1. A
forecast of the price follows as close[t-1] * exp(r_hat). Models are refitted every
`refit` steps by the walk-forward loop in :mod:`evaluate`; between refits they reuse
their parameters but always see the newest history.

The naive model (r_hat = 0, "tomorrow = today") is the random-walk benchmark every other
model has to beat. Optional models (LSTM, Prophet) need extra installs.
"""

from __future__ import annotations

import numpy as np

N_LAGS = 5


def features(returns: np.ndarray) -> np.ndarray:
    """Feature row for predicting the step after `returns[-1]`: last N_LAGS returns,
    5- and 20-day mean return, 20-day volatility. Needs at least 20 returns."""
    r = returns
    lags = r[-N_LAGS:][::-1]
    return np.concatenate([lags, [r[-5:].mean(), r[-20:].mean(), r[-20:].std()]])


def design(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) pairs from a history: features of r[:i] predict r[i]."""
    X = [features(returns[:i]) for i in range(20, len(returns))]
    return np.array(X), returns[20:]


class Naive:
    """Random walk: tomorrow's price = today's price (predicted return 0)."""

    name = "naive"

    def fit(self, returns: np.ndarray) -> None:
        pass

    def predict(self, returns: np.ndarray) -> float:
        return 0.0


class Drift:
    """Random walk with drift: the average historical return."""

    name = "drift"

    def fit(self, returns: np.ndarray) -> None:
        self.mu = float(np.mean(returns))

    def predict(self, returns: np.ndarray) -> float:
        return self.mu


class Arima:
    """ARIMA(p, 0, q) with a constant on returns (= ARIMA(p, 1, q) on log prices).
    Fitted on the last `window` returns at each refit; between refits the fitted
    parameters filter the newest history."""

    def __init__(self, order=(1, 0, 1), window: int = 1000):
        self.order, self.window = order, window
        self.name = f"arima{order[0]}{order[1]}{order[2]}"

    def fit(self, returns: np.ndarray) -> None:
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.params = ARIMA(returns[-self.window :], order=self.order, trend="c").fit().params

    def predict(self, returns: np.ndarray) -> float:
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ARIMA(returns[-self.window :], order=self.order, trend="c").filter(self.params)
        return float(res.forecast(1)[0])


class _Sklearn:
    def __init__(self, estimator, name: str):
        self.estimator, self.name = estimator, name

    def fit(self, returns: np.ndarray) -> None:
        X, y = design(returns)
        self.estimator.fit(X, y)

    def predict(self, returns: np.ndarray) -> float:
        return float(self.estimator.predict(features(returns)[None, :])[0])


def Ridge() -> _Sklearn:
    from sklearn.linear_model import Ridge as R
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # The scaler is part of the pipeline, so it is fitted on training data only.
    return _Sklearn(make_pipeline(StandardScaler(), R(alpha=10.0)), "ridge")


def Boosting() -> _Sklearn:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return _Sklearn(
        HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3, random_state=0
        ),
        "boosting",
    )


class Lstm:
    """Small LSTM over the last `seq` returns (optional: pip install -e ".[lstm]").
    Returns are scaled by the training standard deviation only — the course version
    fitted the scaler on the whole series, test period included."""

    name = "lstm"

    def __init__(self, seq: int = 30, epochs: int = 15, seed: int = 0):
        self.seq, self.epochs, self.seed = seq, epochs, seed

    def fit(self, returns: np.ndarray) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        self.scale = float(np.std(returns)) or 1.0
        r = returns / self.scale
        X = np.array([r[i - self.seq : i] for i in range(self.seq, len(r))])[..., None]
        y = r[self.seq :]
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(self.seq, 1)),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse")
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=64,
            verbose=0,
            validation_split=0.1,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        )

    def predict(self, returns: np.ndarray) -> float:
        x = (returns[-self.seq :] / self.scale)[None, :, None]
        return float(self.model.predict(x, verbose=0)[0, 0]) * self.scale


MODELS = {
    "naive": Naive,
    "drift": Drift,
    "arima": Arima,
    "ridge": Ridge,
    "boosting": Boosting,
    "lstm": Lstm,
}
