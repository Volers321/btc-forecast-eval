import json

import numpy as np
import pandas as pd
import pytest

from btcfe import cli, data, evaluate, leakage, s2f
from btcfe.models import MODELS, Arima, Naive, design, features


def random_walk(n=1500, seed=1, drift=0.0, start="2014-01-01"):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.03, n)
    close = 1000 * np.exp(np.cumsum(r))
    noise = np.abs(rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "date": pd.bdate_range(start, periods=n),
            "open": close * np.exp(-r / 2),
            "high": close * (1 + noise),
            "low": close * (1 - noise),
            "close": close,
        }
    )


def write_stooq(df, path):
    df.rename(columns={v: k for k, v in data.COLUMNS.items()}).to_csv(path, index=False)


def test_load_renames_validates_and_sorts(tmp_path):
    df = random_walk(50)
    write_stooq(df.iloc[::-1], tmp_path / "btc.csv")  # reversed order on disk
    out = data.load(tmp_path / "btc.csv")
    assert list(out.columns[:5]) == ["date", "open", "high", "low", "close"]
    assert out["date"].is_monotonic_increasing
    assert np.isclose(out["ret"].iloc[1], np.log(out["close"].iloc[1] / out["close"].iloc[0]))
    bad = df.copy()
    bad.loc[3, "close"] = 0
    write_stooq(bad, tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="non-positive"):
        data.load(tmp_path / "bad.csv")


def test_features_use_only_the_past():
    r = np.arange(30, dtype=float)
    f = features(r)
    assert f[0] == 29 and f[4] == 25, "lag 1 is the most recent return"
    X, y = design(r)
    assert y[0] == r[20] and X[0][0] == r[19], "row i predicts r[i] from r[:i]"


def test_walk_forward_never_sees_the_future():
    seen = []

    class Spy(Naive):
        def fit(self, returns):
            seen.append(("fit", len(returns)))

        def predict(self, returns):
            seen.append(("predict", len(returns)))
            return 0.0

    r = np.zeros(100)
    evaluate.walk_forward(r, Spy(), test_start=60, refit=10)
    predicts = [n for kind, n in seen if kind == "predict"]
    assert predicts == list(range(60, 100)), "prediction for day t sees exactly r[:t]"
    assert [n for kind, n in seen if kind == "fit"] == [60, 70, 80, 90]


def test_no_model_beats_naive_on_a_random_walk():
    df = data.since(random_walk(1300).assign(ret=lambda d: np.log(d.close).diff()), "2014-01-01")
    out = evaluate.run(
        df, ["drift", "ridge"], test_from=str(df.date.iloc[1100].date()), progress=None
    )
    for m in ("drift", "ridge"):
        r = out["results"][m]
        assert r["mae_vs_naive"] > 0.98, "a random walk is unpredictable"
        assert r["dm_p"] > 0.01
        lo, hi = r["mae_vs_naive_ci95"]
        assert lo <= r["mae_vs_naive"] <= hi


def test_diebold_mariano_detects_a_clearly_better_model():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 500)
    better = base * 0.5
    stat, p = evaluate.diebold_mariano(better, base)
    assert stat < 0 and p < 0.001
    assert evaluate.diebold_mariano(base, base) == (0.0, 1.0)


def test_arima_runs_and_registry_is_complete():
    r = random_walk(400).pipe(lambda d: np.log(d.close).diff().dropna().to_numpy())
    m = Arima(window=300)
    m.fit(r)
    assert np.isfinite(m.predict(r))
    assert {"naive", "drift", "arima", "ridge", "boosting", "lstm"} <= set(MODELS)


def test_leakage_demo_shows_the_gap():
    df = random_walk(1500).assign(ret=lambda d: np.log(d.close).diff())
    out = leakage.run(df)
    s = out["same_day_features"]
    assert s["leaky_accuracy"] > 0.65, "today's close in the features gives the answer away"
    assert abs(s["honest_accuracy"] - 0.5) < 0.07
    p = out["price_level_r2"]
    assert p["model_r2"] > 0.95 and abs(p["model_r2"] - p["naive_r2"]) < 0.01
    a = out["one_shot_arima"]
    assert a["forecast_range"][1] - a["forecast_range"][0] < 0.2 * a["last_price_rmse"]


def test_s2f_supply_schedule():
    for date, expected in zip(s2f.HALVINGS, s2f.SUPPLY_AT_HALVING, strict=True):
        assert s2f.supply(date) == pytest.approx(expected)
    assert s2f.reward(pd.Timestamp("2025-01-01")) == 3.125
    assert s2f.stock_to_flow(pd.Timestamp("2024-04-20")) == pytest.approx(
        19_687_500 / (144 * 365 * 3.125)
    )
    assert s2f.supply(pd.Timestamp("2025-01-22")) == pytest.approx(19.81e6, rel=0.005)


def test_cli_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    df = random_walk(1400, start="2015-01-01")
    write_stooq(df, tmp_path / "data/btcusd_d.csv")
    cli.main(["eval", "--models", "drift,ridge", "--test-from", str(df.date.iloc[1200].date())])
    out = capsys.readouterr().out
    assert "naive" in out and "nan" not in out
    report = json.loads((tmp_path / "results/eval.json").read_text())
    assert set(report["results"]) == {"naive", "drift", "ridge"}
    cli.main(["leakage"])
    cli.main(["s2f", "--fit-until", "2019-12-31"])
    for png in ("eval.png", "leakage.png", "s2f.png"):
        assert (tmp_path / "docs" / png).stat().st_size > 10_000
