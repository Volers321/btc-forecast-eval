"""Stock-to-Flow (S2F), done with the real issuance schedule and tested out of sample.

S2F = coins in circulation / coins issued per year. The model regresses log price on
log S2F. Popularised in 2019, it is widely criticised: the S2F series is a deterministic
step function of time, so the fit is essentially "price grows over time" with steps at
halvings, and S2F-like regressions are prone to spurious correlation between trending
series. Here the model is fitted on monthly data up to 2019 and then scored on 2020–2025,
which it never saw.

Supply: exactly 10.5M, 15.75M, 18.375M and 19.6875M BTC had been issued at the four
halvings (blocks 210,000 × k); between them supply is interpolated linearly in time,
which is accurate to within a few percent because block time averages ~10 minutes.
The course version assumed a halving every 4 calendar years from 2009 and used US-dollar
index values in place of BTC prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENESIS = pd.Timestamp("2009-01-03")
HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
]
SUPPLY_AT_HALVING = [10_500_000.0, 15_750_000.0, 18_375_000.0, 19_687_500.0]
BLOCKS_PER_DAY = 144.0


def reward(date: pd.Timestamp) -> float:
    return 50.0 / 2 ** sum(date >= h for h in HALVINGS)


def supply(date: pd.Timestamp) -> float:
    knots = [GENESIS, *HALVINGS]
    values = [0.0, *SUPPLY_AT_HALVING]
    if date >= knots[-1]:
        days = (date - knots[-1]).days
        return values[-1] + days * BLOCKS_PER_DAY * reward(date)
    for (t0, v0), (t1, v1) in zip(
        zip(knots, values, strict=True), zip(knots[1:], values[1:], strict=True), strict=False
    ):
        if t0 <= date < t1:
            return v0 + (v1 - v0) * (date - t0).days / (t1 - t0).days
    return 0.0


def stock_to_flow(date: pd.Timestamp) -> float:
    flow = BLOCKS_PER_DAY * 365 * reward(date)
    return supply(date) / flow


def run(df: pd.DataFrame, fit_until: str = "2019-12-31") -> dict:
    m = df.set_index("date")["close"].resample("ME").last().dropna().to_frame("price")
    m["s2f"] = [stock_to_flow(d) for d in m.index]
    m = m[m["s2f"] > 0]
    x, y = np.log(m["s2f"].to_numpy()), np.log(m["price"].to_numpy())
    train = m.index <= pd.Timestamp(fit_until)
    b, a = np.polyfit(x[train], y[train], 1)
    pred = a + b * x
    resid = y - pred

    def r2(mask):
        return float(1 - np.sum(resid[mask] ** 2) / np.sum((y[mask] - y[mask].mean()) ** 2))

    test = ~train
    ratio = np.exp(resid[test])  # actual / model price
    m["model_price"] = np.exp(pred)
    return {
        "fit_until": fit_until,
        "slope": float(b),
        "intercept": float(a),
        "r2_in_sample": r2(train),
        "months_out_of_sample": int(test.sum()),
        "out_of_sample_median_actual_over_model": float(np.median(ratio)),
        "out_of_sample_range_actual_over_model": [float(ratio.min()), float(ratio.max())],
        "model_price_last_month": float(np.exp(pred[-1])),
        "actual_price_last_month": float(np.exp(y[-1])),
        "_monthly": m,
    }
