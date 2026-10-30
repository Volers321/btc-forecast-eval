"""Daily BTC/USD prices from stooq.pl.

stooq exports CSV with Polish headers (Data, Otwarcie, Najwyzszy, Najnizszy, Zamkniecie,
optionally Wolumen). The data is not redistributed in this repository; `btcfe download`
fetches it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

STOOQ_URL = "https://stooq.pl/q/d/l/?s=btcusd&i=d"
COLUMNS = {
    "Data": "date",
    "Otwarcie": "open",
    "Najwyzszy": "high",
    "Najnizszy": "low",
    "Zamkniecie": "close",
    "Wolumen": "volume",
}
DEFAULT_PATH = Path("data/btcusd_d.csv")


def download(path: str | Path = DEFAULT_PATH, url: str = STOOQ_URL) -> Path:
    import requests

    resp = requests.get(url, timeout=60, headers={"User-Agent": "btc-forecast-eval/0.1"})
    resp.raise_for_status()
    if not resp.text.startswith("Data,"):
        raise RuntimeError(f"Unexpected response from stooq: {resp.text[:200]!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    return path


def load(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Load and validate prices; adds the daily log return `ret` (ret[t] = log c[t]/c[t-1])."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found — run `btcfe download` first.")
    df = pd.read_csv(path).rename(columns=COLUMNS)
    missing = {"date", "open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)} in {path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in the data")
    df["ret"] = np.log(df["close"]).diff()
    return df


def since(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """Rows from `start` on. The first years (2010–2013) are illiquid and dominated by
    jumps from cents to hundreds of dollars; most analyses start later."""
    return df[df["date"] >= pd.Timestamp(start)].reset_index(drop=True)
