from __future__ import annotations

import numpy as np
import pandas as pd


def load_daily_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
    df = df.dropna(subset=["date", "tmax", "tmin"])
    return df


def august_yearly_table(daily: pd.DataFrame, *, min_days_in_aug: int = 25) -> pd.DataFrame:
    df = daily.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df[df["month"] == 8]

    g = df.groupby("year", as_index=False).agg(
        days=("date", "count"),
        mean_tmax=("tmax", "mean"),
        max_tmax=("tmax", "max"),
        mean_tmin=("tmin", "mean"),
        min_tmin=("tmin", "min"),
    )

    g = g[g["days"] >= min_days_in_aug].copy()
    g["delta_max"] = g["max_tmax"] - g["mean_tmax"]
    g["delta_min"] = g["mean_tmin"] - g["min_tmin"]

    cols = ["year", "mean_tmax", "max_tmax", "mean_tmin", "min_tmin", "delta_max", "delta_min"]
    g = g[cols].sort_values("year").reset_index(drop=True)
    return g


def quantile(series: pd.Series, q: float) -> float:
    x = series.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))
