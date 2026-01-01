from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_daily(
    *,
    latitude: float,
    longitude: float,
    timezone: str,
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": timezone,
    }

    r = requests.get(BASE_URL, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()

    daily = payload.get("daily")
    if not isinstance(daily, dict):
        raise ValueError("Open-Meteo response has no 'daily' field")

    time = daily.get("time")
    tmax = daily.get("temperature_2m_max")
    tmin = daily.get("temperature_2m_min")
    if time is None or tmax is None or tmin is None:
        raise ValueError("Open-Meteo response missing required daily keys")

    if not (len(time) == len(tmax) == len(tmin)):
        raise ValueError("Open-Meteo daily arrays have different lengths")

    df = pd.DataFrame({"date": time, "tmax": tmax, "tmin": tmin})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tmax"] = pd.to_numeric(df["tmax"], errors="coerce")
    df["tmin"] = pd.to_numeric(df["tmin"], errors="coerce")
    df = df.dropna(subset=["date", "tmax", "tmin"]).copy()

    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df[df["tmax"] >= df["tmin"]]

    if df.empty:
        raise ValueError("No valid rows after cleaning Open-Meteo data")

    return df.reset_index(drop=True)
