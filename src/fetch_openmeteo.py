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
    daily = payload["daily"]

    df = pd.DataFrame(
        {
            "date": daily["time"],
            "tmax": daily["temperature_2m_max"],
            "tmin": daily["temperature_2m_min"],
        }
    )
    return df
