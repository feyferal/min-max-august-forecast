from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    latitude: float = 40.4168
    longitude: float = -3.7038
    timezone: str = "Europe/Madrid"
    start_date: str = "1980-01-01"


SETTINGS = Settings()
