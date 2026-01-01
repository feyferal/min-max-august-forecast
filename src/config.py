from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    latitude: float = 40.4168
    longitude: float = -3.7038
    timezone: str = "Europe/Madrid"
    start_date: str = "1980-01-01"

    min_days_in_aug: int = 25

    alpha: float = 1.0
    degree: int = 2
    min_train_years: int = 15
    q_eval_last_years: int = 15


SETTINGS = Settings()
