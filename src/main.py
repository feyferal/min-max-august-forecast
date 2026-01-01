from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import SETTINGS
from src.fetch_openmeteo import fetch_daily
from src.features import august_yearly_table, load_daily_csv
from src.model import forecast_august_2026_extremes


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/raw/madrid_daily.csv")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--window-years", type=int, default=15)
    p.add_argument("--log-level", type=str, default="INFO")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("min-max-august-forecast")

    data_path = (project_root() / args.data).resolve()
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.refresh or not data_path.exists():
        log.info("Fetching daily data from Open-Meteo...")
        df_daily = fetch_daily(
            latitude=SETTINGS.latitude,
            longitude=SETTINGS.longitude,
            timezone=SETTINGS.timezone,
            start_date=SETTINGS.start_date,
        )
        df_daily.to_csv(data_path, index=False)
        log.info("Saved daily data to %s", data_path)

    daily = load_daily_csv(str(data_path))
    yearly = august_yearly_table(daily, min_days_in_aug=SETTINGS.min_days_in_aug)

    pred = forecast_august_2026_extremes(
        yearly=yearly,
        year=2026,
        window_years=args.window_years,
        alpha=SETTINGS.alpha,
        degree=SETTINGS.degree,
        min_train_years=SETTINGS.min_train_years,
        eval_last_years=SETTINGS.q_eval_last_years,
    )

    last5 = yearly.tail(5)[["year", "max_tmax", "min_tmin"]].copy()
    pred_row = pd.DataFrame(
        [{"year": 2026, "max_tmax": pred.max_tmax, "min_tmin": pred.min_tmin}]
    )
    out_df = pd.concat([last5, pred_row], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    table_txt = out_df.to_string(index=False, float_format=lambda x: f"{x:.2f}")
    print(table_txt)

    if args.out:
        out_path = (project_root() / args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_txt + "\n", encoding="utf-8")
        log.info("Saved table to %s", out_path)


if __name__ == "__main__":
    main()
