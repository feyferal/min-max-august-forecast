from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import SETTINGS
from src.fetch_openmeteo import fetch_daily
from src.features import august_yearly_table, load_daily_csv, q75
from src.model import predict_mean_2026, rolling_backtest_yearly


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/raw/madrid_daily.csv")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--window-years", type=int, default=20)
    p.add_argument("--min-train-years", type=int, default=10)
    p.add_argument("--min-days-in-aug", type=int, default=25)
    args = p.parse_args()

    data_path = Path(args.data)
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if args.refresh or not data_path.exists():
        df_daily = fetch_daily(
            latitude=SETTINGS.latitude,
            longitude=SETTINGS.longitude,
            timezone=SETTINGS.timezone,
            start_date=SETTINGS.start_date,
        )
        df_daily.to_csv(data_path, index=False)

    daily = load_daily_csv(str(data_path))
    yearly = august_yearly_table(daily, min_days_in_aug=args.min_days_in_aug)

    bt_max, mae_max = rolling_backtest_yearly(
        yearly,
        target_col="mean_tmax",
        alpha=args.alpha,
        degree=args.degree,
        min_train_years=args.min_train_years,
        window_years=args.window_years,
    )
    bt_min, mae_min = rolling_backtest_yearly(
        yearly,
        target_col="mean_tmin",
        alpha=args.alpha,
        degree=args.degree,
        min_train_years=args.min_train_years,
        window_years=args.window_years,
    )

    mean_tmax_2026 = predict_mean_2026(
        yearly,
        target_col="mean_tmax",
        alpha=args.alpha,
        degree=args.degree,
        window_years=args.window_years,
    )
    mean_tmin_2026 = predict_mean_2026(
        yearly,
        target_col="mean_tmin",
        alpha=args.alpha,
        degree=args.degree,
        window_years=args.window_years,
    )

    dmax_q75 = q75(yearly["delta_max"])
    dmin_q75 = q75(yearly["delta_min"])

    max_2026 = mean_tmax_2026 + dmax_q75
    min_2026 = mean_tmin_2026 - dmin_q75

    last5 = yearly.tail(5)[["year", "mean_tmax", "max_tmax", "mean_tmin", "min_tmin"]].copy()
    pred_row = pd.DataFrame(
        [
            {
                "year": 2026,
                "mean_tmax": mean_tmax_2026,
                "max_tmax": max_2026,
                "mean_tmin": mean_tmin_2026,
                "min_tmin": min_2026,
            }
        ]
    )
    out = pd.concat([last5, pred_row], ignore_index=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\nLast 5 Augusts + 2026 prediction (Madrid):")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nModel settings:")
    print(f"alpha={args.alpha}, degree={args.degree}, window_years={args.window_years}")

    print(f"\nBacktest MAE (mean_tmax): {mae_max:.2f} °C")
    print(f"Backtest MAE (mean_tmin): {mae_min:.2f} °C")

    print(f"\nq75(delta_max): {dmax_q75:.2f} °C")
    print(f"q75(delta_min): {dmin_q75:.2f} °C")

    if len(bt_max):
        print("\nBacktest (mean_tmax) last 8 rows:")
        print(bt_max.tail(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    if len(bt_min):
        print("\nBacktest (mean_tmin) last 8 rows:")
        print(bt_min.tail(8).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
