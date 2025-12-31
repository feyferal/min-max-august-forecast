from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


@dataclass(frozen=True)
class BacktestRow:
    year: int
    y_true: float
    y_pred: float
    abs_error: float


def fit_predict_model(
    train_years: np.ndarray,
    train_y: np.ndarray,
    pred_years: np.ndarray,
    *,
    alpha: float,
    degree: int,
) -> np.ndarray:
    if degree <= 1:
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(train_years.reshape(-1, 1), train_y)
        return model.predict(pred_years.reshape(-1, 1))

    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        Ridge(alpha=alpha, random_state=0),
    )
    model.fit(train_years.reshape(-1, 1), train_y)
    return model.predict(pred_years.reshape(-1, 1))


def rolling_backtest_yearly(
    yearly: pd.DataFrame,
    *,
    target_col: str,
    alpha: float = 1.0,
    degree: int = 2,
    min_train_years: int = 10,
    window_years: int = 0,
) -> tuple[pd.DataFrame, float]:
    years = yearly["year"].to_numpy(dtype=int)
    y = yearly[target_col].to_numpy(dtype=float)

    rows: list[BacktestRow] = []
    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        train_y = y[:i]

        if window_years and len(train_years) > window_years:
            train_years = train_years[-window_years:]
            train_y = train_y[-window_years:]

        test_year = years[i : i + 1]
        y_true = float(y[i])

        y_pred = float(
            fit_predict_model(train_years, train_y, test_year, alpha=alpha, degree=degree)[0]
        )
        rows.append(
            BacktestRow(
                year=int(test_year[0]),
                y_true=y_true,
                y_pred=y_pred,
                abs_error=abs(y_true - y_pred),
            )
        )

    bt = pd.DataFrame([r.__dict__ for r in rows])
    mae = float(bt["abs_error"].mean()) if len(bt) else float("nan")
    return bt, mae


def predict_mean_2026(
    yearly: pd.DataFrame,
    *,
    target_col: str,
    alpha: float = 1.0,
    degree: int = 2,
    window_years: int = 0,
) -> float:
    years = yearly["year"].to_numpy(dtype=int)
    y = yearly[target_col].to_numpy(dtype=float)

    if window_years and len(years) > window_years:
        years = years[-window_years:]
        y = y[-window_years:]

    pred = fit_predict_model(years, y, np.array([2026], dtype=int), alpha=alpha, degree=degree)[0]
    return float(pred)
