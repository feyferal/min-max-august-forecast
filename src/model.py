from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


DEFAULT_Q_GRID: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


@dataclass(frozen=True)
class BacktestRow:
    year: int
    y_true: float
    y_pred: float
    abs_error: float


@dataclass(frozen=True)
class BacktestExtremeRow:
    year: int
    mean_true: float
    mean_pred: float
    delta_q: float
    extreme_true: float
    extreme_pred: float
    abs_error: float


@dataclass(frozen=True)
class August2026Forecast:
    mean_tmax: float
    mean_tmin: float
    max_tmax: float
    min_tmin: float

    q_max: float
    q_min: float
    delta_max: float
    delta_min: float

    mae_max: float
    mae_min: float


def _make_year_pipeline(*, alpha: float, degree: int) -> Pipeline:
    steps: list[tuple[str, object]] = [("scaler", StandardScaler())]
    if degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def fit_predict_model(
    train_years: np.ndarray,
    train_y: np.ndarray,
    pred_years: np.ndarray,
    *,
    alpha: float,
    degree: int,
) -> np.ndarray:
    model = _make_year_pipeline(alpha=alpha, degree=degree)
    X_train = train_years.reshape(-1, 1).astype(float)
    X_pred = pred_years.reshape(-1, 1).astype(float)
    model.fit(X_train, train_y)
    return model.predict(X_pred)


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


def rolling_backtest_extreme_via_delta(
    yearly: pd.DataFrame,
    *,
    mean_col: str,
    extreme_col: str,
    delta_col: str,
    q: float,
    alpha: float = 1.0,
    degree: int = 2,
    min_train_years: int = 10,
    window_years: int = 0,
    direction: str,
) -> tuple[pd.DataFrame, float]:
    if direction not in {"plus", "minus"}:
        raise ValueError("direction must be 'plus' or 'minus'")
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1), got {q}")

    years_all = yearly["year"].to_numpy(dtype=int)

    rows: list[BacktestExtremeRow] = []
    for i in range(min_train_years, len(years_all)):
        train_slice = yearly.iloc[:i].copy()
        if window_years and len(train_slice) > window_years:
            train_slice = train_slice.iloc[-window_years:].copy()

        train_years = train_slice["year"].to_numpy(dtype=int)
        train_mean = train_slice[mean_col].to_numpy(dtype=float)

        train_delta = train_slice[delta_col].to_numpy(dtype=float)
        train_delta = train_delta[~np.isnan(train_delta)]

        test_year = int(years_all[i])
        mean_true = float(yearly.iloc[i][mean_col])
        extreme_true = float(yearly.iloc[i][extreme_col])

        mean_pred = float(
            fit_predict_model(
                train_years,
                train_mean,
                np.array([test_year], dtype=int),
                alpha=alpha,
                degree=degree,
            )[0]
        )

        if train_delta.size:
            delta_q = float(np.quantile(train_delta, q))
        else:
            delta_q = float("nan")

        extreme_pred = mean_pred + delta_q if direction == "plus" else mean_pred - delta_q

        rows.append(
            BacktestExtremeRow(
                year=test_year,
                mean_true=mean_true,
                mean_pred=mean_pred,
                delta_q=delta_q,
                extreme_true=extreme_true,
                extreme_pred=extreme_pred,
                abs_error=abs(extreme_true - extreme_pred),
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


def _select_best_q_for_extreme_internal(
    yearly: pd.DataFrame,
    *,
    mean_col: str,
    extreme_col: str,
    delta_col: str,
    direction: str,
    alpha: float,
    degree: int,
    min_train_years: int,
    window_years: int,
    eval_last_years: int,
    q_grid: tuple[float, ...] = DEFAULT_Q_GRID,
) -> tuple[float, float]:
    best_q = float(q_grid[0])
    best_mae = float("inf")

    for q in q_grid:
        bt, _ = rolling_backtest_extreme_via_delta(
            yearly,
            mean_col=mean_col,
            extreme_col=extreme_col,
            delta_col=delta_col,
            q=float(q),
            alpha=alpha,
            degree=degree,
            min_train_years=min_train_years,
            window_years=window_years,
            direction=direction,
        )

        if len(bt) == 0:
            mae = float("nan")
        elif eval_last_years > 0:
            mae = float(bt.tail(eval_last_years)["abs_error"].mean())
        else:
            mae = float(bt["abs_error"].mean())

        if np.isfinite(mae) and mae < best_mae:
            best_mae = mae
            best_q = float(q)

    return best_q, best_mae


def _tail_by_window(values: np.ndarray, window_years: int) -> np.ndarray:
    if window_years and values.size > window_years:
        return values[-window_years:]
    return values


def forecast_august_2026_extremes(
    *,
    yearly: pd.DataFrame,
    year: int = 2026,
    window_years: int = 0,
    alpha: float = 1.0,
    degree: int = 2,
    min_train_years: int = 10,
    eval_last_years: int = 5,
) -> August2026Forecast:
    if year != 2026:
        raise ValueError("This helper currently expects year=2026 (kept simple on purpose).")
    if yearly.empty:
        raise ValueError("yearly is empty")

    mean_tmax_2026 = predict_mean_2026(
        yearly,
        target_col="mean_tmax",
        alpha=alpha,
        degree=degree,
        window_years=window_years,
    )
    mean_tmin_2026 = predict_mean_2026(
        yearly,
        target_col="mean_tmin",
        alpha=alpha,
        degree=degree,
        window_years=window_years,
    )

    q_max, mae_max = _select_best_q_for_extreme_internal(
        yearly,
        mean_col="mean_tmax",
        extreme_col="max_tmax",
        delta_col="delta_max",
        direction="plus",
        alpha=alpha,
        degree=degree,
        min_train_years=min_train_years,
        window_years=window_years,
        eval_last_years=eval_last_years,
    )
    q_min, mae_min = _select_best_q_for_extreme_internal(
        yearly,
        mean_col="mean_tmin",
        extreme_col="min_tmin",
        delta_col="delta_min",
        direction="minus",
        alpha=alpha,
        degree=degree,
        min_train_years=min_train_years,
        window_years=window_years,
        eval_last_years=eval_last_years,
    )

    delta_max_all = yearly["delta_max"].to_numpy(dtype=float)
    delta_min_all = yearly["delta_min"].to_numpy(dtype=float)

    delta_max_all = delta_max_all[~np.isnan(delta_max_all)]
    delta_min_all = delta_min_all[~np.isnan(delta_min_all)]

    delta_max_tail = _tail_by_window(delta_max_all, window_years)
    delta_min_tail = _tail_by_window(delta_min_all, window_years)

    delta_max = float(np.quantile(delta_max_tail, q_max)) if delta_max_tail.size else float("nan")
    delta_min = float(np.quantile(delta_min_tail, q_min)) if delta_min_tail.size else float("nan")

    max_2026 = mean_tmax_2026 + delta_max
    min_2026 = mean_tmin_2026 - delta_min

    return August2026Forecast(
        mean_tmax=mean_tmax_2026,
        mean_tmin=mean_tmin_2026,
        max_tmax=max_2026,
        min_tmin=min_2026,
        q_max=q_max,
        q_min=q_min,
        delta_max=delta_max,
        delta_min=delta_min,
        mae_max=float(mae_max),
        mae_min=float(mae_min),
    )
