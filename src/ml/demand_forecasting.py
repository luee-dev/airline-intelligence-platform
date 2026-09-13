"""
Model 3 — Passenger Demand Forecasting
========================================
Forecasts daily passenger demand (network-wide, and per-route) using:

  1. Baselines: naive (yesterday), 7-day moving average, seasonal naive
     (same weekday last week).
  2. A regression model (Random Forest / Gradient Boosting) using lag and
     calendar features -- the same features a SARIMA/Prophet model would
     implicitly use, but framed as supervised learning so it is trivial to
     add exogenous regressors later (price, promotions, fuel cost, etc).

Models are compared with a proper time-series backtest (train on the past,a
predict the future -- never a random train/test split, which would leak
information backwards in time).

Statsmodels' SARIMAX/ExponentialSmoothing is a natural next step once
`statsmodels` is installed (see requirements.txt); the interface would slot
into `evaluate_backtest` unchanged.

Usage:
    python src/ml/demand_forecasting.py --db data/processed/airline.db --horizon 14
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def load_daily_demand(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT travel_date, COUNT(*) AS bookings
        FROM fact_bookings_clean
        GROUP BY travel_date
        ORDER BY travel_date
        """,
        conn,
    )
    conn.close()
    df["travel_date"] = pd.to_datetime(df["travel_date"])
    df = df.set_index("travel_date").asfreq("D").fillna(0)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["lag_1"] = df["bookings"].shift(1)
    df["lag_7"] = df["bookings"].shift(7)
    df["rolling_mean_7"] = df["bookings"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df["bookings"].shift(1).rolling(14).mean()
    return df


def baseline_forecasts(train: pd.Series, horizon: int) -> dict[str, np.ndarray]:
    naive = np.repeat(train.iloc[-1], horizon)
    moving_avg = np.repeat(train.tail(7).mean(), horizon)
    # seasonal naive: repeat the last 7 observed values, tiled to horizon
    last_week = train.tail(7).values
    seasonal_naive = np.tile(last_week, int(np.ceil(horizon / 7)))[:horizon]
    return {"naive": naive, "moving_avg_7d": moving_avg, "seasonal_naive": seasonal_naive}


def evaluate_backtest(df: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    feat_df = add_features(df).dropna()
    train = feat_df.iloc[:-horizon]
    test = feat_df.iloc[-horizon:]

    feature_cols = ["dow", "month", "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(train[feature_cols], train["bookings"])
    ml_preds = model.predict(test[feature_cols])

    baselines = baseline_forecasts(df["bookings"].loc[:test.index[0] - pd.Timedelta(days=1)], horizon)

    results = []
    actual = test["bookings"].values
    for name, preds in {**baselines, "gradient_boosting": ml_preds}.items():
        preds = np.clip(preds, 0, None)
        mae = mean_absolute_error(actual, preds)
        mape = mean_absolute_percentage_error(np.where(actual == 0, 1, actual), preds)
        results.append({"model": name, "mae": round(mae, 1), "mape_pct": round(mape * 100, 1)})

    return pd.DataFrame(results).sort_values("mae")


def forecast_future(df: pd.DataFrame, horizon: int = 14) -> pd.DataFrame:
    """Iteratively forecasts `horizon` days beyond the last observed date,
    refitting lag features at each step (a simple recursive forecast)."""
    feat_df = add_features(df).dropna()
    feature_cols = ["dow", "month", "lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]
    model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(feat_df[feature_cols], feat_df["bookings"])

    history = df["bookings"].copy()
    future_dates = pd.date_range(history.index[-1] + pd.Timedelta(days=1), periods=horizon)
    preds = []
    for date in future_dates:
        row = {
            "dow": date.dayofweek,
            "month": date.month,
            "lag_1": history.iloc[-1],
            "lag_7": history.iloc[-7],
            "rolling_mean_7": history.tail(7).mean(),
            "rolling_mean_14": history.tail(14).mean(),
        }
        pred = max(0, model.predict(pd.DataFrame([row]))[0])
        preds.append(pred)
        history.loc[date] = pred  # feed the forecast back in for the next step

    return pd.DataFrame({"date": future_dates, "forecast_bookings": np.round(preds).astype(int)})


def main():
    parser = argparse.ArgumentParser(description="Passenger demand forecasting")
    parser.add_argument("--db", default="data/processed/airline.db")
    parser.add_argument("--horizon", type=int, default=14)
    args = parser.parse_args()

    df = load_daily_demand(args.db)
    print(f"Loaded {len(df)} days of demand history "
          f"({df.index.min().date()} to {df.index.max().date()})\n")

    print(f"Backtest — last {args.horizon} days held out, model trained only on prior data:")
    comparison = evaluate_backtest(df, args.horizon)
    print(comparison.to_string(index=False))

    print(f"\nForecast for the next {args.horizon} days (trained on full history):")
    future = forecast_future(df, args.horizon)
    print(future.to_string(index=False))


if __name__ == "__main__":
    main()
