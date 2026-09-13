"""
Model 1 - Flight Delay Risk Prediction
=======================================
Predicts P(departure delay > 15 minutes) using route, schedule and
aircraft-utilisation features that are known *before* departure (no
leakage from the outcome itself).

Compares Logistic Regression (interpretable baseline) against Random
Forest and Gradient Boosting (nonlinear, usually stronger). Swap in
XGBoost by installing it and replacing GradientBoostingClassifier below
-- the feature pipeline is identical either way.

Usage:
    python src/ml/delay_prediction.py --db data/processed/airline.db
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

DELAY_THRESHOLD_MIN = 15

def load_features(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT f.*, r.distance_km
        FROM fact_flights_clean f
        JOIN dim_route r ON r.route_id = f.route_id
        """,
        conn,
    )
    conn.close()

    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["month"] = df["flight_date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([4, 5]).astype(int)

    # Historical route delay: mean delay of the route computed on PRIOR
    # flights only (expanding mean, shifted by 1) to avoid target leakage.
    df = df.sort_values(["route_id", "flight_date"])
    df["historical_route_delay"] = (
        df.groupby("route_id")["departure_delay_minutes"]
        .apply(lambda s: s.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["historical_route_delay"] = df["historical_route_delay"].fillna(
        df["departure_delay_minutes"].mean()
    )

    df["is_hub_route"] = ((df["origin"] == "NBO") | (df["destination"] == "NBO")).astype(int)
    df["target_delayed"] = (df["departure_delay_minutes"] > DELAY_THRESHOLD_MIN).astype(int)

    return df


FEATURES_NUM = ["distance_km", "scheduled_departure_hour", "historical_route_delay",
                "flight_duration_minutes", "day_of_week", "month"]
FEATURES_CAT = ["aircraft_type", "is_weekend", "is_hub_route"]


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )
    return Pipeline([("prep", preprocessor), ("model", model)])


def train_and_compare(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df["target_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
        ),
    }

    results = []
    for name, model in models.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, proba)
        report = classification_report(y_test, preds, output_dict=True, zero_division=0)
        results.append(
            {
                "model": name,
                "roc_auc": round(auc, 4),
                "precision_delayed": round(report["1"]["precision"], 3),
                "recall_delayed": round(report["1"]["recall"], 3),
                "f1_delayed": round(report["1"]["f1-score"], 3),
            }
        )
        if name == "RandomForest":
            best_pipe = pipe

    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    return comparison, best_pipe, X_test, y_test


def top_risk_flights(pipe: Pipeline, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    X = df[FEATURES_NUM + FEATURES_CAT]
    df = df.copy()
    df["delay_probability"] = pipe.predict_proba(X)[:, 1]
    df["risk_level"] = pd.cut(
        df["delay_probability"], bins=[0, 0.3, 0.6, 1.0], labels=["LOW", "MEDIUM", "HIGH"]
    )
    cols = ["flight_number", "origin", "destination", "flight_date",
            "delay_probability", "risk_level"]
    return df.sort_values("delay_probability", ascending=False)[cols].head(n)


def main():
    parser = argparse.ArgumentParser(description="Train flight delay risk model")
    parser.add_argument("--db", default="data/processed/airline.db")
    args = parser.parse_args()

    print("Loading features...")
    df = load_features(args.db)
    print(f"  {len(df):,} flights | delay rate: {df['target_delayed'].mean():.1%}")

    print("\nTraining and comparing models...")
    comparison, best_pipe, X_test, y_test = train_and_compare(df)
    print(comparison.to_string(index=False))

    print("\nTop 10 highest-risk upcoming flights (illustrative, scored on full dataset):")
    print(top_risk_flights(best_pipe, df).to_string(index=False))


if __name__ == "__main__":
    main()
