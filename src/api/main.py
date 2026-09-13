"""
Airline Intelligence API
=========================
A thin FastAPI layer over the analytical warehouse, demonstrating how the
platform's metrics can be exposed as a real data product (e.g. to power an
internal dashboard, a Slack bot, or a GenAI "analytics copilot") rather than
living only in static notebooks.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive Swagger docs.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

DB_PATH = os.environ.get("AIRLINE_DB_PATH", "data/processed/airline.db")

app = FastAPI(
    title="Airline Intelligence API",
    description="Simulated airline network, revenue and operational analytics.",
    version="1.0.0",
)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


class RouteMetric(BaseModel):
    route: str
    flights: int
    revenue_usd: float
    cost_usd: float
    profit_usd: float
    avg_load_factor_pct: float


class KpiSummary(BaseModel):
    total_revenue_usd: float
    total_passengers: int
    avg_load_factor_pct: float
    on_time_performance_pct: float
    cancellation_rate_pct: float
    data_quality_score_pct: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/kpis/summary", response_model=KpiSummary)
def kpi_summary():
    with get_conn() as conn:
        try:
            flights = pd.read_sql("SELECT * FROM fact_flights_clean", conn)
            bookings = pd.read_sql("SELECT fare_usd FROM fact_bookings_clean", conn)
            dq = pd.read_sql(
                "SELECT dataset, valid_records, total_records FROM data_quality_score_history "
                "ORDER BY run_id DESC LIMIT 2",
                conn,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"Warehouse not ready: {exc}") from exc

    non_cancelled = flights[flights["is_cancelled"] == 0]
    avg_lf = (non_cancelled["passengers"] / non_cancelled["total_capacity"]).mean() * 100
    otp = (non_cancelled["departure_delay_minutes"] <= 15).mean() * 100
    cancel_rate = flights["is_cancelled"].mean() * 100
    dq_score = (
        round(100 * dq["valid_records"].sum() / dq["total_records"].sum(), 2)
        if not dq.empty else 100.0
    )

    return KpiSummary(
        total_revenue_usd=round(float(bookings["fare_usd"].sum()), 2),
        total_passengers=int(non_cancelled["passengers"].sum()),
        avg_load_factor_pct=round(float(avg_lf), 1),
        on_time_performance_pct=round(float(otp), 1),
        cancellation_rate_pct=round(float(cancel_rate), 2),
        data_quality_score_pct=dq_score,
    )


@app.get("/routes/profitability", response_model=list[RouteMetric])
def routes_profitability(limit: int = Query(10, ge=1, le=50)):
    with get_conn() as conn:
        flights = pd.read_sql(
            "SELECT flight_id, route_id, origin || '-' || destination AS route, "
            "total_capacity, passengers, operating_cost_usd FROM fact_flights_clean",
            conn,
        )
        revenue = pd.read_sql(
            "SELECT flight_id, SUM(fare_usd) AS revenue FROM fact_bookings_clean GROUP BY flight_id",
            conn,
        )
    df = flights.merge(revenue, on="flight_id", how="left")
    df["revenue"] = df["revenue"].fillna(0)

    grouped = df.groupby("route").agg(
        flights=("route_id", "count"),
        revenue_usd=("revenue", "sum"),
        cost_usd=("operating_cost_usd", "sum"),
        load_factor=("passengers", lambda s: (s / df.loc[s.index, "total_capacity"]).mean()),
    ).reset_index()
    grouped["profit_usd"] = grouped["revenue_usd"] - grouped["cost_usd"]
    grouped = grouped.sort_values("profit_usd", ascending=False).head(limit)

    return [
        RouteMetric(
            route=row["route"],
            flights=int(row["flights"]),
            revenue_usd=round(row["revenue_usd"], 2),
            cost_usd=round(row["cost_usd"], 2),
            profit_usd=round(row["profit_usd"], 2),
            avg_load_factor_pct=round(row["load_factor"] * 100, 1),
        )
        for _, row in grouped.iterrows()
    ]


@app.get("/data-quality/issues")
def open_data_quality_issues(dataset: str | None = None):
    with get_conn() as conn:
        query = "SELECT * FROM data_quality_issues WHERE status = 'OPEN'"
        params = ()
        if dataset:
            query += " AND dataset = ?"
            params = (dataset,)
        df = pd.read_sql(query, conn, params=params)
    return {"open_issue_count": len(df), "issues": df.to_dict(orient="records")[:100]}
