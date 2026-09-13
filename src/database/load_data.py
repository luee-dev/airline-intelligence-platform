"""
Load synthetic CSVs into the analytical warehouse

Supports two backends, selected via the DATABASE_URL environment variable:
  * SQLite (default, zero setup)   -> sqlite:///data/processed/airline.db
  * PostgreSQL (production/docker) -> postgresql+psycopg2://user:pass@host:5432/airline

Usage:
    python src/database/load_data.py --data-dir data/synthetic
    DATABASE_URL=postgresql+psycopg2://airline:airline@localhost:5432/airline \
        python src/database/load_data.py --data-dir data/synthetic
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

DEFAULT_SQLITE_PATH = "data/processed/airline.db"

def _get_connection(database_url: str | None):
    if database_url and database_url.startswith("postgresql"):
        from sqlalchemy import create_engine
        engine = create_engine(database_url)
        return engine, "postgres"
    else:
        path = Path(DEFAULT_SQLITE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))
        return conn, "sqlite"

def _build_dim_date(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start=start, end=end, freq="D")
    return pd.DataFrame(
        {
            "quarter": dates.quarter,
            "month": dates.month,
            "month_name": dates.strftime("%B"),
            "day_of_week": dates.dayofweek,
            "day_name": dates.strftime("%A"),
            "is_weekend": dates.dayofweek.isin([5, 6]),
            "week_of_year": dates.isocalendar().week.astype(int),
        }
    )

def load(data_dir: str, database_url: str | None = None) -> None:
    data_path = Path(data_dir)
    conn, backend = _get_connection(database_url)
    print(f"Loading into backend: {backend}")

    # Apply DDL for Postgres (SQLite gets a simplified auto-created schema via to_sql)
    if backend == "postgres":
        schema_sql = Path(__file__).parent / "schema.sql"
        with conn.connect() as c:
            for stmt in schema_sql.read_text().split(";"):
                if stmt.strip():
                    c.exec_driver_sql(stmt)
            c.commit()

    tables = {
        "dim_airports": "dim_airport",
        "dim_aircraft": "dim_aircraft",
        "dim_routes": "dim_route",
        "dim_customers": "dim_customer",
        "fact_flights": "fact_flights",
        "fact_bookings": "fact_bookings",
    }

    flights_df = pd.read_csv(data_path / "fact_flights.csv", parse_dates=["flight_date"])
    dim_date = _build_dim_date(
        flights_df["flight_date"].min().date(), flights_df["flight_date"].max().date()
    )
    dim_date.to_sql("dim_date", conn, if_exists="replace", index=False)
    print(f"  loaded dim_date            {len(dim_date):>10,} rows")

    for csv_name, table_name in tables.items():
        df = pd.read_csv(data_path / f"{csv_name}.csv")
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=50_000)
        print(f"  loaded {table_name:20s} {len(df):>10,} rows")

    if backend == "sqlite":
        conn.commit()
        conn.close()

    print("\nWarehouse load complete.")

def main():
    parser = argparse.ArgumentParser(description="Load synthetic CSVs into the analytical warehouse")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    load(args.data_dir, args.database_url)

if __name__ == "__main__":
    main()

