"""
Model 4 — Route Economics & Price Elasticity of Demand
=========================================================
Two related analyses:

1. Route contribution & break-even load factor: standard airline-economics
   arithmetic (revenue - variable operating cost = contribution; break-even
   load factor = cost / (fare * capacity)).

2. An econometric demand model:

        log(bookings) = b0 + b1*log(avg_fare) + b2*month_dummies + e

   estimated by OLS (via scikit-learn's LinearRegression on log-transformed
   variables, so no extra dependency is required -- this is exactly the
   log-log elasticity specification an economist would run in statsmodels;
   swapping in statsmodels.OLS gives you formal p-values/confidence
   intervals with the same feature matrix if you want to report them).

   b1 is the price elasticity of demand: "a 1% increase in average fare is
   associated with a b1% change in bookings, holding season constant."

Usage:
    python src/ml/route_economics.py --db data/processed/airline.db
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def route_contribution(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    flights = pd.read_sql(
        """
        SELECT flight_id, route_id, origin || '-' || destination AS route,
               total_capacity, operating_cost_usd
        FROM fact_flights_clean
        """,
        conn,
    )
    revenue_by_flight = pd.read_sql(
        "SELECT flight_id, SUM(fare_usd) AS flight_revenue "
        "FROM fact_bookings_clean GROUP BY flight_id",
        conn,
    )
    conn.close()

    df = flights.merge(revenue_by_flight, on="flight_id", how="left")
    df["flight_revenue"] = df["flight_revenue"].fillna(0)

    grouped = df.groupby("route").agg(
        flights=("route_id", "count"),
        total_capacity=("total_capacity", "sum"),
        revenue_usd=("flight_revenue", "sum"),
        cost_usd=("operating_cost_usd", "sum"),
    ).reset_index()

    grouped["contribution_usd"] = grouped["revenue_usd"] - grouped["cost_usd"]
    grouped["margin_pct"] = 100 * grouped["contribution_usd"] / grouped["revenue_usd"]
    grouped["avg_fare_per_seat"] = grouped["revenue_usd"] / grouped["total_capacity"]

    # Break-even load factor: what fraction of seats (at the observed avg
    # fare) would need to be sold just to cover operating cost.
    grouped["breakeven_load_factor_pct"] = 100 * (
        grouped["cost_usd"] / (grouped["avg_fare_per_seat"] * grouped["total_capacity"])
    ).clip(upper=2.0) * 1  # clip guards against div-by-zero edge cases

    return grouped.sort_values("contribution_usd", ascending=False)


def price_elasticity(db_path: str) -> tuple[pd.DataFrame, dict]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT travel_date, fare_usd
        FROM fact_bookings_clean
        WHERE cabin_class = 'ECONOMY'
        """,
        conn,
    )
    conn.close()
    df["travel_date"] = pd.to_datetime(df["travel_date"])

    daily = df.groupby("travel_date").agg(
        bookings=("fare_usd", "count"), avg_fare=("fare_usd", "mean")
    ).reset_index()
    daily["month"] = daily["travel_date"].dt.month

    daily["log_bookings"] = np.log(daily["bookings"])
    daily["log_fare"] = np.log(daily["avg_fare"])

    month_dummies = pd.get_dummies(daily["month"], prefix="month", drop_first=True)
    X = pd.concat([daily[["log_fare"]], month_dummies], axis=1).astype(float)
    y = daily["log_bookings"]

    model = LinearRegression()
    model.fit(X, y)
    elasticity = model.coef_[0]
    r2 = model.score(X, y)

    summary = {
        "price_elasticity_of_demand": round(elasticity, 3),
        "r_squared": round(r2, 3),
        "n_obs": len(daily),
        "interpretation": (
            f"A 10% increase in average economy fare is associated with a "
            f"{abs(elasticity) * 10:.1f}% {'decrease' if elasticity < 0 else 'increase'} "
            f"in daily bookings, holding month (seasonality) constant."
        ),
    }
    return daily, summary


def main():
    parser = argparse.ArgumentParser(description="Route economics and price elasticity")
    parser.add_argument("--db", default="data/processed/airline.db")
    args = parser.parse_args()

    print("=" * 70)
    print("ROUTE CONTRIBUTION & BREAK-EVEN LOAD FACTOR")
    print("=" * 70)
    contrib = route_contribution(args.db)
    cols = ["route", "flights", "revenue_usd", "cost_usd", "contribution_usd",
            "margin_pct", "breakeven_load_factor_pct"]
    print(contrib[cols].round(1).head(10).to_string(index=False))

    bottom_quartile_cutoff = contrib["margin_pct"].quantile(0.25)
    print(f"\nRoutes flagged for review (bottom quartile by margin, i.e. margin_pct <= "
          f"{bottom_quartile_cutoff:.1f}%) -- candidates for capacity reallocation or fare review:")
    flagged = contrib[contrib["margin_pct"] <= bottom_quartile_cutoff][cols]
    print(flagged.round(1).to_string(index=False) if not flagged.empty else "  None")

    print("\n" + "=" * 70)
    print("PRICE ELASTICITY OF DEMAND (economy cabin, network-wide)")
    print("=" * 70)
    _, summary = price_elasticity(args.db)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
