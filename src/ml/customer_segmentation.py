"""
Model 2 — Customer Segmentation (RFM + K-Means)
=================================================
Builds Recency / Frequency / Monetary features per customer, standardises
them, clusters with K-Means (elbow + silhouette used to pick k), and
attaches a plain-English business label to each resulting cluster.

Usage:
    python src/ml/customer_segmentation.py --db data/processed/airline.db
"""
from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def build_rfm(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    bookings = pd.read_sql(
        "SELECT customer_id, fare_usd, travel_date FROM fact_bookings_clean", conn
    )
    conn.close()

    bookings["travel_date"] = pd.to_datetime(bookings["travel_date"])
    snapshot_date = bookings["travel_date"].max() + pd.Timedelta(days=1)

    rfm = bookings.groupby("customer_id").agg(
        recency_days=("travel_date", lambda s: (snapshot_date - s.max()).days),
        frequency=("travel_date", "count"),
        monetary=("fare_usd", "sum"),
    ).reset_index()

    return rfm


def score_clusters(X_scaled: np.ndarray, k_range=range(2, 8)) -> pd.DataFrame:
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def label_segments(profile: pd.DataFrame) -> pd.DataFrame:
    """Assigns a business-friendly name to each cluster based on where its
    mean R/F/M sits relative to the overall population (simple quantile
    rules — transparent and easy to defend in an interview, unlike a black
    box heuristic)."""
    profile = profile.copy()
    # Rank clusters by monetary value and frequency instead of thresholding
    # against a single median -- with few clusters this avoids collapsing
    # several genuinely different clusters onto the same label.
    profile["monetary_rank"] = profile["monetary"].rank(ascending=False)
    profile["frequency_rank"] = profile["frequency"].rank(ascending=False)
    n = len(profile)

    def label(row):
        top_third = max(1, round(n / 3))
        is_top_spender = row["monetary_rank"] <= top_third
        is_top_flyer = row["frequency_rank"] <= top_third
        is_recent = row["recency_days"] <= profile["recency_days"].median()
        if is_top_spender and is_top_flyer:
            return "Business Frequent Flyer"
        if is_top_spender and not is_top_flyer:
            return "Premium Leisure"
        if not is_recent and row["monetary_rank"] > n - top_third:
            return "Inactive / At-Risk"
        if is_top_flyer and not is_top_spender:
            return "Price-Sensitive Regular"
        return "Regular Traveller"

    profile["segment_label"] = profile.apply(label, axis=1)
    return profile.drop(columns=["monetary_rank", "frequency_rank"])


def run(db_path: str, k: int | None = None):
    rfm = build_rfm(db_path)

    # Frequency and monetary are heavily right-skewed (a small number of very
    # frequent flyers), which would otherwise dominate Euclidean distance in
    # K-Means. Log-transforming before scaling is standard RFM practice and
    # gives much more balanced, interpretable clusters.
    features = pd.DataFrame(
        {
            "recency_days": rfm["recency_days"],
            "frequency_log": np.log1p(rfm["frequency"]),
            "monetary_log": np.log1p(rfm["monetary"]),
        }
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    scores = score_clusters(X_scaled)
    chosen_k = k or int(scores.loc[scores["silhouette"].idxmax(), "k"])

    km = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X_scaled)

    profile = rfm.groupby("cluster").agg(
        customers=("customer_id", "count"),
        recency_days=("recency_days", "mean"),
        frequency=("frequency", "mean"),
        monetary=("monetary", "mean"),
    ).reset_index()
    profile = label_segments(profile)

    return scores, chosen_k, rfm, profile


def main():
    parser = argparse.ArgumentParser(description="RFM + K-Means customer segmentation")
    parser.add_argument("--db", default="data/processed/airline.db")
    parser.add_argument("--k", type=int, default=4,
                         help="Number of segments (default 4, business-actionable; "
                              "pass 0 to use the silhouette-optimal k instead)")
    args = parser.parse_args()

    k = None if args.k == 0 else args.k
    scores, chosen_k, rfm, profile = run(args.db, k)

    print("Silhouette scores by k:")
    print(scores.round(3).to_string(index=False))
    print(f"\nChosen k = {chosen_k} (highest silhouette score)\n")

    print("Segment profiles:")
    print(profile.round(1).to_string(index=False))

    total = profile["customers"].sum()
    print("\nBusiness summary:")
    for _, row in profile.sort_values("monetary", ascending=False).iterrows():
        pct = 100 * row["customers"] / total
        print(
            f"  {row['segment_label']:26s} {int(row['customers']):>6,} customers "
            f"({pct:4.1f}%) | avg spend ${row['monetary']:,.0f} | "
            f"avg trips {row['frequency']:.1f} | avg recency {row['recency_days']:.0f}d"
        )


if __name__ == "__main__":
    main()