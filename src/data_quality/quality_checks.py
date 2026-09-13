"""
Data Quality Framework
======================
Runs automated completeness / validity / uniqueness / referential-integrity
checks against the raw fact tables, logs every failing record into
data_quality_issues, and computes a per-dataset Data Quality Score:

    quality_score = valid_records / total_records * 100

This is designed to mirror what a production airline data-audit function
would actually check (the brief that inspired this project explicitly
calls out "auditing system data for compliance, accuracy and duplication management")

Usage:
    python src/data_quality/quality_checks.py --db data/processed/airline.db
"""
from __future__ import annotations

import argparse
import sqlite3
from ast import List
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

@dataclass
class QualityIssue:
    dataset: str
    record_id: str
    issue_type: str
    severity: str # LOW, MEDIUM, HIGH, CRITICAL

@dataclass
class QualityReport:
    dataset: str
    total_records: int
    issues: List[QualityIssue] = field(default_factory=list)

    @property
    def invalid_record_ids(self) -> set[str]:
        return {i.record_id for i in self.issues}

    @property
    def valid_records(self) -> int:
        return self.total_records - len(self.invalid_record_ids)

    @property
    def quality_score(self) -> float:
        if self.total_records == 0:
            return 100.0
        return round(100 * self.valid_records / self.total_records, 2)

    def issue_breakdown(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(columns=["issue_type", "count", "pct_of_total"])
        df = pd.DataFrame([vars(i) for i in self.issues])
        counts = df.groupby("issue_type").size().rename("count").reset_index()
        counts["pct_of_total"] = round(100 * counts["count"] / self.total_records, 2)
        return counts.sort_values("count", ascending=False)

def check_bookings(conn: sqlite3.Connection) -> QualityReport:
    bookings = pd.read_sql("SELECT * FROM fact_bookings", conn)
    report = QualityReport(dataset="fact_bookings", total_records=len(bookings))

    # 1. Completeness: missing customer_id
    missing_cust = bookings[bookings["customer_id"].isna()]
    report.issues += [
        QualityIssue("fact_bookings", bid, "missing_customer_id", "MEDIUM")
        for bid in missing_cust["booking_id"]
    ]

    # 2. Validity: negative / impossible fares
    bad_fare = bookings[bookings["fare_usd"] <= 0]
    report.issues += [
        QualityIssue("fact_bookings", bid, "invalid_fare", "HIGH")
        for bid in bad_fare["booking_id"]
    ]

    # 3. Uniqueness: duplicate booking_id (exact duplicate rows)
    dup_mask = bookings.duplicated(subset=["booking_id"], keep="first")
    dup_ids = bookings.loc[dup_mask, "booking_id"]
    report.issues += [
        QualityIssue("fact_bookings", bid, "duplicate_booking", "HIGH")
        for bid in dup_ids
    ]

    # 4. Validity: travel_date before booking_date (logically impossible)
    bookings["booking_date"] = pd.to_datetime(bookings["booking_date"])
    bookings["travel_date"] = pd.to_datetime(bookings["travel_date"])
    bad_dates = bookings[bookings["travel_date"] < bookings["booking_date"]]
    report.issues += [
        QualityIssue("fact_bookings", bid, "travel_before_booking", "CRITICAL")
        for bid in bad_dates["booking_id"]
    ]

    return report


def check_flights(conn: sqlite3.Connection) -> QualityReport:
    flights = pd.read_sql("SELECT * FROM fact_flights", conn)
    airports = set(pd.read_sql("SELECT airport_code FROM dim_airport", conn)["airport_code"])
    report = QualityReport(dataset="fact_flights", total_records=len(flights))

    # 1. Referential integrity: invalid airport codes
    bad_airport = flights[
        ~flights["origin"].isin(airports) | ~flights["destination"].isin(airports)
    ]
    report.issues += [
        QualityIssue("fact_flights", str(fid), "invalid_airport_code", "CRITICAL")
        for fid in bad_airport["flight_id"]
    ]

    # 2. Business-rule validity: passengers exceeding total capacity
    over_capacity = flights[flights["passengers"] > flights["total_capacity"]]
    report.issues += [
        QualityIssue("fact_flights", str(fid), "capacity_violation", "HIGH")
        for fid in over_capacity["flight_id"]
    ]

    # 3. Validity: negative operating/fuel cost (would indicate an upstream bug)
    bad_cost = flights[(flights["operating_cost_usd"] < 0) | (flights["fuel_cost_usd"] < 0)]
    report.issues += [
        QualityIssue("fact_flights", str(fid), "negative_cost", "HIGH")
        for fid in bad_cost["flight_id"]
    ]

    return report


def persist_report(conn: sqlite3.Connection, report: QualityReport) -> None:
    """Writes issues to data_quality_issues and a summary row to
    data_quality_score_history, creating the tables if needed (SQLite demo;
    on Postgres these already exist from schema.sql)."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_issues (
            issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT, record_id TEXT, issue_type TEXT, severity TEXT,
            detected_at TEXT, status TEXT DEFAULT 'OPEN', resolved_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_score_history (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp TEXT, dataset TEXT, total_records INTEGER,
            valid_records INTEGER, quality_score REAL
        )
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    for issue in report.issues:
        conn.execute(
            "INSERT INTO data_quality_issues (dataset, record_id, issue_type, severity, detected_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (issue.dataset, issue.record_id, issue.issue_type, issue.severity, now),
        )
    conn.execute(
        "INSERT INTO data_quality_score_history "
        "(run_timestamp, dataset, total_records, valid_records, quality_score) VALUES (?,?,?,?,?)",
        (now, report.dataset, report.total_records, report.valid_records, report.quality_score),
    )
    conn.commit()


def run_all(db_path: str) -> list[QualityReport]:
    conn = sqlite3.connect(db_path)
    reports = [check_flights(conn), check_bookings(conn)]
    for r in reports:
        persist_report(conn, r)
    conn.close()
    return reports


def print_summary(reports: list[QualityReport]) -> None:
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    for r in reports:
        print(f"\nDataset: {r.dataset}")
        print(f"  Total records : {r.total_records:,}")
        print(f"  Valid records : {r.valid_records:,}")
        print(f"  Quality score : {r.quality_score}%")
        breakdown = r.issue_breakdown()
        if not breakdown.empty:
            print("  Issues:")
            for _, row in breakdown.iterrows():
                print(f"    - {row['issue_type']:25s} {row['count']:>8,} ({row['pct_of_total']}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run data quality checks")
    parser.add_argument("--db", default="data/processed/airline.db")
    args = parser.parse_args()
    reports = run_all(args.db)
    print_summary(reports)


if __name__ == "__main__":
    main()

