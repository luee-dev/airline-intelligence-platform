"""
Unit tests for the data-quality framework.

Run with:  pytest tests/test_data_quality.py -v
"""
import sqlite3

import pytest

from src.data_generation.generate_data import GenConfig, generate
from src.data_quality.quality_checks import check_bookings, check_flights, persist_report


@pytest.fixture()
def in_memory_db():
    cfg = GenConfig(start_date="2024-01-01", end_date="2024-01-10", n_customers=800, seed=7)
    tables = generate(cfg)

    conn = sqlite3.connect(":memory:")
    tables["dim_airports"].rename(columns={}).to_sql("dim_airport", conn, index=False)
    tables["fact_flights"].to_sql("fact_flights", conn, index=False)
    tables["fact_bookings"].to_sql("fact_bookings", conn, index=False)
    yield conn
    conn.close()


def test_check_flights_detects_invalid_airport_codes(in_memory_db):
    report = check_flights(in_memory_db)
    issue_types = {i.issue_type for i in report.issues}
    assert "invalid_airport_code" in issue_types


def test_check_bookings_detects_duplicates_and_missing_values(in_memory_db):
    report = check_bookings(in_memory_db)
    issue_types = {i.issue_type for i in report.issues}
    assert "duplicate_booking" in issue_types
    assert "missing_customer_id" in issue_types
    assert "invalid_fare" in issue_types


def test_quality_score_is_between_0_and_100(in_memory_db):
    flights_report = check_flights(in_memory_db)
    bookings_report = check_bookings(in_memory_db)
    for report in (flights_report, bookings_report):
        assert 0 <= report.quality_score <= 100


def test_quality_score_drops_when_issues_present(in_memory_db):
    report = check_bookings(in_memory_db)
    # We deliberately inject issues in the generator, so a perfect 100% score
    # would indicate the checks are silently failing to detect anything.
    assert report.quality_score < 100.0


def test_persist_report_writes_issues_and_history(in_memory_db):
    report = check_flights(in_memory_db)
    persist_report(in_memory_db, report)

    issue_count = in_memory_db.execute(
        "SELECT COUNT(*) FROM data_quality_issues"
    ).fetchone()[0]
    history_count = in_memory_db.execute(
        "SELECT COUNT(*) FROM data_quality_score_history"
    ).fetchone()[0]

    assert issue_count == len(report.issues)
    assert history_count == 1
