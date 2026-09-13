"""
Unit tests for the synthetic data generator.

Run with:  pytest tests/test_data_generation.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.data_generation.generate_data import GenConfig, generate


@pytest.fixture(scope="module")
def small_dataset():
    cfg = GenConfig(start_date="2024-01-01", end_date="2024-01-07", n_customers=500, seed=1)
    return generate(cfg)


def test_generate_returns_all_expected_tables(small_dataset):
    expected_tables = {
        "dim_airports", "dim_aircraft", "dim_routes",
        "dim_customers", "fact_flights", "fact_bookings",
    }
    assert expected_tables == set(small_dataset.keys())


def test_generation_is_deterministic_given_seed():
    cfg = GenConfig(start_date="2024-01-01", end_date="2024-01-03", n_customers=200, seed=99)
    run_1 = generate(cfg)
    run_2 = generate(cfg)
    pd.testing.assert_frame_equal(run_1["fact_flights"], run_2["fact_flights"])
    pd.testing.assert_frame_equal(run_1["fact_bookings"], run_2["fact_bookings"])


def test_flight_capacity_is_non_negative(small_dataset):
    flights = small_dataset["fact_flights"]
    assert (flights["total_capacity"] > 0).all()
    assert (flights["passengers"] >= 0).all()


def test_bookings_reference_valid_flights(small_dataset):
    valid_flight_ids = set(small_dataset["fact_flights"]["flight_id"])
    booking_flight_ids = set(small_dataset["fact_bookings"]["flight_id"])
    assert booking_flight_ids.issubset(valid_flight_ids)


def test_injected_quality_issues_are_present(small_dataset):
    """The generator deliberately injects messiness -- confirm it's really
    there, since the data-quality module's whole job is to detect it."""
    bookings = small_dataset["fact_bookings"]
    flights = small_dataset["fact_flights"]

    assert bookings["customer_id"].isna().sum() > 0, "expected some missing customer_id"
    assert (bookings["fare_usd"] < 0).sum() > 0, "expected some negative fares"
    assert bookings.duplicated(subset=["booking_id"]).sum() > 0, "expected duplicate booking_ids"
    assert (flights["destination"] == "ZZZ").sum() > 0, "expected some invalid airport codes"


def test_cabin_classes_are_valid(small_dataset):
    assert set(small_dataset["fact_bookings"]["cabin_class"].unique()) <= {"ECONOMY", "BUSINESS"}
