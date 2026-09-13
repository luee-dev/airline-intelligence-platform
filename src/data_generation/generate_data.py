"""
Synthetic Airline Data Generator
=================================
Generates a realistic, internally-consistent airline operations dataset
(flights, bookings, customers, routes, aircraft, delays, revenue) that is
used throughout the rest of this platform.

IMPORTANT: This is 100% simulated data, generated with fixed random seeds
for reproducibility. It is inspired by publicly known airline business
dynamics (load factors, seasonality, hub-and-spoke networks, cost
structures) but contains no real airline, passenger or booking data.

Usage:
    python src/data_generation/generate_data.py --start 2024-01-01 --end 2025-12-31 --out data/synthetic
"""
from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 44


# --------------------------------------------------------------------------- #
# Reference data (dimensions)
# --------------------------------------------------------------------------- #

AIRPORTS = pd.DataFrame(
    [
        # code, name, city, country, region, hub_flag
        ("NBO", "Jomo Kenyatta Intl", "Nairobi", "Kenya", "East Africa", 1),
        ("MBA", "Moi International", "Mombasa", "Kenya", "East Africa", 0),
        ("KIS", "Kisumu International", "Kisumu", "Kenya", "East Africa", 0),
        ("EBB", "Entebbe International", "Entebbe", "Uganda", "East Africa", 0),
        ("DAR", "Julius Nyerere Intl", "Dar es Salaam", "Tanzania", "East Africa", 0),
        ("KGL", "Kigali International", "Kigali", "Rwanda", "East Africa", 0),
        ("ADD", "Bole International", "Addis Ababa", "Ethiopia", "East Africa", 0),
        ("JNB", "O.R. Tambo Intl", "Johannesburg", "South Africa", "Southern Africa", 0),
        ("LOS", "Murtala Muhammed Intl", "Lagos", "Nigeria", "West Africa", 0),
        ("ACC", "Kotoka International", "Accra", "Ghana", "West Africa", 0),
        ("CAI", "Cairo International", "Cairo", "Egypt", "North Africa", 0),
        ("DXB", "Dubai International", "Dubai", "UAE", "Middle East", 0),
        ("LHR", "Heathrow", "London", "United Kingdom", "Europe", 0),
        ("AMS", "Schiphol", "Amsterdam", "Netherlands", "Europe", 0),
        ("CDG", "Charles de Gaulle", "Paris", "France", "Europe", 0),
        ("JFK", "John F Kennedy Intl", "New York", "United States", "North America", 0),
        ("GUA", "Bandaranaike Intl", "Mumbai", "India", "South Asia", 0),
        ("BKK", "Suvarnabhumi", "Bangkok", "Thailand", "Southeast Asia", 0),
    ],
    columns=["airport_code", "airport_name", "city", "country", "region", "is_hub"],
)

AIRCRAFT = pd.DataFrame(
    [
        # type, seats_economy, seats_business, range_km, hourly_op_cost_usd
        ("B787-8", 234, 30, 13500, 8200),
        ("B737-800", 150, 16, 5400, 4600),
        ("E190", 96, 0, 3300, 2900),
        ("A330-300", 260, 30, 11700, 8600),
        ("B737-300F", 0, 0, 4000, 4100),  # cargo, excluded from pax flights
    ],
    columns=["aircraft_type", "seats_economy", "seats_business", "range_km", "hourly_op_cost_usd"],
)
PAX_AIRCRAFT = AIRCRAFT[AIRCRAFT["seats_economy"] > 0].reset_index(drop=True)
CUSTOMER_SEGMENTS = ["Business Frequent", "Premium Leisure", "Regular Traveller", "Price Sensitive"]
CABIN_CLASSES = ["ECONOMY", "BUSINESS"]

# Approximate great-circle-ish distances (km) from NBO hub, used to derive
# flight duration, fuel burn and base fares. Non-hub routes are approximated.
DIST_FROM_NBO = {
    "MBA": 480, "KIS": 340, "EBB": 780, "DAR": 700, "KGL": 1030, "ADD": 1530,
    "JNB": 2930, "LOS": 3800, "ACC": 4600, "CAI": 3540, "DXB": 3500,
    "LHR": 6800, "AMS": 6500, "CDG": 6350, "JFK": 11800, "GUA": 4500, "BKK": 6700,
}

@dataclass
class GenConfig:
    start_date: str
    end_date: str
    n_customers: int = 40_000
    avg_daily_flights: int = 55
    out_dir: str = "data/synthetic"
    seed: int = RNG_SEED

def _make_routes(rng: np.random.Generator) -> pd.DataFrame:
    """Hub-and-spoke network out of NBO, plus a handful of regional routes"""
    hub = "NBO"
    spokes = [c for c in AIRPORTS["airport_code"] if c != hub]
    routes = []
    route_id = 1
    for spoke in spokes:
        distance = DIST_FROM_NBO[spoke]
        # domestic/regional routes fly more frequently than long-haul
        if distance < 1000:
            base_daily_freq = rng.integers(2, 5)
        elif distance < 4000:
            base_daily_freq = rng.integers(1, 3)
        else:
            base_daily_freq = 1
        for direction in [(hub, spoke), (spoke, hub)]:
            routes.append(
                {
                    "route_id": route_id,
                    "origin": direction[0],
                    "destination": direction[1],
                    "distance_km": distance,
                    "base_daily_frequency": base_daily_freq,
                }
            )
            route_id += 1
    return pd.DataFrame(routes)

def _make_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    segment_probs = [0.15, 0.20, 0.40, 0.25]
    segments = rng.choice(CUSTOMER_SEGMENTS, size=n, p=segment_probs)
    signup_dates = pd.to_datetime("2018-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 6, size=n), unit="D"
    )
    countries = rng.choice(AIRPORTS["country"].unique(), size=n)
    return pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "segment": segments,
            "home_country": countries,
            "signup_date": signup_dates.date,
            "loyalty_tier": rng.choice(
                ["Silver", "Gold", "Platinum", "Member"], size=n, p=[0.35, 0.15, 0.05, 0.45]
            ),
        }
    )

def _seasonal_factor(month: int) -> float:
    # Simple East-Africa-outbound seasonality: peaks Dec/Jul-Aug, dip Apr-May
    factors = {1: 1.05, 2: 0.95, 3: 0.90, 4: 0.80, 5: 0.82, 6: 0.95,
               7: 1.15, 8: 1.20, 9: 1.00, 10: 0.95, 11: 0.98, 12: 1.25}
    return factors[month]


# Behavioural parameters that make each customer segment fly, book and pay
# differently -- so unsupervised clustering on the resulting bookings has a
# genuine chance of recovering these same segments (a nice validation story:
# "K-Means recovered the four latent segments used to simulate demand").
SEGMENT_FLY_PROPENSITY = {
    "Business Frequent": 6.0, "Premium Leisure": 2.2,
    "Regular Traveller": 1.0, "Price Sensitive": 0.4,
}
SEGMENT_FARE_MULTIPLIER = {
    "Business Frequent": 1.25, "Premium Leisure": 1.15,
    "Regular Traveller": 1.0, "Price Sensitive": 0.82,
}
SEGMENT_LEAD_TIME_SCALE = {  # exponential-distribution scale, in days
    "Business Frequent": 4, "Premium Leisure": 28,
    "Regular Traveller": 16, "Price Sensitive": 34,
}
SEGMENT_BUSINESS_CABIN_BOOST = {  # multiplier on baseline business-cabin odds
    "Business Frequent": 3.0, "Premium Leisure": 1.4,
    "Regular Traveller": 0.6, "Price Sensitive": 0.15,
}



def _generate_flights_and_bookings(rng: np.random.Generator, cfg: GenConfig,
                                    routes: pd.DataFrame, customers: pd.DataFrame):
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    flights, bookings = [], []
    flight_id_ctr, booking_id_ctr = 1, 1
    customer_ids = customers["customer_id"].values
    customer_segments = customers["segment"].values
    n_customers = len(customer_ids)

    fly_weights = np.array([SEGMENT_FLY_PROPENSITY[s] for s in customer_segments])
    fly_probs = fly_weights / fly_weights.sum()

    TRUE_PRICE_ELASTICITY = -0.6  # ground-truth elasticity the econometric
    # model in src/ml/route_economics.py should approximately recover

    for d in dates:
        season = _seasonal_factor(d.month)
        weekend_boost = 1.1 if d.dayofweek in (4, 5) else 1.0

        # A day-to-day pricing decision independent of season/demand (think:
        # revenue-management fare adjustments, promotions, competitor
        # matching). This creates genuine price variation that is NOT
        # explained by seasonality, which is what makes it possible to
        # separately identify a price elasticity of demand from the data
        # (in real airline data this identification problem is exactly why
        # naive price-quantity correlations are usually misleading).
        price_shock_log = rng.normal(0, 0.15)
        fare_shock_mult = float(np.exp(price_shock_log))
        demand_shock_mult = float(np.exp(TRUE_PRICE_ELASTICITY * price_shock_log))

        for _, route in routes.iterrows():
            n_flights_today = max(
                0, int(round(route["base_daily_frequency"] * rng.uniform(0.7, 1.3)))
            )
            for _ in range(n_flights_today):
                aircraft = PAX_AIRCRAFT.sample(random_state=None, weights=None, n=1, replace=True).iloc[0] \
                    if False else PAX_AIRCRAFT.iloc[rng.integers(0, len(PAX_AIRCRAFT))]
                capacity_econ = int(aircraft["seats_economy"])
                capacity_bus = int(aircraft["seats_business"])
                total_capacity = capacity_econ + capacity_bus

                # --- demand / load factor model ---------------------------------
                base_load = 0.72
                is_hub_route = "NBO" in (route["origin"], route["destination"])
                hub_bonus = 0.05 if is_hub_route else -0.03
                target_load = np.clip(
                    base_load * season * weekend_boost * demand_shock_mult
                    + hub_bonus + rng.normal(0, 0.05),
                    0.35, 0.98,
                )
                passengers = int(round(total_capacity * target_load))

                # --- delay model -------------------------------------------------
                # Systematic drivers (season, hub congestion, aircraft type, weekday)
                # plus a smaller random operational shock, so the delay-prediction
                # model has learnable signal rather than being pure noise.
                month_delay_bonus = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1,
                                      7: 5, 8: 6, 9: 1, 10: 0, 11: 0, 12: 7}[d.month]
                aircraft_delay_bonus = {"E190": 5, "B737-800": 2, "B787-8": 0, "A330-300": 1}.get(
                    aircraft["aircraft_type"], 0
                )
                congestion_penalty = 7 if is_hub_route and d.dayofweek in (0, 4) else 0
                delay_shock = rng.exponential(scale=5) if rng.random() < 0.30 else 0
                dep_delay_min = max(
                    0,
                    int(round(
                        month_delay_bonus + aircraft_delay_bonus + congestion_penalty
                        + delay_shock + rng.normal(1, 3)
                    )),
                )
                is_cancelled = rng.random() < 0.012  # ~1.2% cancellation rate

                flight_duration_min = int(route["distance_km"] / 800 * 60) + 25  # incl. taxi

                fuel_cost = route["distance_km"] * 0.62 * rng.uniform(0.9, 1.15)  # usd, simplified
                crew_cost = flight_duration_min / 60 * 950
                handling_cost = 1200 if is_hub_route else 800
                # Ground-ops overhead (landing fees, gate handling, positioning
                # crew) is largely FIXED per flight regardless of distance --
                # this is what makes short, high-frequency regional routes
                # structurally thinner-margin than they look on fuel/crew cost
                # alone, mirroring real short-haul airline economics.
                ground_ops_overhead = 2200
                op_cost = (
                    aircraft["hourly_op_cost_usd"] * flight_duration_min / 60
                    + fuel_cost + crew_cost + handling_cost + ground_ops_overhead
                )

                flight_number = f"KQ{rng.integers(100, 999)}"
                flights.append(
                    {
                        "flight_id": flight_id_ctr,
                        "flight_number": flight_number,
                        "route_id": route["route_id"],
                        "origin": route["origin"],
                        "destination": route["destination"],
                        "flight_date": d.date(),
                        "aircraft_type": aircraft["aircraft_type"],
                        "seats_economy": capacity_econ,
                        "seats_business": capacity_bus,
                        "total_capacity": total_capacity,
                        "passengers": 0 if is_cancelled else passengers,
                        "scheduled_departure_hour": int(rng.integers(5, 22)),
                        "departure_delay_minutes": 0 if is_cancelled else dep_delay_min,
                        "flight_duration_minutes": flight_duration_min,
                        "is_cancelled": int(is_cancelled),
                        "operating_cost_usd": round(op_cost, 2),
                        "fuel_cost_usd": round(fuel_cost, 2),
                    }
                )

                if not is_cancelled and passengers > 0:
                    base_fare_econ = 45 + route["distance_km"] * 0.085
                    base_fare_bus = base_fare_econ * rng.uniform(2.8, 3.6)

                    # Sample which customers are on this flight, weighted by
                    # each segment's flying propensity (Business Frequent
                    # customers show up far more often than Price Sensitive).
                    cust_idx = rng.choice(n_customers, size=passengers, p=fly_probs, replace=True)
                    pax_segments = customer_segments[cust_idx]

                    business_boost = np.array(
                        [SEGMENT_BUSINESS_CABIN_BOOST[s] for s in pax_segments]
                    )
                    base_business_odds = 0.10 if capacity_bus > 0 else 0.0
                    business_odds = np.clip(base_business_odds * business_boost, 0, 0.85)
                    is_business = (rng.random(passengers) < business_odds) & (capacity_bus > 0)

                    fare_mult = np.array([SEGMENT_FARE_MULTIPLIER[s] for s in pax_segments])
                    lead_scale = np.array([SEGMENT_LEAD_TIME_SCALE[s] for s in pax_segments])

                    base_fare_arr = np.where(is_business, base_fare_bus, base_fare_econ)
                    fares = np.round(
                        base_fare_arr * season * fare_shock_mult * fare_mult
                        * rng.uniform(0.85, 1.2, size=passengers), 2
                    )
                    lead_time_days = rng.exponential(scale=lead_scale).astype(int)
                    booking_dates = d - pd.to_timedelta(lead_time_days, unit="D")
                    no_show = rng.random(size=passengers) < 0.018
                    baggage = rng.poisson(1.1, size=passengers)

                    for i in range(passengers):
                        bookings.append(
                            {
                                "booking_id": f"BK{booking_id_ctr:08d}",
                                "flight_id": flight_id_ctr,
                                "customer_id": customer_ids[cust_idx[i]],
                                "cabin_class": "BUSINESS" if is_business[i] else "ECONOMY",
                                "fare_usd": float(fares[i]),
                                "booking_date": booking_dates[i].date(),
                                "travel_date": d.date(),
                                "lead_time_days": int(lead_time_days[i]),
                                "baggage_count": int(baggage[i]),
                                "no_show": int(no_show[i]),
                            }
                        )
                        booking_id_ctr += 1

                flight_id_ctr += 1

    return pd.DataFrame(flights), pd.DataFrame(bookings)

def inject_data_quality_issues(flights: pd.DataFrame, bookings: pd.DataFrame,
                                rng: np.random.Generator):
    """
    Deliberately injects realistic messiness so the data-quality module
    has something real to detect (this mirrors what auditors actually find
    in production airline systems).
    """
    flights = flights.copy()
    bookings = bookings.copy()

    # 1. Duplicate bookings (~0.4%)
    dup_frac = 0.004
    dup_rows = bookings.sample(frac=dup_frac, random_state=1)
    bookings = pd.concat([bookings, dup_rows], ignore_index=True)

    # 2. Missing customer_id (~0.3%)
    n_missing = int(len(bookings) * 0.003)
    idx = rng.choice(bookings.index, size=n_missing, replace=False)
    bookings.loc[idx, "customer_id"] = None

    # 3. Negative / impossible fares (~0.2%)
    idx = rng.choice(bookings.index, size=int(len(bookings) * 0.002), replace=False)
    bookings.loc[idx, "fare_usd"] = -bookings.loc[idx, "fare_usd"]

    # 4. Invalid airport codes in a few flights (~0.1%)
    idx = rng.choice(flights.index, size=max(1, int(len(flights) * 0.001)), replace=False)
    flights.loc[idx, "destination"] = "ZZZ"

    # 5. Passengers exceeding capacity (~0.15%) - data entry error simulation
    idx = rng.choice(flights.index, size=max(1, int(len(flights) * 0.0015)), replace=False)
    flights.loc[idx, "passengers"] = flights.loc[idx, "total_capacity"] + rng.integers(5, 30, size=len(idx))

    return flights, bookings

def generate(cfg: GenConfig) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)

    airports = AIRPORTS.copy()
    aircraft = AIRCRAFT.copy()
    routes = _make_routes(rng)
    customers = _make_customers(rng, cfg.n_customers)
    flights, bookings = _generate_flights_and_bookings(rng, cfg, routes, customers)
    flights, bookings = inject_data_quality_issues(flights, bookings, rng)

    return {
        "dim_airports": airports,
        "dim_aircraft": aircraft,
        "dim_routes": routes,
        "dim_customers": customers,
        "fact_flights": flights,
        "fact_bookings": bookings,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic airline dataset")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--customers", type=int, default=40_000)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    cfg = GenConfig(
        start_date=args.start, end_date=args.end,
        n_customers=args.customers, out_dir=args.out, seed=args.seed,
    )
    tables = generate(cfg)

    out_path = Path(cfg.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_path / f"{name}.csv", index=False)
        print(f"  wrote {name:20s} {len(df):>10,} rows -> {out_path / f'{name}.csv'}")

    print("\nSynthetic data generation complete.")


if __name__ == "__main__":
    main()
