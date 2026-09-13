-- ============================================================================
-- Airline Intelligence Platform — Analytical Warehouse Schema
-- Dimensional (star) model: one fact table at flight-grain, one at
-- booking-grain, surrounded by conformed dimensions.
-- Target: PostgreSQL 14+
-- ============================================================================

DROP SCHEMA IF EXISTS airline CASCADE;
CREATE SCHEMA airline;
SET search_path TO airline;

-- ---------------------------------------------------------------------------
-- DIMENSIONS
-- ---------------------------------------------------------------------------

CREATE TABLE dim_date (
    date_key        DATE PRIMARY KEY,
    year            SMALLINT NOT NULL,
    quarter         SMALLINT NOT NULL,
    month           SMALLINT NOT NULL,
    month_name      VARCHAR(10) NOT NULL,
    day_of_week     SMALLINT NOT NULL,
    day_name        VARCHAR(10) NOT NULL,
    is_weekend      BOOLEAN NOT NULL,
    week_of_year    SMALLINT NOT NULL
);

CREATE TABLE dim_airport (
    airport_code    CHAR(3) PRIMARY KEY,
    airport_name    VARCHAR(100) NOT NULL,
    city            VARCHAR(100) NOT NULL,
    country         VARCHAR(100) NOT NULL,
    region          VARCHAR(50)  NOT NULL,
    is_hub          BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE dim_aircraft (
    aircraft_type       VARCHAR(20) PRIMARY KEY,
    seats_economy       INT NOT NULL,
    seats_business      INT NOT NULL,
    range_km            INT NOT NULL,
    hourly_op_cost_usd  NUMERIC(10,2) NOT NULL
);

CREATE TABLE dim_route (
    route_id            INT PRIMARY KEY,
    origin              CHAR(3) NOT NULL REFERENCES dim_airport(airport_code),
    destination         CHAR(3) NOT NULL REFERENCES dim_airport(airport_code),
    distance_km         INT NOT NULL,
    base_daily_frequency INT NOT NULL
);

CREATE TABLE dim_customer (
    customer_id     VARCHAR(15) PRIMARY KEY,
    segment         VARCHAR(30) NOT NULL,
    home_country    VARCHAR(100),
    signup_date     DATE,
    loyalty_tier    VARCHAR(20)
);

-- ---------------------------------------------------------------------------
-- FACTS
-- ---------------------------------------------------------------------------

CREATE TABLE fact_flights (
    flight_id                  BIGINT PRIMARY KEY,
    flight_number              VARCHAR(10) NOT NULL,
    route_id                   INT NOT NULL REFERENCES dim_route(route_id),
    origin                     CHAR(3) NOT NULL,
    destination                CHAR(3) NOT NULL,
    flight_date                DATE NOT NULL,
    aircraft_type              VARCHAR(20) NOT NULL,
    seats_economy              INT NOT NULL,
    seats_business             INT NOT NULL,
    total_capacity             INT NOT NULL,
    passengers                 INT NOT NULL,
    scheduled_departure_hour   SMALLINT NOT NULL,
    departure_delay_minutes    INT NOT NULL,
    flight_duration_minutes    INT NOT NULL,
    is_cancelled               BOOLEAN NOT NULL,
    operating_cost_usd         NUMERIC(12,2) NOT NULL,
    fuel_cost_usd               NUMERIC(12,2) NOT NULL
);
CREATE INDEX idx_flights_date  ON fact_flights(flight_date);
CREATE INDEX idx_flights_route ON fact_flights(route_id);

CREATE TABLE fact_bookings (
    booking_id      VARCHAR(15) PRIMARY KEY,
    flight_id       BIGINT NOT NULL REFERENCES fact_flights(flight_id),
    customer_id     VARCHAR(15) REFERENCES dim_customer(customer_id),
    cabin_class     VARCHAR(10) NOT NULL,
    fare_usd        NUMERIC(10,2) NOT NULL,
    booking_date    DATE NOT NULL,
    travel_date     DATE NOT NULL,
    lead_time_days  INT NOT NULL,
    baggage_count   INT NOT NULL,
    no_show         BOOLEAN NOT NULL
);
CREATE INDEX idx_bookings_flight   ON fact_bookings(flight_id);
CREATE INDEX idx_bookings_customer ON fact_bookings(customer_id);
CREATE INDEX idx_bookings_traveldt ON fact_bookings(travel_date);

-- ---------------------------------------------------------------------------
-- DATA QUALITY / OPERATIONAL METADATA
-- ---------------------------------------------------------------------------

CREATE TABLE data_quality_issues (
    issue_id        SERIAL PRIMARY KEY,
    dataset         VARCHAR(50) NOT NULL,
    record_id       VARCHAR(50),
    issue_type      VARCHAR(50) NOT NULL,
    severity        VARCHAR(10) NOT NULL CHECK (severity IN ('LOW','MEDIUM','HIGH','CRITICAL')),
    detected_at     TIMESTAMP NOT NULL DEFAULT now(),
    status          VARCHAR(15) NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','RESOLVED','IGNORED')),
    resolved_at     TIMESTAMP
);

CREATE TABLE data_quality_score_history (
    run_id          SERIAL PRIMARY KEY,
    run_timestamp   TIMESTAMP NOT NULL DEFAULT now(),
    dataset         VARCHAR(50) NOT NULL,
    total_records   BIGINT NOT NULL,
    valid_records   BIGINT NOT NULL,
    quality_score   NUMERIC(5,2) NOT NULL
);
