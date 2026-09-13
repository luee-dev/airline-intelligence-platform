-- ============================================================================
-- Cleaned analytical views ("silver layer").
-- Raw fact tables are never queried directly by analytics/BI/ML - everything
-- reads from these views instead, so records that fail data-quality rules
-- (invalid airport codes, impossible fares, capacity violations) can't
-- silently distort KPIs. This mirrors a bronze -> silver -> gold pattern.
-- ============================================================================

DROP VIEW IF EXISTS fact_flights_clean;
CREATE VIEW fact_flights_clean AS
SELECT f.*
FROM fact_flights f
WHERE f.origin      IN (SELECT airport_code FROM dim_airport)
  AND f.destination IN (SELECT airport_code FROM dim_airport)
  AND f.passengers <= f.total_capacity
  AND f.passengers >= 0;

DROP VIEW IF EXISTS fact_bookings_clean;
CREATE VIEW fact_bookings_clean AS
SELECT b.*
FROM fact_bookings b
WHERE b.fare_usd > 0
  AND b.customer_id IS NOT NULL
  AND b.booking_id NOT IN (
        -- drop the second occurrence of any duplicated booking_id
        SELECT booking_id FROM (
            SELECT booking_id, ROW_NUMBER() OVER (PARTITION BY booking_id ORDER BY rowid) AS rn
            FROM fact_bookings
        ) WHERE rn > 1
  );
