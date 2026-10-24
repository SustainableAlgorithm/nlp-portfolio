-- SQL exploration: synthetic daily transaction metrics (PostgreSQL)
WITH daily_metrics (metric_date, code_size_kb, transaction_count, price_usd, topic_posts) AS (
    VALUES
        (DATE '2025-01-01', 1200, 180000, 42000.00, 15000),
        (DATE '2025-01-02', 1205, 192000, 43500.00, 17200),
        (DATE '2025-01-03', 1210, 175000, 42800.00, 16100),
        (DATE '2025-01-04', 1212, 201000, 44100.00, 18900)
)
SELECT metric_date, price_usd,
       code_size_kb::numeric / NULLIF(transaction_count, 0) AS code_size_per_transaction,
       topic_posts
FROM daily_metrics
WHERE transaction_count > 0
ORDER BY metric_date;
