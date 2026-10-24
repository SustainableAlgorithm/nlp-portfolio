-- Aggregate synthetic sales by category for one year.
WITH sales (sale_year, category, revenue) AS (
    VALUES
        (2024, 'Books', 120.00::numeric),
        (2024, 'Home', 240.00::numeric),
        (2024, 'Books', 80.00::numeric),
        (2025, 'Home', 310.00::numeric),
        (2025, 'Garden', 95.00::numeric)
)
SELECT category, SUM(revenue) AS total_revenue, AVG(revenue) AS average_sale
FROM sales
WHERE sale_year = 2024
GROUP BY category
ORDER BY total_revenue DESC;
