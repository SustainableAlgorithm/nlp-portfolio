-- Synthetic customer/account/loan relations.
WITH depositor (customer_id, account_number) AS (
    VALUES (1, 'A-100'), (2, 'A-200'), (3, 'A-300')
),
borrower (customer_id, loan_number) AS (
    VALUES (1, 'L-10'), (3, 'L-30'), (4, 'L-40')
)
SELECT d.customer_id, d.account_number, b.loan_number
FROM depositor AS d
JOIN borrower AS b USING (customer_id)
ORDER BY d.customer_id;

-- Customers with an account, whether or not they have a loan.
WITH depositor (customer_id, account_number) AS (
    VALUES (1, 'A-100'), (2, 'A-200'), (3, 'A-300')
),
borrower (customer_id, loan_number) AS (
    VALUES (1, 'L-10'), (3, 'L-30'), (4, 'L-40')
)
SELECT d.customer_id, d.account_number, b.loan_number
FROM depositor AS d
LEFT JOIN borrower AS b USING (customer_id)
ORDER BY d.customer_id;
