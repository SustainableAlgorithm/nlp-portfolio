-- Illustrative relational schema; execute only in a development database.
CREATE TABLE IF NOT EXISTS customer (
    customer_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    full_name TEXT NOT NULL,
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account (
    account_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customer(customer_id),
    account_type TEXT NOT NULL CHECK (account_type IN ('checking', 'savings')),
    opened_on DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS transaction_record (
    transaction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES account(account_id),
    amount NUMERIC(12, 2) NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (amount <> 0)
);

CREATE INDEX IF NOT EXISTS idx_account_customer_id ON account(customer_id);
CREATE INDEX IF NOT EXISTS idx_transaction_account_time ON transaction_record(account_id, occurred_at);
