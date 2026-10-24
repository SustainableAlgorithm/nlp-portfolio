# DTSC 660 — Data and Database Management with SQL

This folder organizes independent portfolio examples inspired by seven topic areas from the course. Original graded submissions and course-provided datasets are not copied here. The scripts below are clean, documented examples of transferable SQL and data-management skills.

## Project areas

1. **Database setup and schema design** — PostgreSQL database creation and relational table definitions.
2. **Exploratory SQL** — filtering, sorting, arithmetic expressions, and selecting useful columns.
3. **Aggregations and analytical queries** — daily Bitcoin metrics and customer spending by year/category.
4. **Banking database design** — relational schema and primary/foreign-key relationships.
5. **Joins and set operations** — joining depositor/borrower-style tables and comparing query results.
6. **Data cleaning and quality checks** — profiling a local salary CSV, standardizing text, identifying missing/invalid values, and producing a clean output table.
7. **Python-to-SQL workflow** — connecting to PostgreSQL, executing a query, loading results into pandas, and saving a CSV.

## Run and review

Examples use PostgreSQL syntax unless noted. Review each file before executing; database creation and table setup scripts modify the target database. The data-cleaning script accepts a local CSV and does not bundle course data. The Python connection example requires a PostgreSQL DSN supplied through an environment variable.

## Privacy and integrity

The original assignments, course datasets, screenshots, and submission artifacts are not included. These examples are independently written and are intended to demonstrate SQL reasoning and data engineering basics, not to reproduce graded submissions. No performance or data-quality claims are made without running against a documented dataset.
