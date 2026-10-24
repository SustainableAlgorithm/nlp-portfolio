# Code Review: Undergraduate Crime Statistics Project

## Source-derived summary

The original notebook uses pandas and Matplotlib to load Phoenix crime records, select occurrence date and UCR crime category, remove rows without dates, create month-year labels, filter to dates from May 2020 onward, count incidents by month, and plot monthly trends. It contains hard-coded chart arrays in addition to data-derived processing.

## Improvements in the independent rewrite

- Parses dates with explicit invalid-date handling.
- Groups by calendar month chronologically rather than sorting month labels by frequency.
- Calculates counts directly from the input dataset instead of relying on hand-entered chart arrays.
- Keeps all crime categories by default; the original notebook removed two categories without documenting a research justification.
- Writes a small data-quality summary and keeps generated artifacts out of version control.
- Includes tests for month aggregation, invalid dates, missing columns, and empty periods.

## Validation status

Tests cover deterministic synthetic examples. They do not establish the correctness or completeness of the full source dataset. No dataset-specific results are claimed here because the original CSVs are not included in this repository.

## Analytical limitations

Monthly incident counts are not population-adjusted crime rates. The data may reflect reporting practices, data-quality issues, or policy changes. This descriptive analysis does not establish causal explanations and should not be used to infer individual-level risk.


## Validation against the supplied CSV files (2026-10-24)

Both supplied CSVs, `crimestat.csv` and `crimestat_1.csv`, have the same 444,802 rows, the same eight columns, and produce the same monthly aggregates in this check.

- 419 rows have a missing or unparseable `OCCURRED ON` value.
- 148,326 records have a valid occurrence date on or after 2020-05-01, covering 28 months from May 2020 through August 2022.
- Of those in-period records, 17,819 are in the two categories excluded by the original notebook (`RAPE` and `AGGRAVATED ASSAULT`). Retaining the other categories gives 130,507 records.
- The notebook's hand-entered monthly series matches the CSV-derived counts after those two exclusions for 27 of 28 months. April 2022 differs by one record: the notebook says 4,715, while the CSV-derived count is 4,714. This should be treated as a likely manual-entry discrepancy unless the original analysis used a different data snapshot or rule.

These counts were computed directly from the uploaded CSVs. They are incident-record counts, not population-adjusted crime rates. No raw CSV has been committed to the repository.
