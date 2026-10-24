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
