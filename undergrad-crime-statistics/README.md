# Undergraduate Crime Statistics Project: Phoenix Monthly Crime Trends

This project documents an early exploratory analysis of Phoenix crime incident records, which later informed a broader crime-statistics and forecasting interest. It is kept separate from the later graduate capstone.

## What the original notebook did

The submitted notebook loaded a CSV with incident date, UCR crime category, and other fields; removed several columns; dropped rows without an occurrence date; created month-year labels; filtered records before May 2020; counted incidents by month; and plotted monthly totals and year-over-year monthly comparisons.

The original notebook also excluded the categories `RAPE` and `AGGRAVATED ASSAULT`. The independent rewrite does **not** silently exclude these categories. If a researcher wants a restricted analysis, they must explicitly configure the categories and explain the rationale.

## Clean implementation

`crime_trends.py` provides a reproducible pipeline that:
- validates expected columns;
- parses occurrence timestamps with invalid dates reported;
- filters to a configurable start date (default 2020-05-01);
- counts records by calendar month;
- retains all crime categories by default;
- reports missing/invalid date counts;
- exports a tidy monthly-count CSV and a line chart.

## Run

```bash
python -m pip install -r requirements.txt
python crime_trends.py --input path/to/crimestat.csv --output-dir outputs
python -m pytest -q
```

The source dataset is not included. Supply a permitted local copy of the CSV. Output files are generated locally and are ignored by Git. The supplied source files were checked during review: both contained 444,802 rows and produced the same monthly aggregates.

## Interpretation and limitations

These are counts of records in the supplied dataset, not necessarily crime rates. They are not normalized by population, reporting rates, or changes in data collection. Date parsing and incident-date definitions affect results. Counts alone do not establish causality, and trends should not be interpreted as changes in individual risk. The notebook's hand-entered plotting arrays are not treated as authoritative; results are computed from the source CSV.
