import pandas as pd
import pytest
from crime_trends import monthly_counts

def test_counts_monthly_and_keeps_all_categories():
    frame = pd.DataFrame({
        "OCCURRED ON": ["05/01/2020 12:00", "05/21/2020 09:00", "06/01/2020 00:00", "04/30/2020 23:59"],
        "UCR CRIME CATEGORY": ["RAPE", "AGGRAVATED ASSAULT", "LARCENY-THEFT", "BURGLARY"],
    })
    counts, summary = monthly_counts(frame)
    assert counts["INCIDENT_COUNT"].tolist() == [2, 1]
    assert summary["rows_in_period"] == 3
    assert summary["categories_excluded"] == []

def test_invalid_dates_are_reported_and_dropped():
    frame = pd.DataFrame({
        "OCCURRED ON": ["not a date", None, "2021-01-01"],
        "UCR CRIME CATEGORY": ["THEFT", "BURGLARY", "THEFT"],
    })
    counts, summary = monthly_counts(frame)
    assert summary["invalid_or_missing_dates"] == 2
    assert counts["INCIDENT_COUNT"].tolist() == [1]

def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing required columns"):
        monthly_counts(pd.DataFrame({"DATE": ["2021-01-01"]}))

def test_empty_period_returns_empty_monthly_table():
    frame = pd.DataFrame({"OCCURRED ON": ["2019-01-01"], "UCR CRIME CATEGORY": ["THEFT"]})
    counts, _ = monthly_counts(frame)
    assert counts.empty
