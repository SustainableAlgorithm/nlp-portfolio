from pathlib import Path
import importlib.util
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

cleaner = load_module("clean_salary_data", "data-cleaning/clean_salary_data.py")
db_export = load_module("query_to_dataframe", "python-postgres/query_to_dataframe.py")


def test_clean_salary_csv_standardizes_and_removes_duplicates(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "clean.csv"
    pd.DataFrame({"Employee Name": [" Alice ", " Alice ", "Bob"], "Salary": [100, 100, None]}).to_csv(source, index=False)
    report = cleaner.clean(source, output)
    result = pd.read_csv(output)
    assert report["input_rows"] == 3
    assert report["output_rows"] == 2
    assert report["duplicate_rows_removed"] == 1
    assert list(result.columns) == ["employee_name", "salary"]
    assert result["employee_name"].tolist() == ["Alice", "Bob"]


def test_cleaner_reports_missing_values(tmp_path):
    source = tmp_path / "input.csv"
    output = tmp_path / "clean.csv"
    pd.DataFrame({"Name": ["Alice", None]}).to_csv(source, index=False)
    report = cleaner.clean(source, output)
    assert report["missing_values_before"]["Name"] == 1


@pytest.mark.parametrize("query", ["DELETE FROM users", "DROP TABLE users", "UPDATE users SET x=1", "INSERT INTO users VALUES (1)"])
def test_database_export_rejects_non_read_queries(query):
    with pytest.raises(ValueError, match="Only SELECT or WITH"):
        db_export.export_query("postgresql://unused", query, "unused.csv")


def test_sql_examples_have_no_course_data_or_credentials():
    paths = [
        ROOT / "sql-exploration/queries.sql",
        ROOT / "aggregation-analysis/queries.sql",
        ROOT / "joins-and-set-operations/queries.sql",
        ROOT / "database-design/schema.sql",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert text.strip()
        assert "password=" not in text
        assert "api_key" not in text
