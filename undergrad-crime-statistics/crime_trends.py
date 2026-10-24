"""Reproducible monthly incident-count analysis for a local Phoenix crime CSV."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REQUIRED_COLUMNS = {"OCCURRED ON", "UCR CRIME CATEGORY"}

def monthly_counts(frame: pd.DataFrame, start_date: str = "2020-05-01") -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = frame.loc[:, ["OCCURRED ON", "UCR CRIME CATEGORY"]].copy()
    original_rows = len(data)
    data["DATE"] = pd.to_datetime(data["OCCURRED ON"], errors="coerce")
    invalid_dates = int(data["DATE"].isna().sum())
    data = data.dropna(subset=["DATE"])
    data = data.loc[data["DATE"] >= pd.Timestamp(start_date)].copy()
    data["UCR CRIME CATEGORY"] = data["UCR CRIME CATEGORY"].fillna("Unknown").astype(str).str.strip()
    data["MONTH"] = data["DATE"].dt.to_period("M").dt.to_timestamp()
    counts = (data.groupby("MONTH", as_index=False)
              .size().rename(columns={"size": "INCIDENT_COUNT"})
              .sort_values("MONTH").reset_index(drop=True))
    summary = {
        "input_rows": original_rows,
        "invalid_or_missing_dates": invalid_dates,
        "rows_in_period": int(len(data)),
        "start_date": str(pd.Timestamp(start_date).date()),
        "months_counted": int(len(counts)),
        "categories_excluded": [],
    }
    return counts, summary

def create_outputs(input_path: str | Path, output_dir: str | Path, start_date: str = "2020-05-01") -> dict:
    input_path, output_dir = Path(input_path), Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    frame = pd.read_csv(input_path, low_memory=False)
    counts, summary = monthly_counts(frame, start_date=start_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "monthly_crime_counts.csv"
    summary_path = output_dir / "data_quality_summary.csv"
    chart_path = output_dir / "monthly_crime_trend.png"
    counts.to_csv(csv_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(counts["MONTH"], counts["INCIDENT_COUNT"], marker="o", linewidth=1.5)
    ax.set_title("Phoenix Reported Crime Incidents by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Incident records")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return {"monthly_counts": str(csv_path), "quality_summary": str(summary_path), "chart": str(chart_path), **summary}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a local crime CSV")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--start-date", default="2020-05-01", help="Inclusive ISO date, YYYY-MM-DD")
    args = parser.parse_args()
    print(create_outputs(args.input, args.output_dir, args.start_date))

if __name__ == "__main__":
    main()
