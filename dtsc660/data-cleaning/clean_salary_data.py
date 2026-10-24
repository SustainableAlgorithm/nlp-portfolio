"""Profile and lightly clean a local salary CSV without bundling course data."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def clean(input_path: str | Path, output_path: str | Path) -> dict:
    frame = pd.read_csv(input_path)
    original_rows = len(frame)
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    missing_before = frame.isna().sum().to_dict()
    duplicate_rows = int(frame.duplicated().sum())
    for col in frame.select_dtypes(include=["object", "string"]).columns:
        frame[col] = frame[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    frame = frame.drop_duplicates()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return {"input_rows": original_rows, "output_rows": len(frame), "duplicate_rows_removed": duplicate_rows,
            "missing_values_before": missing_before, "output_path": str(output_path)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True);parser.add_argument("--output",default="cleaned_data.csv")
    args=parser.parse_args()
    print(clean(args.input,args.output))
if __name__=="__main__":main()
