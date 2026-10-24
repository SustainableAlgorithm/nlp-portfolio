"""Run a read-only PostgreSQL query and save results as CSV."""
from __future__ import annotations
import os
import argparse
import pandas as pd
from sqlalchemy import create_engine, text

def export_query(database_url: str, query: str, output_path: str) -> int:
    if not query.lstrip().lower().startswith(("select", "with")):
        raise ValueError("Only SELECT or WITH queries are permitted by this example")
    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            with connection.begin():
                connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                frame = pd.read_sql_query(text(query), connection)
        frame.to_csv(output_path, index=False)
        return len(frame)
    finally:
        engine.dispose()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query",required=True);parser.add_argument("--output",default="query_results.csv")
    args=parser.parse_args()
    url=os.environ.get("DATABASE_URL")
    if not url: raise SystemExit("Set DATABASE_URL in your environment; do not hard-code credentials.")
    print(f"Exported {export_query(url,args.query,args.output)} rows to {args.output}")
if __name__=="__main__":main()
