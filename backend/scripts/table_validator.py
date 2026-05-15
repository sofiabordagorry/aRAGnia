from pathlib import Path
import pandas as pd
from pathlib import Path
from institutional_graphrag.ingest.table_extractors import clean_table
import csv


BASE_DIR = Path(__file__).resolve().parents[2]

tables_dir = BASE_DIR / "data" / "tables"
input_csv = tables_dir / "equipos_i+d_2012_2018_pinco.csv"
clean_csv_path = tables_dir / "equipos_i+d_2012_2018_pinco_clean.csv"


def validate_csv(csv_path: str | Path) -> bool:
    csv_path = Path(csv_path)
    max_errors_to_show = 5
    print("\n" + "=" * 60)
    print(f"Validating CSV: {csv_path.name}")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Try loading with pandas
    # --------------------------------------------------

    try:
        df = pd.read_csv(csv_path, dtype=str)

        print("\nPandas successfully loaded the CSV")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")
        print("\nRandom sample rows:")

        sample_size = min(5, len(df))

        sample_df = df.sample(sample_size)

        columns_to_show = []

        if "nombres" in df.columns:
            columns_to_show.append("nombres")

        if "titulo" in df.columns:
            columns_to_show.append("titulo")

        if columns_to_show:
            print(sample_df[columns_to_show])
        else:
            print("Columns 'nombres' or 'titulo' not found")

    except Exception as e:

        print("\nERROR: Pandas could not read the CSV")
        print(type(e).__name__, "-", e)

        return False

    # --------------------------------------------------
    # 2. Validate row structure manually
    # --------------------------------------------------

    invalid_rows = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:

        reader = csv.reader(f)

        header = next(reader)
        expected_columns = len(header)

        print(f"\nExpected columns: {expected_columns}")

        for line_number, row in enumerate(reader, start=2):

            if len(row) != expected_columns:

                invalid_rows += 1

                if invalid_rows <= max_errors_to_show:

                    print("\n" + "-" * 60)
                    print(f"Invalid row at line {line_number}")
                    print(f"Columns found: {len(row)}")
                    print(f"Expected: {expected_columns}")
        if invalid_rows > max_errors_to_show:
            print(f"\n... and {invalid_rows - max_errors_to_show} more invalid rows")
    # --------------------------------------------------
    # 3. Final result
    # --------------------------------------------------

    print("\n" + "=" * 60)

    if invalid_rows == 0:
        print("CSV structure is valid")
    else:
        print(f"Invalid rows found: {invalid_rows}")

    return invalid_rows == 0

if __name__ == "__main__":

    # Validate original CSV
    original_ok = validate_csv(input_csv)

    # Clean CSV
    print("\n" + "=" * 60)
    print("Cleaning CSV...")
    print("=" * 60)

    clean_table(input_csv, clean_csv_path)

    # Validate cleaned CSV
    cleaned_ok = validate_csv(clean_csv_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Original CSV valid: {original_ok}")
    print(f"Cleaned CSV valid: {cleaned_ok}")