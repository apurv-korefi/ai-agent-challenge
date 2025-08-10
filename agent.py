import argparse
import importlib.util
import sys
from pathlib import Path
import pandas as pd
import pdfplumber

MAX_ATTEMPTS = 3


def extract_table_from_pdf(pdf_path):
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not any(row):
                        continue
                    rows.append(row)
    return rows


def plan_parser_code(columns):
    return f'''import pandas as pd
import pdfplumber

def parse(pdf_path):
    data_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if not any(row):
                        continue
                    if row[0] == "{columns[0]}" or row[0] is None:
                        continue
                    # Strip spaces
                    cleaned = [(cell.strip() if isinstance(cell, str) else cell) for cell in row]
                    # Ensure length matches columns
                    if len(cleaned) > {len(columns)}:
                        cleaned = cleaned[:{len(columns)}]
                    elif len(cleaned) < {len(columns)}:
                        cleaned += [None] * ({len(columns)} - len(cleaned))
                    data_rows.append(cleaned)
    df = pd.DataFrame(data_rows, columns={columns})
    # Convert numeric columns
    for col in ["Debit Amt", "Credit Amt", "Balance"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
'''


def write_parser_file(code_str, parser_path):
    parser_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parser_path, "w", encoding="utf-8") as f:
        f.write(code_str)


def run_parser_and_compare(parser_path, pdf_path, csv_path):
    spec = importlib.util.spec_from_file_location("custom_parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)

    df_parsed = parser_module.parse(str(pdf_path))
    df_expected = pd.read_csv(csv_path)

    df_parsed = df_parsed[df_expected.columns]

    return df_parsed.equals(df_expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank target key, e.g., icici")
    args = parser.parse_args()

    target = args.target
    base_folder = Path("data") / target

    pdf_path = base_folder / f"{target}_sample.pdf"
    csv_path = base_folder / "result.csv"
    parser_file_path = Path("custom_parsers") / f"{target}_parser.py"

    if not pdf_path.exists() or not csv_path.exists():
        print(f"Error: Files not found for target '{target}'.")
        sys.exit(1)

    df_expected = pd.read_csv(csv_path)
    columns = df_expected.columns.tolist()

    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[Attempt {attempt}] Generating parser for '{target}'...")

        code_str = plan_parser_code(columns)
        write_parser_file(code_str, parser_file_path)

        success = run_parser_and_compare(parser_file_path, pdf_path, csv_path)

        if success:
            print(f"[SUCCESS] Parser generated and verified for '{target}'.")
            sys.exit(0)
        else:
            print(f"[FAIL] Output mismatch. Retrying...")

    print(f"[ERROR] Failed to generate a correct parser in {MAX_ATTEMPTS} attempts.")
    sys.exit(1)


if __name__ == "__main__":
    main()


