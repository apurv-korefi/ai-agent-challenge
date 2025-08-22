import importlib
import pathlib
import pytest

DATA_DIR = pathlib.Path("data")

@pytest.mark.parametrize("target", [p.name for p in DATA_DIR.iterdir() if p.is_dir()])
def test_contract(target):
    """
    Ensures every custom parser implements parse_pdf and produces a CSV
    with at least one row.
    """
    parser_module = importlib.import_module(f"custom_parsers.{target}_parser")

    # Must have a parse_pdf function
    assert hasattr(parser_module, "parse_pdf"), f"{target}_parser.py missing parse_pdf()"

    pdf_path = DATA_DIR / target / f"{target}_sample.pdf"
    csv_path = DATA_DIR / target / f"{target}_sample.csv"

    # Run the parser
    rows = parser_module.parse_pdf(pdf_path)

    # Must return a list of dicts
    assert isinstance(rows, list), "parse_pdf() must return a list"
    assert all(isinstance(r, dict) for r in rows), "Each row must be a dict"
    assert len(rows) > 0, "parse_pdf() must extract at least 1 row"

    # Write to CSV
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Validate CSV written
    assert csv_path.exists(), f"CSV not written: {csv_path}"
    assert csv_path.stat().st_size > 0, "CSV is empty"

