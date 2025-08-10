import pandas as pd
import sys
from pathlib import Path

def test_icici_parser():
    base = Path(__file__).parent.parent
    sys.path.insert(0, str(base))
    from custom_parsers.icici_parser import parse

    pdf_path = base / "data/icici/icici_sample.pdf"
    csv_path = base / "data/icici/result.csv"

    df_actual = parse(str(pdf_path))
    df_expected = pd.read_csv(csv_path)

    df_actual = df_actual[df_expected.columns]
    assert df_actual.equals(df_expected)

