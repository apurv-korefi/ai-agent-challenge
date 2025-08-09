from pathlib import Path
import textwrap

def generate_test_file(target: str, pdf_path: Path, csv_path: Path):
    test_code = textwrap.dedent(f"""
    import pandas as pd
    import custom_parsers.{target}_parser as parser

    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['Date'] = df['Date'].astype(str).str.strip()
        df['Description'] = df['Description'].astype(str).str.strip()
        for c in ['Debit Amt','Credit Amt','Balance']:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
        return df.reset_index(drop=True)

    def test_parse_equals_expected():
        parsed = parser.parse(r\"{pdf_path}\")
        expected = pd.read_csv(r\"{csv_path}\")
        assert normalize(parsed).equals(normalize(expected)), "Parsed output does not match expected"
    """)
    test_path = Path("tests") / f"test_{target}.py"
    test_path.write_text(test_code, encoding="utf-8")
    print(f"[INFO] Test file created: {test_path}")
    
