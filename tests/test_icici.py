
import pandas as pd
import custom_parsers.icici_parser as parser

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Description'] = df['Description'].astype(str).str.strip()
    for c in ['Debit Amt','Credit Amt','Balance']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
    return df.reset_index(drop=True)

def test_parse_equals_expected():
    parsed = parser.parse(r"data\icici\icici_sample.pdf")
    expected = pd.read_csv(r"data\icici\icici_sample.csv")
    assert normalize(parsed).equals(normalize(expected)), "Parsed output does not match expected"

