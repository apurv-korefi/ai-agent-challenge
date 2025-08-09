import re
import pandas as pd
import pdfplumber

# -------- Helper functions --------
def find_header_index(lines):
    for i, L in enumerate(lines):
        if 'Date' in L and 'Description' in L and 'Balance' in L:
            return i + 1
    return 0

def format_row(parts):
    date_token = parts[0]
    balance_token = parts[-1]
    credit_token = parts[-2] if len(parts) >= 3 else ''
    debit_token = parts[-3] if len(parts) >= 4 else ''
    description = ' '.join(parts[1:-3]) if len(parts) > 4 else (parts[1] if len(parts) >= 2 else '')
    return {
        'Date': date_token.strip(),
        'Description': description.strip(),
        'Debit Amt': debit_token.replace(',','').strip(),
        'Credit Amt': credit_token.replace(',','').strip(),
        'Balance': balance_token.replace(',','').strip()
    }

def normalize_numeric(df):
    for c in ['Debit Amt','Credit Amt','Balance']:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
    return df

# -------- Upgraded Strategies --------
def parse_line_split(pdf_path: str) -> pd.DataFrame:
    """Parse using line split by 2+ spaces, with table detection first."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # --- First try table extraction ---
            table = page.extract_table()
            if table and len(table) > 1:
                header = [h.strip() for h in table[0]]
                for tr in table[1:]:
                    if not any(tr):
                        continue
                    d = dict(zip(header, tr))
                    rows.append({
                        'Date': str(d.get('Date') or tr[0]).strip(),
                        'Description': str(d.get('Description') or '').strip(),
                        'Debit Amt': str(d.get('Debit Amt') or '').replace(',','').strip(),
                        'Credit Amt': str(d.get('Credit Amt') or '').replace(',','').strip(),
                        'Balance': str(d.get('Balance') or tr[-1]).replace(',','').strip()
                    })
                continue  # move to next page if table worked

            # --- Fallback to line-split text ---
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            start_idx = find_header_index(lines)
            for line in lines[start_idx:]:
                if not line.strip():
                    continue
                parts = re.split(r'\s{2,}', line.strip())
                if len(parts) < 4:
                    continue
                rows.append(format_row(parts))

    df = pd.DataFrame(rows, columns=['Date','Description','Debit Amt','Credit Amt','Balance'])
    return normalize_numeric(df)


def parse_regex(pdf_path: str) -> pd.DataFrame:
    """Parse using regex pattern, with table detection first."""
    rows = []
    pattern = re.compile(
        r'^(?P<Date>\d{2}-\d{2}-\d{4})\s+(?P<Desc>.*?)\s+'
        r'(?P<Debit>[\d,]+\.\d{1,2}|)\s+(?P<Credit>[\d,]+\.\d{1,2}|)\s+'
        r'(?P<Balance>-?[\d,]+\.\d{1,2})\s*$'
    )
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # --- First try table extraction ---
            table = page.extract_table()
            if table and len(table) > 1:
                header = [h.strip() for h in table[0]]
                for tr in table[1:]:
                    if not any(tr):
                        continue
                    d = dict(zip(header, tr))
                    rows.append({
                        'Date': str(d.get('Date') or tr[0]).strip(),
                        'Description': str(d.get('Description') or '').strip(),
                        'Debit Amt': str(d.get('Debit Amt') or '').replace(',','').strip(),
                        'Credit Amt': str(d.get('Credit Amt') or '').replace(',','').strip(),
                        'Balance': str(d.get('Balance') or tr[-1]).replace(',','').strip()
                    })
                continue

            # --- Fallback to regex text parsing ---
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            start_idx = find_header_index(lines)
            for line in lines[start_idx:]:
                m = pattern.match(line.strip())
                if m:
                    rows.append({
                        'Date': m.group('Date').strip(),
                        'Description': m.group('Desc').strip(),
                        'Debit Amt': m.group('Debit').replace(',','').strip(),
                        'Credit Amt': m.group('Credit').replace(',','').strip(),
                        'Balance': m.group('Balance').replace(',','').strip()
                    })

    df = pd.DataFrame(rows, columns=['Date','Description','Debit Amt','Credit Amt','Balance'])
    return normalize_numeric(df)


def parse_table_extract(pdf_path: str) -> pd.DataFrame:
    """Pure table extraction."""
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table:
                continue
            header = [h.strip() for h in table[0]]
            for tr in table[1:]:
                if not any(tr):
                    continue
                d = dict(zip(header, tr))
                rows.append({
                    'Date': str(d.get('Date') or tr[0]).strip(),
                    'Description': str(d.get('Description') or '').strip(),
                    'Debit Amt': str(d.get('Debit Amt') or '').replace(',','').strip(),
                    'Credit Amt': str(d.get('Credit Amt') or '').replace(',','').strip(),
                    'Balance': str(d.get('Balance') or tr[-1]).replace(',','').strip()
                })
    df = pd.DataFrame(rows, columns=['Date','Description','Debit Amt','Credit Amt','Balance'])
    return normalize_numeric(df)
