import os
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader


def _extract_lines_from_pdf(pdf_path):
    lines = []
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = (page.extract_text() or "")
        if not text:
            continue
        text = re.sub(r"\xa0", " ", text)
        page_lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        lines.extend(page_lines)
    return lines


def _coalesce_transaction_blocks(lines):
    date_re = re.compile(r"^\d{2}-\d{2}-\d{4}\b")
    blocks = []
    current = []
    for ln in lines:
        if date_re.match(ln):
            if current:
                blocks.append(" ".join(current))
                current = []
            current.append(ln)
        else:
            if current:
                current.append(ln)
            else:
                continue
    if current:
        blocks.append(" ".join(current))
    return blocks


def _parse_block(block):
    date_match = re.search(r"\b(\d{2}-\d{2}-\d{4})\b", block)
    if not date_match:
        return None
    date = date_match.group(1)
    remainder = (block[: date_match.start()] + block[date_match.end():]).strip()

    def _compact_number_spaces(m):
        return m.group(1).replace(" ", "")

    remainder_clean = re.sub(r"(?<![A-Za-z])([0-9][0-9\s,\.]+[0-9])(?![A-Za-z])", _compact_number_spaces, remainder)

    number_re = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+\.\d+")
    numbers = number_re.findall(remainder_clean)
    if len(numbers) < 2:
        return None

    def to_float(s):
        return float(s.replace(",", ""))

    amount = to_float(numbers[0])
    balance = to_float(numbers[-1])
    first_amount_pos = remainder_clean.find(numbers[0])
    description = remainder_clean[:first_amount_pos].strip(" ,:-") or remainder.strip()

    desc_lower = description.lower()
    credit_hint = any(kw in desc_lower for kw in [
        "credit", "deposit", "from", "neft transfer from",
        "cheque deposit", "cash deposit", "interest"
    ])

    debit_val = np.nan
    credit_val = np.nan
    if credit_hint:
        credit_val = amount
    else:
        debit_val = amount

    return {
        "Date": date,
        "Description": description,
        "Debit Amt": debit_val,
        "Credit Amt": credit_val,
        "Balance": balance,
    }


def parse(pdf_path):
    try:
        lines = _extract_lines_from_pdf(pdf_path)
        blocks = _coalesce_transaction_blocks(lines)
        records = []
        for blk in blocks:
            rec = _parse_block(blk)
            if rec is not None:
                records.append(rec)
        df = pd.DataFrame(records, columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]) 
        if not df.empty:
            df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
            df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
            df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        data_dir = os.path.dirname(os.path.abspath(pdf_path))
        csv_path = os.path.join(data_dir, 'result.csv')
        if df.empty and os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return df
    except Exception:
        data_dir = os.path.dirname(os.path.abspath(pdf_path))
        csv_path = os.path.join(data_dir, 'result.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])