import pandas as pd
import re
from PyPDF2 import PdfReader

def parse(pdf_path: str) -> pd.DataFrame:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        lines = text.splitlines()
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^(\d{2}-\d{2}-\d{4})\s+(.+?)(?=[\\d.]+\\s|$)", line)
            if match:
                date = match.group(1)
                description = match.group(2)
                debit_match = re.search(r"(?<=\\s)\\d+(?:\\.\\d{2})?(?=\\s|$)", line)
                credit_match = re.search(r"(?<=\\s)\\d+(?:\\.\\d{2})?(?=\\s|$)", line, re.IGNORECASE)
                balance_match = re.search(r"\\d+(?:\\.\\d{2})?$", line)

                debit = float(debit_match.group(0)) if debit_match else None
                credit = float(credit_match.group(0)) if credit_match and debit is None else None
                balance = float(balance_match.group(0)) if balance_match else None

                if debit is not None:
                    credit = None
                elif credit is not None:
                    debit = None

                data.append({'Date': date, 'Description': description, 'Debit Amt': debit, 'Credit Amt': credit, 'Balance': balance})


        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
        df['Debit Amt'] = pd.to_numeric(df['Debit Amt'])
        df['Credit Amt'] = pd.to_numeric(df['Credit Amt'])
        df['Balance'] = pd.to_numeric(df['Balance'])
        return df

    except FileNotFoundError:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])
    except Exception as e:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])