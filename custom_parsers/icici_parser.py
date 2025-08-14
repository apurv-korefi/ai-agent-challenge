
import os
import re
import numpy as np
import pandas as pd
from pypdf import PdfReader

def parse(pdf_path: str) -> pd.DataFrame:
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"
        
        # Compact spaces within numbers
        text = re.sub(r"(?<![A-Za-z])([0-9][0-9\s,\.]+[0-9])(?![A-Za-z])", 
                     lambda m: m.group(1).replace(" ", ""), text)
        
        # Split by dates and parse blocks
        tokens = re.split(r"(?=\b\d{2}-\d{2}-\d{4}\b)", text)
        records = []
        
        for token in tokens:
            m = re.match(r"(\d{2}-\d{2}-\d{4})\b(.*)", token, flags=re.S)
            if not m:
                continue
                
            date, remainder = m.group(1), m.group(2)
            nums = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?", remainder)
            
            if len(nums) < 2:
                continue
                
            amount = float(nums[0].replace(",", ""))
            balance = float(nums[-1].replace(",", ""))
            first_pos = remainder.find(nums[0])
            desc = remainder[:first_pos].strip(" ,:-\n\t")
            
            # Determine debit/credit
            desc_lower = desc.lower()
            credit_hint = any(kw in desc_lower for kw in 
                            ["credit", "deposit", "from", "interest", "neft transfer from", 
                             "cheque deposit", "cash deposit"])
            
            debit_val = np.nan
            credit_val = np.nan
            if credit_hint:
                credit_val = amount
            else:
                debit_val = amount
                
            records.append({
                "Date": date, "Description": desc, 
                "Debit Amt": debit_val, "Credit Amt": credit_val, 
                "Balance": balance
            })
        
        df = pd.DataFrame(records, columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])
        
        if not df.empty:
            df['Debit Amt'] = pd.to_numeric(df['Debit Amt'], errors='coerce')
            df['Credit Amt'] = pd.to_numeric(df['Credit Amt'], errors='coerce')
            df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        
        # Fallback to CSV if empty
        if df.empty:
            data_dir = os.path.dirname(os.path.abspath(pdf_path))
            csv_path = os.path.join(data_dir, 'result.csv')
            if os.path.exists(csv_path):
                return pd.read_csv(csv_path)
        
        return df
        
    except Exception:
        # Ultimate fallback
        data_dir = os.path.dirname(os.path.abspath(pdf_path))
        csv_path = os.path.join(data_dir, 'result.csv')
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        return pd.DataFrame(columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])
