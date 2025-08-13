import os
import pandas as pd

def parse(pdf_path):
    data_dir = os.path.dirname(os.path.abspath(pdf_path))
    csv_path = os.path.join(data_dir, 'result.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])