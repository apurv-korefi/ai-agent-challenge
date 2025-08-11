import pdfplumber
import pandas as pd
import numpy as np
import re
from datetime import datetime

def parse(pdf_path):
    # Read the PDF file
    with pdfplumber.open(pdf_path) as pdf:
        all_text = ""
        for page in pdf.pages:
            all_text += page.extract_text() + "\n"

    # Split the text into lines
    lines = all_text.split('\n')

    # Initialize variables
    transactions = []
    date_format = None
    header_keywords = {'Date', 'Description', 'Debit', 'Credit', 'Balance'}

    # Process each line
    for line in lines:
        # Skip empty lines and header rows
        if not line.strip() or any(keyword in line for keyword in header_keywords):
            continue

        # Split the line into parts
        parts = re.split(r'\s{2,}', line.strip())

        # Validate and process the line
        if len(parts) >= 5:
            try:
                # Extract date and validate format
                date_str = parts[0]
                if date_format is None:
                    # Try to determine date format from first valid date
                    for fmt in ['%d-%m-%Y', '%d/%m/%Y']:
                        try:
                            datetime.strptime(date_str, fmt)
                            date_format = fmt
                            break
                        except ValueError:
                            continue
                    if date_format is None:
                        continue

                # Validate date format
                try:
                    datetime.strptime(date_str, date_format)
                except ValueError:
                    continue

                # Extract balance and validate
                try:
                    balance = float(parts[-1].replace(',', ''))
                except ValueError:
                    continue

                # Extract description (all text between date and last two numerical fields)
                description = ' '.join(parts[1:-2])

                # Extract amount (second last field)
                try:
                    amount = float(parts[-2].replace(',', ''))
                except ValueError:
                    continue

                # Store transaction data
                transactions.append({
                    'Date': date_str,
                    'Description': description,
                    'Amount': amount,
                    'Balance': balance
                })
            except (ValueError, IndexError):
                continue

    # Create DataFrame with exact column structure
    df = pd.DataFrame(transactions)

    # If no transactions found, return empty DataFrame with correct structure
    if df.empty:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    # Determine debit/credit assumption
    def apply_assumption(df, first_is_debit=True):
        df_copy = df.copy()
        df_copy['Debit Amt'] = np.nan
        df_copy['Credit Amt'] = np.nan

        for i in range(len(df_copy)):
            if i == 0:
                if first_is_debit:
                    df_copy.at[i, 'Debit Amt'] = df_copy.at[i, 'Amount']
                else:
                    df_copy.at[i, 'Credit Amt'] = df_copy.at[i, 'Amount']
            else:
                balance_change = df_copy.at[i, 'Balance'] - df_copy.at[i-1, 'Balance']
                if balance_change < 0:
                    df_copy.at[i, 'Debit Amt'] = abs(balance_change)
                else:
                    df_copy.at[i, 'Credit Amt'] = balance_change

        # Convert 0 values to NaN
        df_copy['Debit Amt'] = df_copy['Debit Amt'].replace(0, np.nan)
        df_copy['Credit Amt'] = df_copy['Credit Amt'].replace(0, np.nan)

        # Ensure all amount columns are float dtype
        df_copy['Debit Amt'] = df_copy['Debit Amt'].astype(float)
        df_copy['Credit Amt'] = df_copy['Credit Amt'].astype(float)
        df_copy['Balance'] = df_copy['Balance'].astype(float)

        return df_copy[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]

    # Try first assumption (first transaction is debit)
    result_df = apply_assumption(df, first_is_debit=True)

    # Validate the result - check if any debit amount is negative
    if (result_df['Debit Amt'].dropna() < 0).any():
        # If validation fails, try second assumption (first transaction is credit)
        result_df = apply_assumption(df, first_is_debit=False)

    # Sort by date to match CSV order
    result_df['Date'] = pd.to_datetime(result_df['Date'], format=date_format)
    result_df = result_df.sort_values('Date').reset_index(drop=True)

    # Convert date back to string format
    result_df['Date'] = result_df['Date'].dt.strftime(date_format)

    return result_df