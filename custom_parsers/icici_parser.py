import pdfplumber
import csv
import re

def parse_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        pages = []
        for page in pdf.pages:
            text = page.extract_text()
            text = re.sub(r'\x0c', '', text)
            lines = text.split('\n')
            transactions = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(' ')
                date, description = parts[0], ' '.join(parts[1:])
                debit = ''
                credit = ''
                balance = ''
                for part in parts:
                    if re.match(r'^\d{1,2}\/\d{1,2}\/\d{2,4}$', part):
                        date = part
                    elif re.match(r'^[\d\$]+$', part):
                        if debit:
                            credit = part
                        else:
                            debit = part
                    elif re.match(r'^Balance (.*)$', part):
                        balance = part[8:]
                transactions.append({
                    'Date': re.sub(r'/[^,]*\.', '', date),
                    'Description': re.sub(r'[^\w\s]', '', description),
                    'Debit Amt': debit or '',
                    'Credit Amt': credit or '',
                    'Balance': balance or ''
                })
            pages.append(transactions)
    return pages[0]
