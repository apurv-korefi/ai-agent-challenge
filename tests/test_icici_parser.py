
import sys
import os
from importlib import import_module
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    parser = import_module("custom_parser.icici_parser")
except ImportError:
    sys.path.append(os.path.join(project_root, 'custom_parser'))
    parser = import_module("icici_parser")

pdf_path = f"data/icici/icici sample.pdf"
csv_path = f"data/icici/result.csv"
main_df = pd.read_csv(csv_path)
df_out = parser.parse(pdf_path)
count_df = main_df.count()
count_df_out = df_out.count()

columns_df = main_df.columns
columns_df_out = main_df.columns

assert columns_df_out.equals(columns_df),"columns is not matched"
assert count_df_out.equals(count_df), "count is not same"
assert df_out.equals(main_df), "Data mismatch"

