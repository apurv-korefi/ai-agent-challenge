import pandas as pd
import pytest
import os
import importlib
import glob 

def test_parser_output():
    """
    Tests the output of the dynamically imported parser against the expected CSV.
    The target bank is passed via an environment variable `PARSER_TARGET`.
    """
    # 1. Get the target bank from an environment variable set by the agent
    target_bank = os.getenv("PARSER_TARGET")
    if not target_bank:
        pytest.fail("FATAL: The 'PARSER_TARGET' environment variable was not set.")

    parser_module_path = f"custom_parsers.{target_bank}_parser"
    
    # --- THIS IS THE FIX ---
    # Dynamically find the PDF file in the target directory instead of hardcoding the name.
    pdf_files = glob.glob(f"data/{target_bank}/*.pdf")
    if not pdf_files:
        pytest.fail(f"FATAL: No PDF file found in the directory 'data/{target_bank}/'")
    pdf_path = pdf_files[0] # Use the first PDF found
    # --- END OF FIX ---

    expected_csv_path = f"data/{target_bank}/result.csv"
    if target_bank == "simple":
        expected_csv_path = f"data/{target_bank}/simple_result.csv"


    # 2. Dynamically import the agent-generated parser module
    try:
        # We need to force a reload in case the agent has rewritten the file.
        if parser_module_path in locals() or parser_module_path in globals():
             parser_module = importlib.reload(importlib.import_module(parser_module_path))
        else:
             parser_module = importlib.import_module(parser_module_path)
    except ImportError:
        pytest.fail(f"Could not import the parser module at '{parser_module_path}'.")

    # 3. Check if the 'parse' function exists
    if not hasattr(parser_module, 'parse'):
        pytest.fail(f"The generated parser '{parser_module_path}' does not have a 'parse' function.")

    # 4. Run the parser to get the actual DataFrame
    print(f"\n🧪 Running generated parser: {parser_module_path}.parse('{pdf_path}')")
    actual_df = parser_module.parse(pdf_path)

    # 5. Load the ground truth DataFrame
    expected_df = pd.read_csv(expected_csv_path)

    # 6. Standardize DataFrames for reliable comparison
    try:
        expected_df = expected_df[actual_df.columns]
    except KeyError as e:
        pytest.fail(f"Column mismatch. Missing columns in actual DataFrame: {e}")

    actual_df = actual_df.reset_index(drop=True)
    expected_df = expected_df.reset_index(drop=True)

    actual_df = actual_df.astype(str)
    expected_df = expected_df.astype(str)

    # 7. Assert that the DataFrames are equal
    print("-----EXPECTED DF (HEAD)----")
    print(expected_df.head())
    print("\n-----ACTUAL DF (HEAD)------")
    print(actual_df.head())
    print("--------------------")
    pd.testing.assert_frame_equal(actual_df, expected_df)