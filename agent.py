import argparse
import subprocess
import sys
from pathlib import Path
from utils import save_parser_code
from test_generator import generate_test_file
import parser_strategies  
import shutil

def print_boxed_message(lines):
    """Print a message inside a simple ASCII box."""
    # Get terminal width (fallback to 80 if not found)
    term_width = shutil.get_terminal_size((80, 20)).columns
    max_len = max(len(line) for line in lines)
    box_width = min(max_len + 4, term_width)

    print("+" + "-" * (box_width - 2) + "+")
    for line in lines:
        print("| " + line.ljust(box_width - 4) + " |")
    print("+" + "-" * (box_width - 2) + "+")




def run_pytest(test_path: Path) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0

def strategy_to_code(func_name: str) -> str:
    # Code for generated parser that imports the strategy as parse()
    return f"import pandas as pd\nfrom parser_strategies import {func_name} as parse\n"

# def agent_loop(target, pdf_path, csv_path):
#     strategies = [
#         "parse_line_split",
#         "parse_regex",
#         "parse_table_extract"
#     ]

#     for attempt, name in enumerate(strategies, start=1):
#         print(f"\n[Attempt {attempt}] Trying strategy: {name}")
#         code = strategy_to_code(name)
#         save_parser_code(target, code)
#         generate_test_file(target, pdf_path, csv_path)

#         ok = run_pytest(Path("tests") / f"test_{target}.py")
#         if ok:
#             # Check if function used table extraction or text
#             func = getattr(parser_strategies, name)
#             df = func(pdf_path)

#             # Save extracted CSV
#             output_csv = Path(f"output_{target}.csv")
#             df.to_csv(output_csv, index=False)
#             print(f"[SAVED] Extracted data saved to: {output_csv}")
            
#             with_table = check_if_table_used(func, pdf_path)
#             if with_table:
#                 print(f"[SUCCESS] Strategy '{name}' worked using TABLE extraction ✅")
#             else:
#                 print(f"[SUCCESS] Strategy '{name}' worked using TEXT parsing ✅")
#             return
#         else:
#             print(f"[FAIL] Strategy '{name}' failed. Trying next...")

#     print("[ERROR] All strategies failed ❌")


def agent_loop(target, pdf_path, csv_path):
    strategies = [
        "parse_line_split",
        "parse_regex",
        "parse_table_extract"
    ]

    for attempt, name in enumerate(strategies, start=1):
        print(f"\n[Attempt {attempt}] Trying strategy: {name}")
        code = strategy_to_code(name)
        save_parser_code(target, code)
        generate_test_file(target, pdf_path, csv_path)

        ok = run_pytest(Path("tests") / f"test_{target}.py")
        if ok:
            # Run the parser function
            func = getattr(parser_strategies, name)
            df = func(pdf_path)

            # Save extracted CSV
            output_csv = Path(f"output_{target}.csv")
            df.to_csv(output_csv, index=False)

            # Prepare boxed success message
            with_table = check_if_table_used(func, pdf_path)
            msg_lines = [f"[SAVED] Extracted data saved to: {output_csv}"]
            if with_table:
                msg_lines.append(f"[SUCCESS] Strategy '{name}' worked using TABLE extraction✅")
            else:
                msg_lines.append(f"[SUCCESS] Strategy '{name}' worked using TEXT parsing ✅")

            print_boxed_message(msg_lines)
            return
        else:
            print(f"[FAIL] Strategy '{name}' failed. Trying next...")

    print("[ERROR] All strategies failed ❌")



def check_if_table_used(func, pdf_path: Path) -> bool:
    """Small helper to detect if function extracted at least one row from a table."""
    with pdf_path.open("rb") as f:
        import pdfplumber
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                if page.extract_table():
                    return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank target name (e.g., icici)")
    parser.add_argument("--pdf", required=True, help="Path to sample PDF")
    parser.add_argument("--csv", required=True, help="Path to expected CSV")
    args = parser.parse_args()

    agent_loop(args.target, Path(args.pdf), Path(args.csv))

