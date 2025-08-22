# agent.py
import argparse
import subprocess
import sys
import os
from pathlib import Path

import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================
# API Keys (from environment variables)
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    print("⚠️  Warning: API keys not found in environment variables.")
    print("Please copy .env.example to .env and add your API keys.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)


# ======================
# AI Backend Selection
# ======================
def choose_backend(preferred: str | None = None):
    if preferred == "gemini":
        return "gemini"
    elif preferred == "groq":
        return "groq"
    return "gemini"  # default


def call_ai_model(prompt: str, backend: str = "gemini") -> str:
    if backend == "gemini":
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt)
        return resp.text
    elif backend == "groq":
        resp = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ======================
# Core Loop
# ======================
def generate_parser_code(bank: str, csv_sample: str, backend: str) -> str:
    prompt = f"""
Write ONLY Python code for a bank statement parser. No explanations, no markdown, no comments outside the code.

Bank: {bank}
CSV columns: {csv_sample.split('\n')[0]}

Requirements:
- Function: parse_pdf(pdf_path) -> list[dict]
- Use pdfplumber library
- Return list of dictionaries with keys: {csv_sample.split('\n')[0].split(',')}
- Extract transaction data from PDF text
- Handle empty values as empty strings
- No explanatory text, just pure Python code
"""
    return call_ai_model(prompt, backend=backend)


def write_parser_file(bank: str, code: str):
    parser_dir = Path("custom_parsers")
    parser_dir.mkdir(exist_ok=True)
    file_path = parser_dir / f"{bank}_parser.py"

    # Clean the code - remove null bytes and markdown blocks
    clean_code = code.replace("\x00", "").replace("\ufeff", "")  # Remove null bytes and BOM
    
    # Remove any explanatory text before the code
    lines = clean_code.split('\n')
    code_start = -1
    
    # Find the first line that looks like Python code
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith('import ') or 
            stripped.startswith('from ') or 
            stripped.startswith('def ') or
            stripped.startswith('class ')):
            code_start = i
            break
    
    if code_start != -1:
        clean_code = '\n'.join(lines[code_start:])
    
    # Remove markdown code blocks if present
    if "```python" in clean_code:
        lines = clean_code.split('\n')
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith("```python"):
                start_idx = i + 1
            elif line.strip() == "```" and start_idx != -1:
                end_idx = i
                break
        
        if start_idx != -1 and end_idx != -1:
            clean_code = '\n'.join(lines[start_idx:end_idx])
        elif start_idx != -1:
            clean_code = '\n'.join(lines[start_idx:])
    
    # Remove any trailing markdown or explanatory text
    if "```" in clean_code:
        clean_code = clean_code.split("```")[0]
    
    # Write as binary to avoid encoding issues, then ensure it's clean
    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(clean_code)
    print(f"✅ Wrote parser to {file_path}")
    return file_path


def run_pytest(bank: str) -> bool:
    print(f"🔍 Running tests for {bank}...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/test_parser_contract.py::test_contract[{bank}]", "-v"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank target, e.g. icici")
    parser.add_argument("--backend", choices=["gemini", "groq"], help="AI backend to use")
    args = parser.parse_args()

    bank = args.target.lower()
    backend = choose_backend(args.backend)

    pdf_path = Path(f"data/{bank}/{bank}_sample.pdf")
    csv_path = Path(f"data/{bank}/{bank}_sample.csv")

    assert pdf_path.exists(), f"Missing {pdf_path}"
    assert csv_path.exists(), f"Missing {csv_path}"

    csv_sample = csv_path.read_text(encoding="utf-8")

    success = False
    for attempt in range(3):
        print(f"\n🌀 Attempt {attempt+1}/3 using {backend}...")
        code = generate_parser_code(bank, csv_sample, backend)
        write_parser_file(bank, code)

        if run_pytest(bank):
            print("✅ Tests passed!")
            success = True
            break
        else:
            print("❌ Tests failed, retrying...\n")

    if not success:
        print("🚨 Failed after 3 attempts.")


if __name__ == "__main__":
    main()
