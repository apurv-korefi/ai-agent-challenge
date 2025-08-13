import argparse
import os
import sys
import subprocess
import importlib
from typing import TypedDict
from langgraph.graph import StateGraph, END
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-pro')


class AgentState(TypedDict):
    target: str
    code: str
    test_output: str
    retries_left: int
    attempt: int
    last_error: str


def code_generation_node(state: AgentState):
    """
    Generate or update the parser code for the target using Gemini AI.
    The agent learns from previous failures and generates progressively better code.
    """
    print("---GENERATING CODE WITH GEMINI AI---")
    target = state["target"]
    attempt = state.get("attempt", 0)
    last_error = state.get("last_error", "")

    # Create context for Gemini based on attempt and previous errors
    if attempt == 0:
        # First attempt: Generate initial parser
        prompt = f"""
        You are an expert Python developer specializing in PDF parsing. Create a robust parser for ICICI bank statements.

        CRITICAL REQUIREMENTS:
        - Function signature: def parse(pdf_path: str) -> pd.DataFrame
        - Output columns: ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]
        - Use pypdf for text extraction
        - Handle malformed PDF text with embedded spaces in numbers (e.g., "6 864.58" → "6864.58")
        - Handle multi-line descriptions and mashed transactions
        - The PDF is extremely malformed but you MUST extract the actual transaction data from it
        - Focus on robust text extraction and pattern matching to reconstruct transactions
        - The goal is to extract 100 transactions that match the expected CSV format

        PARSING STRATEGY:
        - Extract text from all pages using pypdf
        - Clean up text (remove non-breaking spaces, normalize whitespace)
        - Find transaction blocks using date patterns (DD-MM-YYYY format)
        - Extract amounts, descriptions, and balances from each block
        - Use heuristics to determine debit vs credit based on description keywords
        - Handle edge cases like missing values and malformed text

        Generate clean, working Python code that can parse the ICICI PDF and return a DataFrame matching the expected CSV format.
        """
    else:
        # Subsequent attempts: Learn from previous failures
        prompt = f"""
        You are an expert Python developer. The previous parser attempt failed with this error:
        
        {last_error}
        
        This is attempt {attempt + 1} for parsing ICICI bank statements. Analyze the error and generate improved code.
        
        CRITICAL REQUIREMENTS:
        - Function signature: def parse(pdf_path: str) -> pd.DataFrame
        - Output columns: ["Date", "Description", "Debit Amt", "Credit Amt", "Balance"]
        - Use pypdf for text extraction
        - The PDF is extremely malformed but you MUST extract the actual transaction data from it
        - Focus on robust text extraction and pattern matching to reconstruct transactions
        - Fix the specific issues identified in the error message
        - The goal is to extract 100 transactions that match the expected CSV format
        
        LEARNING FROM FAILURE:
        - Analyze what went wrong in the previous attempt
        - Improve the parsing logic based on the error
        - Handle edge cases that caused the failure
        - Ensure robust error handling and data validation
        
        Generate improved Python code that addresses the previous failures and can successfully parse the malformed PDF.
        """

    try:
        # Generate code using Gemini
        response = model.generate_content(prompt)
        generated_code = response.text
        
        # Clean up the response (remove markdown formatting if present)
        if "```python" in generated_code:
            code_start = generated_code.find("```python") + 9
            code_end = generated_code.find("```", code_start)
            generated_code = generated_code[code_start:code_end].strip()
        elif "```" in generated_code:
            code_start = generated_code.find("```") + 3
            code_end = generated_code.find("```", code_start)
            generated_code = generated_code[code_start:code_end].strip()
        
        print(f"Gemini generated code for attempt {attempt + 1}")
        state['code'] = generated_code
        
    except Exception as e:
        print(f"Gemini API error: {e}")
        # If Gemini fails, create a minimal working parser that can be improved
        minimal_parser = '''
import os
import pandas as pd
from pypdf import PdfReader

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Minimal parser - needs improvement through AI learning.
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += (page.extract_text() or "") + "\\n"
        
        # Basic attempt at parsing - this will likely fail and need improvement
        lines = text.splitlines()
        data = []
        
        for line in lines:
            # Very basic parsing - needs enhancement
            parts = line.split()
            if len(parts) >= 3:
                try:
                    # This is a placeholder - needs real parsing logic
                    data.append([parts[0], " ".join(parts[1:-2]), parts[-2], parts[-1], parts[-1]])
                except:
                    continue
        
        df = pd.DataFrame(data, columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])
        return df
        
    except Exception as e:
        print(f"Parsing failed: {e}")
        return pd.DataFrame(columns=["Date", "Description", "Debit Amt", "Credit Amt", "Balance"])
'''
        
        state['code'] = minimal_parser
        print(f"Using minimal parser due to Gemini API error for attempt {attempt + 1}")

    # Write parser for both import paths used by tests
    for pkg in ["custom_parsers", "custom_parser"]:
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "__init__.py"), "w") as f:
            pass
        parser_path = os.path.join(pkg, f"{target}_parser.py")
        with open(parser_path, "w") as f:
            f.write(state['code'])
        print(f"Code written to {parser_path} (attempt {attempt + 1})")

    return state


def test_harness_node(state: AgentState):
    """Run both test suites to validate the generated parser."""
    print("---TESTING CODE---")
    env = os.environ.copy()
    env["PARSER_TARGET"] = state["target"]

    # Run both test directories if present
    args = [sys.executable, "-m", "pytest"]
    if os.path.isdir("tests"):
        args.append("tests/")
    if os.path.isdir("test1/tests"):
        args.append("test1/tests/")

    process = subprocess.run(args, env=env, capture_output=True, text=True)
    if process.returncode == 0:
        print("✅ Tests Passed!")
        state['test_output'] = "success"
        state['last_error'] = ""
    else:
        print("❌ Tests Failed!")
        state['test_output'] = "failure"
        state['last_error'] = (process.stdout or "") + "\n" + (process.stderr or "")
        print(state['last_error'])
    return state


def should_continue(state: AgentState):
    if state['test_output'] == "success":
        print("---AGENT FINISHED SUCCESSFULLY---")
        return "end"
    if state.get('retries_left', 0) > 0:
        print(f"---SELF-FIX: attempts left {state['retries_left'] - 1}---")
        return "retry"
    print("---AGENT FAILED (NO RETRIES LEFT)---")
    return "end"


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("coder", code_generation_node)
    workflow.add_node("tester", test_harness_node)
    
    # Self-fix node to mutate attempt and retries
    def self_fix_node(state: AgentState):
        state['attempt'] = state.get('attempt', 0) + 1
        state['retries_left'] = max(0, state.get('retries_left', 0) - 1)
        return state
    workflow.add_node("self_fix", self_fix_node)

    workflow.set_entry_point("coder")
    workflow.add_edge("coder", "tester")
    workflow.add_conditional_edges("tester", should_continue, {"end": END, "retry": "self_fix"})
    workflow.add_edge("self_fix", "coder")
    return workflow.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Self-Fixing Agent for parsing PDF bank statements.")
    parser.add_argument("--target", required=True, help="Bank target name (e.g., 'icici').")
    args = parser.parse_args()

    app = build_graph()
    initial_state = {
        "target": args.target,
        "retries_left": 2,  # total attempts = 3
        "attempt": 0,
        "last_error": "",
        "code": "",
        "test_output": "",
    }
    final_state = app.invoke(initial_state)

    print("\n---FINAL PARSER CODE---")
    print(final_state.get('code', 'No code was generated.'))
