from typing_extensions import TypedDict, List, Sequence,Optional
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
import os
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_community.document_loaders import PDFPlumberLoader
import pandas as pd
import subprocess
from importlib import import_module
import time

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

code_content = ""

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    bank: str
    attempts: int
    pdf_content: str
    csv_schema: dict
    code_content: str
    test_results: str
    error_logs: str
    test_code: Optional[str] = None

def extract_code(raw: str) -> str:
    """Extract clean Python code from LLM response text."""
    raw = str(raw).strip()
    
    # Try to find python code block first
    if "```python" in raw:
        start_idx = raw.find("```python") + 9
        end_idx = raw.rfind("```")
        if end_idx > start_idx:
            return raw[start_idx:end_idx].strip()
    
    # If no code block found, return the raw content (assuming it's all code)
    return raw

@tool
def update_code(code: str) -> str:
    """Update the in-memory code content for the parser."""
    global code_content
    code_content = code
    return code_content

@tool
def save_file(file_path: str) -> str:
    """Save the file and create __init__.py if needed."""
    global code_content
    if not file_path.endswith(".py"):
        file_path = f"{file_path}.py"
    try:
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w", encoding="utf-8") as init_f:
                init_f.write("# Custom parser module\n")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(code_content)
        return f"Saved {file_path}"
    except Exception as e:
        return f"Error saving file: {str(e)}"

@tool
def read_pdf_text(pdf_path: str) -> str:
    """Read the PDF file text. and convert it the data frame"""
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    return "".join([str(doc.page_content) for doc in docs])

@tool
def read_csv_schema(csv_path: str) -> dict:
    """Read CSV file schema and sample data."""
    df = pd.read_csv(csv_path)
    return {"columns_data": df.columns.to_list(), "sample_data": df.head(7).to_dict('records'),"data_types":df.shape}

@tool
def run_tests(test_code: str, bank: str) -> str:
    """Run parser tests."""
    import os, subprocess

    os.makedirs("tests", exist_ok=True)
    test_file = f"tests/test_{bank}_parser.py"

    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_code)

    result = subprocess.run(["pytest", "-q", test_file], capture_output=True, text=True)
    return "PASS" if result.returncode == 0 else f"fail: {result.stdout} {result.stderr}"


@tool
def run_parser_result(bank: str,test_code: str) -> str:
    """Generate parser output CSV."""
    parser = import_module(f"custom_parser.{bank}_parser")
    pdf_path = f"data/{bank}/{bank}_sample.pdf"
    return parser.parse(pdf_path).to_csv(index=False)

tools = [read_pdf_text, read_csv_schema, save_file, update_code, run_tests, run_parser_result]
llm = ChatMistralAI(model_name="devstral-medium-latest", temperature=0.9,api_key=API_KEY).bind_tools(tools)
memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

def plan(state: State):
    bank = state['bank']
    pdf_data = read_pdf_text(f"data/{bank}/{bank} sample.pdf")
    csv_data = read_csv_schema(f"data/{bank}/result.csv")
    pdf_data_snippet = pdf_data[:1000]
    csv_data_snippet = str(csv_data)[:1000]
    sys_msg = SystemMessage(content="Plan a parsing strategy based on PDF and CSV schema.hint analyze the csv_snippet based on this give me ")
    human_msg = HumanMessage(content=f"PDF:\n{pdf_data_snippet}\n\nCSV Schema:\n{csv_data_snippet} and ")
    resp = llm.invoke([sys_msg, human_msg] + state["messages"])
    print(resp.content)
    # Pass these snippets forward
    return {**state, "messages": state["messages"] + [resp], "pdf_content": pdf_data_snippet, "csv_schema": csv_data_snippet}


def generate_code(state: State):
    bank = state["bank"]
    pdf_snippet = state.get("pdf_content", "<no pdf snippet>")
    csv_snippet = state.get("csv_schema", "<no csv snippet>")

    system_msg = SystemMessage(
        content=("""
            Write Python parser that constructs DataFrame based on CSV structure.
Function def parse(pdf_path) must be mandatory. Use pdfplumber only.
Wrap your code inside triple backticks with ```python at the start and ``` at the end.

CSV-BASED CONSTRUCTION STRATEGY:
1. Extract ALL transactions from PDF (Date, Description, Balance) - not just samples
2. Create DataFrame with exact same structure as target CSV
3. Use ASSUMPTION-BASED logic for Debit/Credit determination
4. If first assumption fails during validation, automatically switch to opposite assumption

ASSUMPTION LOGIC FOR AMOUNT PLACEMENT:
ASSUMPTION 1 (Try First): First transaction amount goes to DEBIT column
- Calculate: balance_change = current_balance - previous_balance
- If balance_change < 0: Place amount in Debit Amt, NaN in Credit Amt
- If balance_change > 0: Place amount in Credit Amt, NaN in Debit Amt

ASSUMPTION 2 (Fallback): First transaction amount goes to CREDIT column
- Same calculation logic but reverse the initial assumption

DATA MATCHING REQUIREMENTS:
1. Date format MUST match CSV exactly - inspect CSV date format and replicate
2. Amount columns: float dtype, convert 0 values to NaN
3. Extract ALL PDF data, ensure row count matches CSV
4. Balance column is the last numerical column from PDF
5. Description includes all text between date and numerical fields
6. Skip header rows properly by detecting keywords"""
        )
    )
    
    test_code_str = state.get('test_code', '<no test code yet>')

    human_msg = HumanMessage(
        content=(
            f"PDF content snippet:\n{pdf_snippet}\n\n"
            f"CSV schema:\n{csv_snippet}\n\n"
            f"Write parser for [data/{bank}/{bank} sample.pdf] matching [data/{bank}/result.csv]. "
            f"Test code: {test_code_str}\n\n"
            
            "PARSING REQUIREMENTS: "
            "1. Skip ALL header rows - check if line contains 'Date', 'Balance', 'Description' etc. "
            "2. Validate numeric data - use try/except when converting to float "
            "3. Example validation: "
            "   ```python"
            "   if parts and len(parts) >= 5:"
            "       try:"
            "           # Check if last part is actually a number"
            "           balance = float(parts[-1])"
            "           # Additional validation..."
            "       except ValueError:"
            "           continue  # Skip invalid rows"
            "   ```"
            "4. Date format should match CSV exactly (check if it's DD-MM-YYYY or DD/MM/YYYY) "
            "5. Handle cases where description contains multiple words "
            "6. Use pandas date parsing with correct format parameter "
        )
    )

    

    trace = llm.invoke([system_msg, human_msg])
    print("Raw LLM response:")
    print(trace.content)
    cleaned_code = extract_code(trace.content)
    print(f"Cleaned code extracted:\n{cleaned_code}")
    print(csv_snippet)
    print()
    print(pdf_snippet)

    # Update the global variable directly
    global code_content
    code_content = cleaned_code

    # Save file using updated code_content
    save_file(f"custom_parser/{bank}_parser.py")

    return {**state, "messages": state["messages"] + [trace], "code_content": cleaned_code}


def run_tests_node(state: State):
    bank = state['bank']
    
    test_code = f"""
import sys
import os
from importlib import import_module
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    parser = import_module("custom_parser.{bank}_parser")
except ImportError:
    sys.path.append(os.path.join(project_root, 'custom_parser'))
    parser = import_module("{bank}_parser")

pdf_path = f"data/{bank}/{bank} sample.pdf"
csv_path = f"data/{bank}/result.csv"
main_df = pd.read_csv(csv_path)
df_out = parser.parse(pdf_path)
count_df = main_df.count()
count_df_out = df_out.count()

columns_df = main_df.columns
columns_df_out = main_df.columns

assert columns_df_out.equals(columns_df),"columns is not matched"
assert count_df_out.equals(count_df), "count is not same"
assert df_out.equals(main_df), "Data mismatch"

"""
    
    result = run_tests.invoke({"test_code": test_code, "bank": bank})


    new_state = {**state, "test_code": test_code, "test_results": result}

    if result.startswith("PASS"):
        print(f"[Attempt {state['attempts']}] ✅ PASS")
        return new_state
    else:
        print(result)
        return {**new_state, "error_logs": result}

    

def self_fixer(state: State):
    bank = state["bank"]
    print(f"[Attempt {state['attempts']}] 🔧 Fixing code after FAIL...")
    
    sys_msg = SystemMessage(
        content=(
            """Fix parser using CSV-DataFrame construction approach. Provide only clean Python code.

CONSTRUCTION-BASED FIXING STRATEGY:
1. Build DataFrame to match CSV structure exactly
2. Use assumption-based logic for amount assignment
3. Implement assumption switching if validation fails
4. Focus on data type and format matching

ERROR-SPECIFIC FIXING INSTRUCTIONS:
- 'could not convert string to float': Add proper header row detection and skip logic
- 'columns not matched': Ensure exact column names and order match CSV
- 'count not same': Extract all PDF rows, check for data loss during parsing
- 'Data mismatch': Fix assumption logic or date formatting issues

ASSUMPTION SWITCHING APPROACH:
- If current assumption fails validation, reverse the debit/credit assignment logic
- Keep balance calculation consistent: current_balance - previous_balance
- Switch which column gets the amount based on positive/negative balance change

DATA TYPE ENFORCEMENT:
- All amount columns must be float dtype
- Convert 0 values to NaN using np.nan
- Date parsing must match CSV format exactly
- Handle missing or malformed data gracefully"""
        )
    )
    
    human_msg = HumanMessage(content=(
        f"SPECIFIC ERROR: {state['error_logs']}\n\n"
        f"Current parser code:\n{state['code_content']}\n\n"
        f"Test expectations:\n{state['test_code']}\n\n"
        
        "DEBUGGING CHECKLIST: "
        "✓ Are you skipping header rows properly? "
        "✓ Are you validating numeric data before float() conversion? "
        "✓ Are column names exactly: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']? "
        "✓ Are you handling NaN values with np.nan (not None or '')? "
        "✓ Is date format matching the expected output? "
        "✓ Are you sorting by date correctly? "
        "✓ Is the debit/credit logic based on balance changes? "
        
        "SPECIFIC FIX FOR CURRENT ERROR: "
        "If error contains 'could not convert string to float', add: "
        "1. Header detection and skipping "
        "2. try/except blocks around float conversions "
        "3. Data validation before processing "
    ))
    
    # Rest of the function remains the same...
    time.sleep(3)
    resp = llm.invoke([sys_msg, human_msg])
    fixed_code = extract_code(resp.content)
    print(f"fix code is {resp.content}")
    update_code(fixed_code)
    save_file(f"custom_parser/{bank}_parser.py")
    return {**state, "code_content": fixed_code, "attempts": state["attempts"] + 1}

def should_continue(state: State):
    return "end" if state["test_results"].startswith("PASS") or state["attempts"] >= 3 else "self_fix"

graph_builder = StateGraph(State)
graph_builder.add_node("the_planner", plan)
graph_builder.add_node("gen_code", generate_code)
graph_builder.add_node("run_test_node", run_tests_node)
graph_builder.add_node("self_fixer", self_fixer)
graph_builder.add_edge(START, "the_planner")
graph_builder.add_edge("the_planner", "gen_code")
graph_builder.add_edge("gen_code", "run_test_node")
graph_builder.add_conditional_edges("run_test_node", should_continue, {"self_fix": "self_fixer", "end": END})
graph_builder.add_edge("self_fixer", "run_test_node")
graph = graph_builder.compile(checkpointer=memory)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Bank name (e.g., icici)")
    args = parser.parse_args()

    initial_state = State(messages=[], bank=args.target, attempts=0, pdf_content="", csv_schema={}, code_content="", test_results="", error_logs="")
    graph.invoke(initial_state, config)
