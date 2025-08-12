import os
import pandas as pd
import pdfplumber
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END, START
import asyncio
import google.generativeai as genai
import logging
import argparse
from dotenv import load_dotenv
import sys
import difflib
import importlib.util

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_code: str
    error_message: str
    attempts: int
    max_attempts: int
    success: bool
    plan: str
    debug_info: str

class BankStatementParserAgent:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Obtain it from https://ai.google.dev.")
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            raise
        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self.plan)
        graph.add_node("generate_code", self.generate_code)
        graph.add_node("test_code", self.test_code)
        graph.add_node("fix_code", self.fix_code)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "generate_code")
        graph.add_edge("generate_code", "test_code")
        graph.add_conditional_edges(
            "test_code",
            lambda state: "fix_code" if not state["success"] and state["attempts"] < state["max_attempts"] else END,
            {"fix_code": "fix_code", END: END}
        )
        graph.add_edge("fix_code", "generate_code")
        return graph.compile()

    async def _clean_generated_code(self, code_text: str) -> str:
        """Removes markdown code block markers while preserving indentation."""
        lines = code_text.splitlines()
        cleaned_lines = []
        in_code_block = False
        for line in lines:
            line_stripped = line.strip()
            if line_stripped in ("```python", "```py"):
                in_code_block = True
                continue
            if line_stripped == "```":
                in_code_block = False
                continue
            if in_code_block or (not line_stripped.startswith('```') and not line_stripped.endswith('```')):
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines).strip()

    async def plan(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Planning parser generation for {state['target_bank']}")
        plan_prompt = f"""
        Create a plan to generate a parser for {state['target_bank']} bank statement PDF.
        The parser should:
        1. Read PDF from {state['pdf_path']}
        2. Extract transaction data
        3. Return a pandas DataFrame matching the schema in {state['csv_path']}
        4. Handle common bank statement formats (date, description, amount, etc.)
        """
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(plan_prompt))
            plan = response.text
            logger.info(f"Generated plan: {plan[:100]}...")
            return {"plan": plan}
        except Exception as e:
            logger.error(f"Failed to generate plan: {str(e)}")
            return {"error_message": str(e), "success": False}

    async def generate_code(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Generating parser code for {state['target_bank']}, attempt {state['attempts'] + 1}")
        
        try:
            expected_df = pd.read_csv(state['csv_path'])
            schema = expected_df.columns.tolist()
        except Exception as e:
            logger.error(f"Failed to read CSV schema: {str(e)}")
            return {"error_message": f"Failed to read CSV schema: {str(e)}", "success": False}

        prompt = f"""
        Write a complete and functional Python parser for {state['target_bank']} bank statement PDF.
        The parser code must be fully self-contained and ready to be written to a file.
        The entire output must be just the Python code itself, with correct indentation. Do not include any markdown formatting, text descriptions, or code block delimiters like (```).

        The parser must:
        1. Use the `pdfplumber` and `pandas` libraries.
        2. Extract all transaction data and organize it into a `pandas.DataFrame`.
        3. The final DataFrame must match this exact schema: {schema}.
        4. Define a single function `parse_pdf(path: str) -> pd.DataFrame` that encapsulates all the parsing logic.
        5. The function must include error handling for parsing issues and return an empty DataFrame or raise a clear exception on failure.
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(prompt))
            parser_code = response.text.strip()
            
            parser_code = await self._clean_generated_code(parser_code)
            
            if not parser_code:
                logger.error("Generated code is empty after cleaning")
                return {"error_message": "Generated code is empty after cleaning", "success": False}
            
            # Create custom_parser directory and __init__.py if they don't exist
            custom_parser_dir = "custom_parser"
            os.makedirs(custom_parser_dir, exist_ok=True)
            init_path = os.path.join(custom_parser_dir, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write("")
                logger.info("Created __init__.py in custom_parser")
            
            # Write parser code to file
            parser_path = os.path.join(custom_parser_dir, f"{state['target_bank']}_parser.py")
            with open(parser_path, "w", encoding="utf-8") as f:
                f.write(parser_code)
            logger.info(f"Parser code successfully written to {parser_path}")
            
            if not os.path.exists(parser_path) or os.path.getsize(parser_path) == 0:
                logger.error(f"Parser file {parser_path} is invalid or empty")
                return {"error_message": f"Parser file {parser_path} is invalid or empty", "success": False}
            
            return {"parser_code": parser_code, "attempts": state["attempts"] + 1}
        except Exception as e:
            logger.error(f"Failed to generate code: {str(e)}")
            return {"error_message": f"Error during code generation: {str(e)}", "success": False}

    async def test_code(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Testing parser for {state['target_bank']}")
        try:
            parser_path = os.path.join("custom_parser", f"{state['target_bank']}_parser.py")
            if not os.path.exists(parser_path):
                logger.error(f"Parser file {parser_path} does not exist")
                return {"success": False, "error_message": f"Parser file {parser_path} does not exist", "debug_info": ""}

            spec = importlib.util.spec_from_file_location(f"{state['target_bank']}_parser", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            sys.modules[f"{state['target_bank']}_parser"] = parser_module
            spec.loader.exec_module(parser_module)
            
            if not os.path.exists(state["pdf_path"]):
                logger.error(f"PDF file {state['pdf_path']} does not exist")
                return {"success": False, "error_message": f"PDF file {state['pdf_path']} does not exist", "debug_info": ""}

            result_df = parser_module.parse_pdf(state["pdf_path"])
            
            if result_df.empty:
                logger.error("Parser returned an empty DataFrame")
                return {"success": False, "error_message": "Empty DataFrame returned from parser", "debug_info": "Empty DataFrame"}
            
            expected_df = pd.read_csv(state["csv_path"])
            
            if list(result_df.columns) != list(expected_df.columns):
                debug_info = f"Schema mismatch: Expected {expected_df.columns}, got {result_df.columns}"
                logger.error(debug_info)
                return {"success": False, "error_message": "Schema mismatch", "debug_info": debug_info}

            if result_df.equals(expected_df):
                logger.info("Parser output matches expected CSV")
                return {"success": True, "debug_info": ""}
            else:
                result_str = result_df.to_string()
                expected_str = expected_df.to_string()
                diff = '\n'.join(difflib.unified_diff(
                    expected_str.splitlines(), result_str.splitlines(),
                    fromfile='expected', tofile='actual', lineterm=''
                ))
                debug_info = f"DataFrame mismatch:\n{diff}"
                logger.error(f"Parser output does not match expected CSV:\n{debug_info}")
                return {"success": False, "error_message": "DataFrame mismatch", "debug_info": debug_info}
                
        except (ImportError, AttributeError) as e:
            error_message = f"Code execution failed due to an import or attribute error: {str(e)}"
            logger.error(error_message)
            return {"success": False, "error_message": error_message, "debug_info": f"Import/Attribute Error: {str(e)}"}
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return {"success": False, "error_message": str(e), "debug_info": f"Test error: {str(e)}"}
    
    async def fix_code(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Fixing parser code for {state['target_bank']}")
        
        fix_prompt = f"""
        The previous attempt to generate a parser for {state['target_bank']} failed.
        The error message was: {state['error_message']}
        Additional debug information: {state.get('debug_info', 'No debug info available')}
        The current code that failed is:
        ```python
        {state['parser_code']}
        ```
        Please provide a new, corrected version of the Python code. The output should ONLY be the new, complete Python code. Do NOT include any explanations, markdown formatting, or code block delimiters (```). Ensure the new code has correct indentation and is a full replacement for the old code.
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(fix_prompt))
            fixed_code = response.text.strip()
            
            fixed_code = await self._clean_generated_code(fixed_code)
            
            if not fixed_code:
                logger.error("Fixed code is empty after cleaning")
                return {"error_message": "Fixed code is empty after cleaning", "success": False}
            
            return {"parser_code": fixed_code}
        except Exception as e:
            logger.error(f"Failed to fix code: {str(e)}")
            return {"error_message": str(e), "success": False}

    async def run(self, target_bank: str, pdf_path: str, csv_path: str) -> bool:
        initial_state = AgentState(
            target_bank=target_bank,
            pdf_path=pdf_path,
            csv_path=csv_path,
            parser_code="",
            error_message="",
            attempts=0,
            max_attempts=3,
            success=False,
            plan="",
            debug_info=""
        )
        final_state = await self.workflow.ainvoke(initial_state)
        return final_state["success"]

def main():
    parser = argparse.ArgumentParser(description="Bank Statement Parser Agent")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici)")
    args = parser.parse_args()
    
    agent = BankStatementParserAgent()
    pdf_path = f"data/{args.target}/{args.target}_sample.pdf"
    csv_path = f"data/{args.target}/{args.target}_sample.csv"
    
    if not os.path.exists(pdf_path) or not os.path.exists(csv_path):
        logger.error(f"Required PDF file ({pdf_path}) or CSV file ({csv_path}) not found.")
        return
    
    success = asyncio.run(agent.run(args.target, pdf_path, csv_path))
    if success:
        logger.info(f"Successfully generated parser for {args.target}")
    else:
        logger.error(f"Failed to generate parser for {args.target} after max attempts")

if __name__ == "__main__":
    main()
