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
            self.model = genai.GenerativeModel('gemini-1.5-pro')
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
        Write a Python parser for {state['target_bank']} bank statement PDF.
        The parser must:
        1. Use pdfplumber to read {state['pdf_path']}
        2. Extract transaction data into a pandas DataFrame
        3. Match the exact schema: {schema}
        4. Implement a function parse_pdf(path) -> pd.DataFrame
        5. Handle common bank statement formats (e.g., date, description, amount)
        6. Return only pure Python code without markdown, code block indicators (```), or any non-Python text
        7. Ensure the code is robust against common parsing errors
        8. Include error handling for parsing issues
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(prompt))
            parser_code = response.text.strip()
            logger.debug(f"Raw Gemini response:\n{parser_code}")
            
            lines = parser_code.splitlines()
            cleaned_code = []
            for line in lines:
                line = line.strip()
                if line.startswith("```") or line.endswith("```") or not line:
                    continue
                cleaned_code.append(line)
            parser_code = "\n".join(cleaned_code)
            
            if not parser_code:
                logger.error("Generated code is empty after cleaning")
                return {"error_message": "Generated code is empty after cleaning", "success": False}
            
            #create custom_parser directory if it doesn't exist
            custom_parser_dir = "custom_parser"
            try:
                os.makedirs(custom_parser_dir, exist_ok=True)
                logger.info(f"Created/verified custom_parser directory at {custom_parser_dir}")
            except Exception as e:
                logger.error(f"Failed to create custom_parser directory: {str(e)}")
                return {"error_message": f"Failed to create custom_parser directory: {str(e)}", "success": False}
            
            init_path = os.path.join(custom_parser_dir, "__init__.py")
            try:
                if not os.path.exists(init_path):
                    with open(init_path, "w", encoding="utf-8") as f:
                        f.write("")
                    logger.info("Created __init__.py in custom_parser")
            except Exception as e:
                logger.error(f"Failed to create __init__.py: {str(e)}")
                return {"error_message": f"Failed to create __init__.py: {str(e)}", "success": False}
            
            # Write parser code to file
            parser_path = os.path.join(custom_parser_dir, f"{state['target_bank']}_parser.py")
            try:
                with open(parser_path, "w", encoding="utf-8") as f:
                    f.write(parser_code)
                logger.info(f"Parser code successfully written to {parser_path}")
            except Exception as e:
                logger.error(f"Failed to write parser file {parser_path}: {str(e)}")
                return {"error_message": f"Failed to write parser file: {str(e)}", "success": False}
            
            # Verify file exists and is not empty
            if not os.path.exists(parser_path):
                logger.error(f"Parser file {parser_path} was not created")
                return {"error_message": f"Parser file {parser_path} not created", "success": False}
            if os.path.getsize(parser_path) == 0:
                logger.error(f"Parser file {parser_path} is empty")
                return {"error_message": f"Parser file {parser_path} is empty", "success": False}
                
            # Verify file permissions
            if not os.access(parser_path, os.R_OK):
                logger.error(f"Parser file {parser_path} is not readable")
                return {"error_message": f"Parser file {parser_path} is not readable", "success": False}
            
            return {"parser_code": parser_code, "attempts": state["attempts"] + 1}
        except Exception as e:
            logger.error(f"Failed to generate code: {str(e)}")
            return {"error_message": f"Gemini API error: {str(e)}", "success": False}

    async def test_code(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Testing parser for {state['target_bank']}")
        try:
            parser_path = os.path.join("custom_parser", f"{state['target_bank']}_parser.py")
            if not os.path.exists(parser_path):
                logger.error(f"Parser file {parser_path} does not exist")
                return {"success": False, "error_message": f"Parser file {parser_path} does not exist", "debug_info": ""}

            # Direct import workaround
            spec = importlib.util.spec_from_file_location(f"{state['target_bank']}_parser", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            sys.modules[f"{state['target_bank']}_parser"] = parser_module
            spec.loader.exec_module(parser_module)
            
            # Check if PDF file exists and is valid
            if not os.path.exists(state["pdf_path"]):
                logger.error(f"PDF file {state['pdf_path']} does not exist")
                return {"success": False, "error_message": f"PDF file {state['pdf_path']} does not exist", "debug_info": ""}

            result_df = parser_module.parse_pdf(state["pdf_path"])
            
            if result_df.empty:
                logger.error("Parser returned an empty DataFrame")
                return {"success": False, "error_message": "Empty DataFrame returned from parser", "debug_info": "Empty DataFrame"}
            
            expected_df = pd.read_csv(state["csv_path"])
            
            # Check schema match
            if list(result_df.columns) != list(expected_df.columns):
                logger.error(f"Schema mismatch: Expected {expected_df.columns}, got {result_df.columns}")
                debug_info = f"Schema mismatch: Expected {expected_df.columns}, got {result_df.columns}"
                return {"success": False, "error_message": "Schema mismatch", "debug_info": debug_info}
            
            # Check DataFrame equality
            if result_df.equals(expected_df):
                logger.info("Parser output matches expected CSV")
                return {"success": True, "debug_info": ""}
            else:
                # Generate debug info with DataFrame differences
                result_str = result_df.to_string()
                expected_str = expected_df.to_string()
                diff = '\n'.join(difflib.unified_diff(
                    expected_str.splitlines(), result_str.splitlines(),
                    fromfile='expected', tofile='actual', lineterm=''
                ))
                debug_info = f"DataFrame mismatch:\n{diff}"
                logger.error(f"Parser output does not match expected CSV:\n{debug_info}")
                return {"success": False, "error_message": "DataFrame mismatch", "debug_info": debug_info}
                
        except ImportError as e:
            logger.error(f"Import failed: {str(e)}")
            with open(parser_path, "r", encoding="utf-8") as f:
                logger.debug(f"Parser file contents:\n{f.read()}")
            return {"success": False, "error_message": f"ImportError: {str(e)}", "debug_info": f"ImportError: {str(e)}"}
        except Exception as e:
            logger.error(f"Test failed: {str(e)}")
            return {"success": False, "error_message": str(e), "debug_info": f"Test error: {str(e)}"}
    
    async def fix_code(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"Fixing parser code for {state['target_bank']}")
        
        fix_prompt = f"""
        The parser for {state['target_bank']} failed with error: {state['error_message']}
        Debug info: {state.get('debug_info', 'No debug info available')}
        Current code:
        {state['parser_code']}
        
        Suggest fixes to:
        1. Correctly parse {state['pdf_path']}
        2. Match the schema in {state['csv_path']}
        3. Handle the specific error encountered
        4. Address DataFrame mismatches if any
        5. Ensure the output is pure Python code without markdown or code block indicators (```)
        """
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content(fix_prompt))
            fixed_code = response.text.strip()
            lines = fixed_code.splitlines()
            cleaned_code = [line for line in lines if not line.strip().startswith("```") and line.strip()]
            fixed_code = "\n".join(cleaned_code)
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
        logger.error("Required PDF or CSV file not found")
        return
    
    success = asyncio.run(agent.run(args.target, pdf_path, csv_path))
    if success:
        logger.info(f"Successfully generated parser for {args.target}")
    else:
        logger.error(f"Failed to generate parser for {args.target} after max attempts")

if __name__ == "__main__":
    main()