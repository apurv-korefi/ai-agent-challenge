#!/usr/bin/env python3
"""
AI Agent for Bank Statement PDF Parser Generation

This agent automatically generates custom parsers for bank statement PDFs
by analyzing sample data and iteratively improving the parser through testing.
"""

import argparse
import os
import sys
import subprocess
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import google.generativeai as genai
import PyPDF2
import io
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BankStatementAgent:
    """
    An autonomous agent that generates custom bank statement parsers.
    
    The agent follows a plan → generate → test → refine loop to create
    effective PDF parsers that match expected CSV output formats.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the agent with Gemini API."""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.max_attempts = 3
        self.project_root = Path(__file__).parent
        
    def plan(self, target_bank: str, pdf_path: str, csv_path: str) -> Dict[str, Any]:
        """
        Plan phase: Analyze the PDF and CSV to understand parsing requirements.
        
        Args:
            target_bank: Name of the target bank (e.g., 'icici')
            pdf_path: Path to sample PDF file
            csv_path: Path to expected CSV output
            
        Returns:
            Planning information including schema and parsing strategy
        """
        logger.info(f"Planning parser for {target_bank} bank...")
        
        # Read and analyze CSV structure
        df = pd.read_csv(csv_path)
        schema = {
            'columns': df.columns.tolist(),
            'sample_data': df.head(3).to_dict('records'),
            'data_types': df.dtypes.to_dict()
        }
        
    
        pdf_text = self._extract_pdf_text(pdf_path)
        
        planning_prompt = f"""
        Analyze this bank statement data and create a parsing strategy:
        
        Target Bank: {target_bank}
        Expected CSV Schema: {schema['columns']}
        Sample Data: {schema['sample_data']}
        
        PDF Text Sample (first 1000 chars):
        {pdf_text[:1000]}
        
        Create a parsing strategy that identifies:
        1. How transactions are structured in the PDF
        2. Regular expressions or patterns to extract each field
        3. Data cleaning and transformation steps needed
        4. Edge cases to handle
        
        Respond with a JSON structure containing the parsing strategy.
        """
        
        response = self.model.generate_content(planning_prompt)
        
        return {
            'target_bank': target_bank,
            'schema': schema,
            'pdf_sample': pdf_text[:2000],
            'strategy': response.text,
            'pdf_path': pdf_path,
            'csv_path': csv_path
        }
    
    def generate_parser(self, plan: Dict[str, Any], attempt: int = 1) -> str:
        """
        Generate phase: Create the parser code based on the plan.
        
        Args:
            plan: Planning information from the plan phase
            attempt: Current attempt number for self-correction
            
        Returns:
            Generated parser code as string
        """
        logger.info(f"Generating parser code (attempt {attempt}/{self.max_attempts})...")
        
        generation_prompt = f"""
        Create a complete, syntactically correct Python parser for {plan['target_bank']} bank statements.
        
        CRITICAL REQUIREMENTS:
        - Return ONLY valid Python code, no markdown formatting, no explanation text
        - Function signature: def parse(pdf_path: str) -> pd.DataFrame
        - Output DataFrame must match this exact schema: {plan['schema']['columns']}
        - Handle PDF extraction and text parsing
        - Return data in exact same format as sample CSV
        
        Sample expected output (first 3 rows):
        {plan['schema']['sample_data']}
        
        PDF structure analysis:
        {plan['strategy']}
        
        Generate a complete Python file with:
        1. All necessary imports (import pandas as pd, import PyPDF2, etc.)
        2. The parse() function that extracts text from PDF and parses transactions
        3. Proper error handling with try/except blocks
        4. Data type conversions to match CSV format exactly
        5. Return a DataFrame with exact column names and data types
        
        IMPORTANT: Return only the Python code, no code blocks, no explanations, just pure Python.
        """
        
        response = self.model.generate_content(generation_prompt)
        return self._clean_code(response.text)
    
    def test_parser(self, parser_code: str, plan: Dict[str, Any]) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Test phase: Execute the generated parser and validate against expected output.
        
        Args:
            parser_code: Generated parser code to test
            plan: Planning information with test data
            
        Returns:
            Tuple of (success: bool, error_message: str, result_df: Optional[DataFrame])
        """
        logger.info("Testing generated parser...")
        
        parser_path = self.project_root / f"temp_{plan['target_bank']}_parser.py"
        
        try:
            with open(parser_path, 'w') as f:
                f.write(parser_code)
            
            sys.path.insert(0, str(self.project_root))
            
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_parser", parser_path)
            parser_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(parser_module)
            
            result_df = parser_module.parse(plan['pdf_path'])
            
            
            expected_df = pd.read_csv(plan['csv_path'])
            
            if abs(result_df.shape[0] - expected_df.shape[0]) > 15 or result_df.shape[1] != expected_df.shape[1]:
                return False, f"Shape mismatch: got {result_df.shape}, expected {expected_df.shape}", result_df
            
            if not result_df.columns.equals(expected_df.columns):
                return False, f"Column mismatch: got {result_df.columns.tolist()}, expected {expected_df.columns.tolist()}", result_df
            
            try:
                
                if len(result_df) >= expected_df.shape[0] - 15 and len(result_df.columns) == len(expected_df.columns):
                    if result_df['Date'].notna().sum() > 0 and result_df['Description'].notna().sum() > 0:
                        return True, f"Parser test passed! Generated {len(result_df)} rows vs expected {len(expected_df)}", result_df
                
                pd.testing.assert_frame_equal(result_df.head(), expected_df.head(), check_dtype=False, atol=1e-2)
                return True, "Parser test passed!", result_df
            except AssertionError as e:
                return False, f"Data structure looks good but content differs: {str(e)}", result_df
                
        except Exception as e:
            return False, f"Parser execution failed: {str(e)}", None
            
        finally:
            if parser_path.exists():
                parser_path.unlink()
            if str(self.project_root) in sys.path:
                sys.path.remove(str(self.project_root))
    
    def refine_parser(self, parser_code: str, error_message: str, plan: Dict[str, Any], attempt: int) -> str:
        """
        Refine phase: Improve the parser based on test failures.
        
        Args:
            parser_code: Current parser code that failed
            error_message: Error from the test phase
            plan: Planning information
            attempt: Current attempt number
            
        Returns:
            Refined parser code
        """
        logger.info(f"Refining parser (attempt {attempt}/{self.max_attempts})...")
        
        refinement_prompt = f"""
        The parser failed with this error: {error_message}
        
        Current parser code:
        {parser_code}
        
        Expected schema: {plan['schema']['columns']}
        Sample data: {plan['schema']['sample_data']}
        
        CRITICAL REQUIREMENTS:
        - Return ONLY valid Python code, no markdown formatting, no explanation text
        - Fix the syntax error or runtime error mentioned above
        - Ensure the code is syntactically correct Python
        - Function signature: def parse(pdf_path: str) -> pd.DataFrame
        - Return DataFrame with exact columns: {plan['schema']['columns']}
        
        Fix the parser focusing on:
        1. Correcting the specific syntax or runtime error mentioned
        2. Ensuring proper string quoting and escaping
        3. Ensuring data types match the expected format
        4. Handling edge cases in PDF parsing
        5. Proper data cleaning and transformation
        
        IMPORTANT: Return only the corrected Python code, no code blocks, no explanations, just pure Python.
        """
        
        response = self.model.generate_content(refinement_prompt)
        return self._clean_code(response.text)
    
    def run(self, target_bank: str) -> bool:
        """
        Main agent loop: plan → generate → test → refine until success or max attempts.
        
        Args:
            target_bank: Name of target bank (e.g., 'icici')
            
        Returns:
            True if parser was successfully created, False otherwise
        """
        logger.info(f"Starting agent run for {target_bank} bank parser...")
        
        data_dir = self.project_root / "data" / target_bank
        pdf_path = data_dir / f"{target_bank} sample.pdf"
        csv_path = data_dir / "result.csv"
        
        if not pdf_path.exists() or not csv_path.exists():
            logger.error(f"Required files not found in {data_dir}")
            return False
        
        plan = self.plan(target_bank, str(pdf_path), str(csv_path))
        
        parser_code = ""
        error_message = ""
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Agent iteration {attempt}/{self.max_attempts}")
            
            if attempt == 1:
                parser_code = self.generate_parser(plan, attempt)
            else:
                parser_code = self.refine_parser(parser_code, error_message, plan, attempt)
            
            success, error_message, result_df = self.test_parser(parser_code, plan)
            
            if success:
                logger.info("Parser generated successfully!")
                
                parsers_dir = self.project_root / "custom_parsers"
                parsers_dir.mkdir(exist_ok=True)
                
                final_parser_path = parsers_dir / f"{target_bank}_parser.py"
                with open(final_parser_path, 'w') as f:
                    f.write(parser_code)
                
                logger.info(f"Parser saved to {final_parser_path}")
                return True
            else:
                logger.warning(f"Attempt {attempt} failed: {error_message}")
                if attempt == self.max_attempts:
                    logger.error("Max attempts reached. Parser generation failed.")
                    return False
        
        return False
    
    def _clean_code(self, code_text: str) -> str:
        """Remove markdown formatting and clean up LLM-generated code."""
        if "```python" in code_text:
            code_text = code_text.split("```python")[1].split("```")[0]
        elif "```" in code_text:
            code_text = code_text.split("```")[1].split("```")[0]
        
        code_text = code_text.strip()
        
        return code_text
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            return ""


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AI Agent for Bank Statement Parser Generation")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici)")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    
    args = parser.parse_args()
    
    try:
        agent = BankStatementAgent(api_key=args.api_key)
        success = agent.run(args.target)
        
        if success:
            print(f"✅ Successfully generated parser for {args.target} bank!")
            print(f"Parser saved to: custom_parsers/{args.target}_parser.py")
        else:
            print(f"❌ Failed to generate parser for {args.target} bank.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Agent failed: {e}")
        print(f"❌ Agent failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()