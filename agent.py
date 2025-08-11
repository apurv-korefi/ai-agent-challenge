#!/usr/bin/env python3
"""
PDF Parser Generation Agent
Automatically generates custom parsers for bank statement PDFs using LangGraph.
"""

import os
import sys
import argparse
import pandas as pd
import PyPDF2
import pdfplumber
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor


@dataclass
class ParserState:
    """State object for the parser generation workflow"""
    pdf_path: str
    target_bank: str
    extracted_text: str = ""
    sample_data: List[Dict] = None
    parser_code: str = ""
    test_results: Dict = None
    iteration: int = 0
    max_iterations: int = 3
    error_message: str = ""


class PDFParserAgent:
    """Main agent class for generating PDF parsers"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the agent with OpenAI API key"""
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for parser generation"""
        workflow = StateGraph(ParserState)
        
        # Add nodes
        workflow.add_node("extract_text", self._extract_text_node)
        workflow.add_node("analyze_structure", self._analyze_structure_node)
        workflow.add_node("generate_parser", self._generate_parser_node)
        workflow.add_node("test_parser", self._test_parser_node)
        workflow.add_node("self_fix", self._self_fix_node)
        
        # Define the workflow edges
        workflow.set_entry_point("extract_text")
        workflow.add_edge("extract_text", "analyze_structure")
        workflow.add_edge("analyze_structure", "generate_parser")
        workflow.add_edge("generate_parser", "test_parser")
        
        # Conditional edge for self-fixing
        workflow.add_conditional_edges(
            "test_parser",
            self._should_continue_fixing,
            {
                "continue": "self_fix",
                "end": END
            }
        )
        workflow.add_edge("self_fix", "generate_parser")
        
        return workflow.compile()
    
    def _extract_text_node(self, state: ParserState) -> ParserState:
        """Extract text from PDF using multiple methods"""
        print(f"📄 Extracting text from {state.pdf_path}...")
        
        try:
            # Try pdfplumber first (better for tables)
            with pdfplumber.open(state.pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            
            if not text.strip():
                # Fallback to PyPDF2
                with open(state.pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            
            state.extracted_text = text
            print("✅ Text extraction completed")
            
        except Exception as e:
            state.error_message = f"Failed to extract text: {str(e)}"
            print(f"❌ {state.error_message}")
        
        return state
    
    def _analyze_structure_node(self, state: ParserState) -> ParserState:
        """Analyze PDF structure and identify transaction patterns"""
        print("🔍 Analyzing PDF structure and patterns...")
        
        prompt = f"""
        Analyze this bank statement text and identify the transaction structure:
        
        Bank: {state.target_bank}
        
        Text sample (first 2000 chars):
        {state.extracted_text[:2000]}
        
        Please identify:
        1. Transaction line patterns (regex patterns)
        2. Date formats used
        3. Column structure (date, description, debit, credit, balance)
        4. Header/footer patterns to ignore
        5. Any special formatting or delimiters
        
        Provide your analysis in JSON format with clear patterns and examples.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content="You are a PDF structure analysis expert."), 
                                      HumanMessage(content=prompt)])
            
            # Extract sample transactions for testing
            sample_transactions = self._extract_sample_transactions(state.extracted_text)
            state.sample_data = sample_transactions
            
            print("✅ Structure analysis completed")
            
        except Exception as e:
            state.error_message = f"Failed to analyze structure: {str(e)}"
            print(f"❌ {state.error_message}")
        
        return state
    
    def _generate_parser_node(self, state: ParserState) -> ParserState:
        """Generate the parser code based on analysis"""
        print("🛠️  Generating parser code...")
        
        template_code = self._get_parser_template()
        
        prompt = f"""
        Generate a complete Python parser for {state.target_bank} bank statements.
        
        Use this template structure:
        {template_code}
        
        Based on this text analysis:
        Text sample: {state.extracted_text[:1500]}
        
        Requirements:
        1. Return a pandas DataFrame with columns: date, description, debit, credit, balance
        2. Handle different date formats
        3. Clean and normalize transaction descriptions
        4. Convert amounts to float (handle commas, currency symbols)
        5. Skip header/footer lines
        6. Handle edge cases gracefully
        
        Error from previous attempt (if any): {state.error_message}
        
        Generate ONLY the complete Python code for the parse() function.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert Python developer specializing in PDF parsing."),
                HumanMessage(content=prompt)
            ])
            
            state.parser_code = response.content
            print("✅ Parser code generated")
            
        except Exception as e:
            state.error_message = f"Failed to generate parser: {str(e)}"
            print(f"❌ {state.error_message}")
        
        return state
    
    def _test_parser_node(self, state: ParserState) -> ParserState:
        """Test the generated parser code"""
        print("🧪 Testing parser code...")
        
        try:
            # Save parser code to temporary file
            parser_file = f"temp_{state.target_bank.lower()}_parser.py"
            
            with open(parser_file, 'w') as f:
                f.write(state.parser_code)
            
            # Import and test the parser
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_parser", parser_file)
            temp_parser = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_parser)
            
            # Test parsing
            result_df = temp_parser.parse(state.pdf_path)
            
            # Validate results
            if isinstance(result_df, pd.DataFrame) and len(result_df) > 0:
                required_cols = ['date', 'description', 'debit', 'credit', 'balance']
                if all(col in result_df.columns for col in required_cols):
                    state.test_results = {
                        "success": True,
                        "rows": len(result_df),
                        "columns": list(result_df.columns),
                        "sample": result_df.head().to_dict()
                    }
                    print(f"✅ Parser test successful! Parsed {len(result_df)} transactions")
                else:
                    state.test_results = {"success": False, "error": "Missing required columns"}
                    print("❌ Parser test failed: Missing required columns")
            else:
                state.test_results = {"success": False, "error": "No data parsed or invalid format"}
                print("❌ Parser test failed: No data parsed")
            
            # Cleanup
            os.remove(parser_file)
            
        except Exception as e:
            state.test_results = {"success": False, "error": str(e)}
            state.error_message = str(e)
            print(f"❌ Parser test failed: {str(e)}")
        
        return state
    
    def _self_fix_node(self, state: ParserState) -> ParserState:
        """Self-fix the parser based on test results"""
        print(f"🔧 Self-fixing parser (attempt {state.iteration + 1}/{state.max_iterations})...")
        
        state.iteration += 1
        
        fix_prompt = f"""
        The parser code failed with this error: {state.test_results.get('error', state.error_message)}
        
        Current parser code:
        {state.parser_code}
        
        Please fix the code to handle this error. Common issues:
        1. Regex patterns not matching the actual text format
        2. Date parsing errors
        3. Amount conversion issues (commas, currency symbols)
        4. Column name mismatches
        5. Empty or malformed data handling
        
        Return the complete corrected Python code.
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are debugging and fixing Python PDF parser code."),
                HumanMessage(content=fix_prompt)
            ])
            
            state.parser_code = response.content
            print("✅ Parser code fixed")
            
        except Exception as e:
            state.error_message = f"Failed to fix parser: {str(e)}"
            print(f"❌ {state.error_message}")
        
        return state
    
    def _should_continue_fixing(self, state: ParserState) -> str:
        """Decide whether to continue fixing or end"""
        if (state.test_results and 
            state.test_results.get("success", False)):
            return "end"
        
        if state.iteration >= state.max_iterations:
            print(f"⚠️  Max iterations ({state.max_iterations}) reached")
            return "end"
        
        return "continue"
    
    def _extract_sample_transactions(self, text: str) -> List[Dict]:
        """Extract sample transactions for validation"""
        # Basic pattern matching for common transaction formats
        lines = text.split('\n')
        transactions = []
        
        for line in lines:
            # Skip empty lines and headers
            if not line.strip() or any(header in line.upper() for header in 
                                     ['DATE', 'DESCRIPTION', 'BALANCE', 'STATEMENT']):
                continue
            
            # Look for lines with date patterns and amounts
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
            amount_pattern = r'\d+[,\d]*\.?\d*'
            
            if re.search(date_pattern, line) and re.search(amount_pattern, line):
                transactions.append({"raw_line": line.strip()})
                if len(transactions) >= 5:  # Sample first 5 transactions
                    break
        
        return transactions
    
    def _get_parser_template(self) -> str:
        """Get the parser code template"""
        return '''
import pandas as pd
import re
from datetime import datetime
import pdfplumber

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parse bank statement PDF and return DataFrame with columns:
    date, description, debit, credit, balance
    """
    # Your implementation here
    pass
'''
    
    def generate_parser(self, pdf_path: str, target_bank: str) -> str:
        """Main method to generate parser"""
        print(f"🚀 Starting parser generation for {target_bank}...")
        
        # Initialize state
        initial_state = ParserState(
            pdf_path=pdf_path,
            target_bank=target_bank
        )
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        if final_state.test_results and final_state.test_results.get("success"):
            # Save the final parser
            parser_filename = f"custom_parsers/{target_bank.lower()}_parser.py"
            os.makedirs("custom_parsers", exist_ok=True)
            
            with open(parser_filename, 'w') as f:
                f.write(final_state.parser_code)
            
            print(f"✅ Parser successfully generated: {parser_filename}")
            return parser_filename
        else:
            print("❌ Failed to generate working parser")
            return None


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="PDF Parser Generation Agent")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., 'icici', 'sbi')")
    parser.add_argument("--pdf", required=True, help="Path to sample PDF file")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pdf):
        print(f"❌ PDF file not found: {args.pdf}")
        sys.exit(1)
    
    # Initialize agent
    agent = PDFParserAgent(openai_api_key=args.api_key)
    
    # Generate parser
    result = agent.generate_parser(args.pdf, args.target)
    
    if result:
        print(f"\n🎉 Success! Parser generated at: {result}")
        print(f"\nTo test your parser:")
        print(f"python -c \"from {result.replace('.py', '').replace('/', '.')} import parse; print(parse('{args.pdf}'))\"")
    else:
        print("\n💥 Failed to generate parser. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()