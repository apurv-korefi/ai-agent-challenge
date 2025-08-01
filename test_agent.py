#!/usr/bin/env python3
"""
Test suite for the Bank Statement Agent

This file contains pytest tests to verify the agent's functionality.
"""

import pytest
import pandas as pd
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))

from agent import BankStatementAgent


class TestBankStatementAgent:
    """Test class for BankStatementAgent functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    return BankStatementAgent()
    
    @pytest.fixture
    def sample_plan(self):
        """Create sample plan data for testing."""
        return {
            'target_bank': 'icici',
            'schema': {
                'columns': ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'],
                'sample_data': [
                    {'Date': '01-08-2024', 'Description': 'Test', 'Debit Amt': 100.0, 'Credit Amt': '', 'Balance': 900.0}
                ]
            },
            'pdf_sample': 'Sample PDF text...',
            'strategy': 'Test parsing strategy',
            'pdf_path': 'data/icici/icici sample.pdf',
            'csv_path': 'data/icici/result.csv'
        }
    
    def test_agent_initialization_with_api_key(self):
        """Test agent initialization with API key."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    agent = BankStatementAgent()
                    assert agent.api_key == 'test_key'
                    assert agent.max_attempts == 3
    
    def test_agent_initialization_without_api_key(self):
        """Test agent initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
                BankStatementAgent()
    
    def test_extract_pdf_text_file_not_found(self, mock_agent):
        """Test PDF text extraction with non-existent file."""
        result = mock_agent._extract_pdf_text('nonexistent.pdf')
        assert result == ""
    
    def test_plan_phase(self, mock_agent, sample_plan):
        """Test the planning phase functionality."""
        # Mock the CSV file reading
        sample_df = pd.DataFrame([
            {'Date': '01-08-2024', 'Description': 'Test', 'Debit Amt': 100.0, 'Credit Amt': '', 'Balance': 900.0}
        ])
        
        # Mock the PDF text extraction
        with patch.object(mock_agent, '_extract_pdf_text', return_value='Sample PDF text'):
            with patch('pandas.read_csv', return_value=sample_df):
                with patch.object(mock_agent.model, 'generate_content') as mock_generate:
                    mock_generate.return_value.text = 'Test strategy'
                    
                    result = mock_agent.plan('icici', 'test.pdf', 'test.csv')
                    
                    assert result['target_bank'] == 'icici'
                    assert 'schema' in result
                    assert 'Date' in result['schema']['columns']
    
    def test_generate_parser(self, mock_agent, sample_plan):
        """Test parser code generation."""
        with patch.object(mock_agent.model, 'generate_content') as mock_generate:
            mock_generate.return_value.text = 'def parse(pdf_path): return pd.DataFrame()'
            
            result = mock_agent.generate_parser(sample_plan)
            
            assert 'def parse(pdf_path)' in result
            mock_generate.assert_called_once()
    
    def test_successful_parser_test(self, mock_agent, sample_plan):
        """Test successful parser testing."""
        # Create a simple working parser
        parser_code = '''
import pandas as pd

def parse(pdf_path):
    return pd.DataFrame([
        {'Date': '01-08-2024', 'Description': 'Test', 'Debit Amt': 100.0, 'Credit Amt': '', 'Balance': 900.0}
    ])
'''
        
        # Mock the expected CSV data
        expected_df = pd.DataFrame([
            {'Date': '01-08-2024', 'Description': 'Test', 'Debit Amt': 100.0, 'Credit Amt': '', 'Balance': 900.0}
        ])
        
        with patch('pandas.read_csv', return_value=expected_df):
            success, error_msg, result_df = mock_agent.test_parser(parser_code, sample_plan)
            
            assert success is True
            assert "passed" in error_msg.lower()
            assert result_df is not None
    
    def test_failed_parser_test_shape_mismatch(self, mock_agent, sample_plan):
        """Test parser testing with shape mismatch."""
        # Create a parser that returns wrong shape
        parser_code = '''
import pandas as pd

def parse(pdf_path):
    return pd.DataFrame([
        {'Date': '01-08-2024', 'Description': 'Test'}  # Missing columns
    ])
'''
        
        # Mock the expected CSV data
        expected_df = pd.DataFrame([
            {'Date': '01-08-2024', 'Description': 'Test', 'Debit Amt': 100.0, 'Credit Amt': '', 'Balance': 900.0}
        ])
        
        with patch('pandas.read_csv', return_value=expected_df):
            success, error_msg, result_df = mock_agent.test_parser(parser_code, sample_plan)
            
            assert success is False
            assert "shape mismatch" in error_msg.lower()
    
    def test_refine_parser(self, mock_agent, sample_plan):
        """Test parser refinement functionality."""
        parser_code = 'def parse(pdf_path): pass'
        error_message = 'Shape mismatch'
        
        with patch.object(mock_agent.model, 'generate_content') as mock_generate:
            mock_generate.return_value.text = 'def parse(pdf_path): return pd.DataFrame()'
            
            result = mock_agent.refine_parser(parser_code, error_message, sample_plan, 2)
            
            assert 'def parse(pdf_path)' in result
            mock_generate.assert_called_once()


def test_integration_with_real_files():
    """Integration test that checks if required files exist."""
    project_root = Path(__file__).parent
    
    # Check if required data files exist
    icici_dir = project_root / "data" / "icici"
    assert icici_dir.exists(), "ICICI data directory should exist"
    
    pdf_file = icici_dir / "icici sample.pdf"
    csv_file = icici_dir / "result.csv"
    
    assert pdf_file.exists(), f"PDF file should exist at {pdf_file}"
    assert csv_file.exists(), f"CSV file should exist at {csv_file}"
    
    df = pd.read_csv(csv_file)
    expected_columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']
    
    assert list(df.columns) == expected_columns, f"CSV should have columns {expected_columns}"
    assert len(df) > 0, "CSV should contain data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])