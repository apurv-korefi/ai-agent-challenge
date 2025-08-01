#!/usr/bin/env python3
"""
Demo script to showcase the AI Agent functionality without requiring API keys.

This creates a mock demonstration of how the agent would work.
"""

import os
import time
import sys
from pathlib import Path

def simulate_agent_run():
    """Simulate the agent running through its phases."""
    
    print("🤖 AI Agent for Bank Statement PDF Parser")
    print("=" * 50)
    print()
    
    # Phase 1: Planning
    print("📋 PHASE 1: PLANNING")
    print("- Analyzing ICICI sample PDF...")
    time.sleep(1)
    print("- Extracting CSV schema: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']")
    time.sleep(1)
    print("- Identifying parsing patterns in PDF structure...")
    time.sleep(1)
    print("✅ Planning complete!")
    print()
    
    # Phase 2: Code Generation
    print("🔧 PHASE 2: CODE GENERATION")
    print("- Generating parser code using LLM...")
    time.sleep(2)
    print("- Creating parse(pdf_path) -> pd.DataFrame function...")
    time.sleep(1)
    print("- Adding error handling and data validation...")
    time.sleep(1)
    print("✅ Parser code generated!")
    print()
    
    # Phase 3: Testing
    print("🧪 PHASE 3: TESTING")
    print("- Executing generated parser on sample PDF...")
    time.sleep(1)
    print("- Comparing output with expected CSV...")
    time.sleep(1)
    print("- Validating schema and data types...")
    time.sleep(1)
    
    # Simulate a failure and retry
    print("❌ Test failed: Column type mismatch")
    print()
    
    # Phase 4: Self-Correction
    print("🔄 PHASE 4: SELF-CORRECTION (Attempt 2/3)")
    print("- Analyzing test failure...")
    time.sleep(1)
    print("- Refining parser to fix type conversion issues...")
    time.sleep(2)
    print("- Re-testing improved parser...")
    time.sleep(1)
    print("✅ Test passed! Parser validated successfully!")
    print()
    
    # Success
    print("🎉 SUCCESS!")
    print(f"- Parser saved to: custom_parsers/icici_parser.py")
    print("- Agent completed autonomous generation cycle")
    print("- Ready for production use!")
    print()
    
    return True

def show_project_structure():
    """Display the current project structure."""
    print("📁 PROJECT STRUCTURE")
    print("=" * 30)
    
    project_root = Path(".")
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
            
        items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
        for i, item in enumerate(items[:10]):  # Limit to first 10 items
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            print(f"{prefix}{current_prefix}{item.name}")
            
            if item.is_dir() and not item.name.startswith('.'):
                next_prefix = prefix + ("    " if is_last else "│   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(project_root)
    print()

def main():
    """Main demo function."""
    print("🚀 AI Agent Challenge Demo")
    print("This demo simulates the agent's autonomous operation")
    print()
    
    # Show project structure
    show_project_structure()
    
    # Show what files exist
    icici_dir = Path("data/icici")
    if icici_dir.exists():
        print("✅ Found ICICI sample data:")
        for file in icici_dir.iterdir():
            print(f"   - {file.name}")
        print()
    else:
        print("❌ ICICI sample data not found")
        print("   Please ensure data/icici/ directory exists with sample files")
        return
    
    # Run simulation
    input("Press Enter to start agent simulation...")
    print()
    
    success = simulate_agent_run()
    
    if success:
        print("Demo completed! In a real run with API key:")
        print("$ export GEMINI_API_KEY='your-key'")
        print("$ python agent.py --target icici")
        print()
        print("The agent would autonomously:")
        print("1. Plan the parsing strategy")
        print("2. Generate working Python code")
        print("3. Test against sample data")
        print("4. Self-correct any issues")
        print("5. Save the final parser")

if __name__ == "__main__":
    main()