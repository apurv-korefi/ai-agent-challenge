#  AI Agent Karbon - Intelligent Bank Statement Parser Generator

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

> **An autonomous AI agent that automatically generates custom parsers for bank statement PDFs using Google's Gemini AI**

##  Overview

AI Agent Karbon is an intelligent system that revolutionizes the way we handle bank statement parsing. Instead of manually writing parsers for each bank's unique PDF format, this agent automatically analyzes sample data and generates custom parsers through an iterative plan → generate → test → refine loop.

###  Key Features
 
-  Autonomous Parser Generation: Automatically creates custom parsers for any bank's PDF format
-  Intelligent Analysis: Analyzes both PDF structure and expected CSV output
-  Iterative Refinement: Continuously improves parsers through testing and feedback
-  Multi-Bank Support: Designed to work with ICICI, HDFC, SBI, and other banks
-  Self-Learning: Uses LLM feedback to enhance parsing accuracy

##  Technology Stack

- **Python 3.8+**
- **Google Gemini AI** - For intelligent code generation
- **Pandas** - Data manipulation and analysis
- **PyPDF2** - PDF text extraction
- **Pytest** - Testing framework

##  Installation

### Prerequisites

1. **Python 3.8 or higher**
2. **Google AI Studio API Key** - Get yours at [aistudio.google.com](https://aistudio.google.com/)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Agent-Karbon.git
   cd AI-Agent-Karbon
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key**
   
   **Option A: Environment Variable**
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY = "your-api-key-here"
   
   # Windows Command Prompt
   set GEMINI_API_KEY=your-api-key-here
   
   # Linux/Mac
   export GEMINI_API_KEY=your-api-key-here
   ```
   
   **Option B: .env File**
   ```bash
   echo "GEMINI_API_KEY=your-api-key-here" > .env
   ```

##  Usage

### Quick Start

```python
from agent import BankStatementAgent

# Initialize the agent
agent = BankStatementAgent()

# Generate a parser for ICICI bank
success = agent.run('icici')

if success:
    print("✅ Parser generated successfully!")
else:
    print("❌ Parser generation failed")
```

### Command Line Interface

```bash
# Run the agent for a specific bank
python agent.py --bank icici

# Run the demo
python demo.py

# Run tests
python test_agent.py
```

##  Project Structure

```
AI-Agent-Karbon/
├── agent.py                 # Main AI agent implementation
├── demo.py                  # Demo script
├── test_agent.py           # Test suite
├── requirements.txt         # Python dependencies
├── .env                    # Environment variables (create this)
├── README.md              # This file
├── custom_parsers/        # Generated parsers
│   └── icici_parser.py   # ICICI bank parser
└── data/                  # Sample data
    └── icici/
        ├── icici sample.pdf
        └── result.csv
```

##  How It Works

### 1. **Planning Phase** 
- Analyzes the target PDF structure
- Examines expected CSV output format
- Creates a parsing strategy using Gemini AI

### 2. **Generation Phase** 
- Generates Python parser code based on the strategy
- Implements regex patterns and data extraction logic
- Handles edge cases and data cleaning

### 3. **Testing Phase** 
- Executes the generated parser
- Compares output with expected results
- Validates data accuracy and format

### 4. **Refinement Phase** 
- Uses error feedback to improve the parser
- Iteratively refines until success criteria are met
- Maximum 3 attempts per bank

##  Demo
https://github.com/user-attachments/assets/d7bcacee-f303-443d-a9ed-6c3fef674945

##  Supported Banks

| Bank | Status | Parser Location |
|------|--------|-----------------|
| ICICI | ✅ Complete | `custom_parsers/icici_parser.py` |
| HDFC | 🚧 In Progress | - |
| SBI | 🚧 In Progress | - |
| Axis | 🚧 In Progress | - |

##  Testing

Run the test suite to verify everything works:

```bash
python test_agent.py
```

##  Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests for new functionality
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

##  Acknowledgments

- **Google Gemini AI** - For providing the intelligent code generation capabilities
- **Open Source Community** - For the amazing libraries that make this possible
- **Bank Statement Data** - For providing real-world test cases






