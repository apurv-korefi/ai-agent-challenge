AI Agent Bank Statement Parser
==============================

An autonomous coding agent that writes custom PDF parsers for bank statements. The agent analyzes PDF structure, generates parsing code, runs tests, and self-corrects until it produces a working parser.

How It Works
------------

The agent follows a simple but powerful loop:

1.  **Plan** - Analyzes the PDF structure and CSV schema to understand what needs to be extracted
    
2.  **Generate** - Writes Python parsing code using pdfplumber to extract transactions
    
3.  **Test** - Runs automated tests to verify the parser output matches expected CSV format
    
4.  **Self-Fix** - If tests fail, analyzes errors and rewrites code (up to 3 attempts)
    

The magic happens in the self-correction cycle - when tests fail, the agent examines the error messages and automatically fixes issues like header detection, data type mismatches, or debit/credit logic problems.

Quick Start
-----------

### 1\. Clone and Setup

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   git clone   cd ai-agent-challenge  pip install -r requirements.txt   `

### 2\. Set API Key

Create a .env file with your API key:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   GOOGLE_API_KEY=your_gemini_api_key_here   `

### 3\. Run the Agent

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python agent.py --target icici   `

### 4\. Watch It Work

The agent will:

*   Read data/icici/icici sample.pdf
    
*   Analyze the expected output format from data/icici/result.csv
    
*   Generate custom\_parser/icici\_parser.py
    
*   Test and fix until it works
    

### 5\. Verify Results

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pytest tests/test_icici_parser.py -v   `

Agent Architecture
------------------

The system uses **LangGraph** to orchestrate the agent workflow:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   START → Plan → Generate Code → Run Tests → [PASS] → END                      ↑              ↓                      ←── Self-Fix ←──┘ [FAIL & attempts < 3]   `

**Key Components:**

*   **Planner Node**: Analyzes PDF/CSV structure to create parsing strategy
    
*   **Code Generator**: Uses LLM to write parsing logic with specific prompts for bank statement patterns
    
*   **Test Runner**: Executes pytest to validate parser output against expected CSV
    
*   **Self-Fixer**: Examines test failures and regenerates improved code automatically
    

Adding New Banks
----------------

To support a new bank (e.g., SBI):

1.  Create the data structure:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   data/sbi/  ├── sbi sample.pdf  └── result.csv   `

1.  Run the agent:
    

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   python agent.py --target sbi   `

The agent will automatically adapt its parsing strategy to the new bank's PDF format and generate a working parser.

Technical Details
-----------------

*   **PDF Processing**: Uses pdfplumber for reliable text extraction
    
*   **Data Processing**: Pandas for DataFrame operations and CSV matching
    
*   **Testing**: Automated pytest validation ensuring output matches expected format
    
*   **Error Handling**: Robust parsing with header detection, data validation, and type conversion
    
*   **Memory**: Maintains conversation context for iterative improvements
    

The agent handles common parsing challenges like varying header formats, date parsing, debit/credit detection, and numerical data validation automatically through its self-correction mechanism.

_Built for the Karbon AI Challenge - creating autonomous coding agents that actually work._