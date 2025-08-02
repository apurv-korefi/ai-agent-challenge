# Bank Statement Parser Agent

## Overview
This project implements an autonomous agent for generating and refining Python parsers to extract transaction data from bank statement PDFs. The agent uses the LangGraph framework to manage a workflow that plans, generates, tests, and fixes parser code, leveraging the Gemini 1.5 Pro model for code generation.

## Features
- **Automated Parser Generation**: Generates Python code to parse bank statement PDFs into pandas DataFrames.
- **Schema Matching**: Ensures parsed data matches the expected schema defined in a reference CSV file.
- **Error Handling and Recovery**: Iteratively fixes parser code based on test failures, with a maximum of three attempts.
- **Logging**: Comprehensive logging for debugging and monitoring the parsing process.
- **Modular Workflow**: Uses LangGraph to orchestrate planning, code generation, testing, and fixing stages.

## Prerequisites
- Python 3.8+
- Required Python packages:
  ```bash
  pip install pandas pdfplumber langgraph google-generativeai python-dotenv
  ```
- A valid Gemini API key from [Google AI](https://ai.google.dev).
- Input files:
  - A sample bank statement PDF (e.g., `data/icici/icici_sample.pdf`).
  - A reference CSV file defining the expected schema (e.g., `data/icici/icici_sample.csv`).

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up Environment Variables**:
   Create a `.env` file in the project root and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your-api-key-here
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Input Files**:
   Place your bank statement PDF and reference CSV in the `data/<bank_name>/` directory, e.g., `data/icici/icici_sample.pdf` and `data/icici/icici_sample.csv`.

## Usage
Run the agent using the command-line interface:
```bash
python agent.py --target <bank_name>
```
- Replace `<bank_name>` with the name of the bank (e.g., `icici`).
- Ensure the corresponding PDF and CSV files exist in `data/<bank_name>/`.

The agent will:
1. Plan the parser generation.
2. Generate initial parser code.
3. Test the parser against the reference CSV.
4. Fix the code if necessary, up to three attempts.

## Output
- A parser script is saved in `custom_parser/<bank_name>_parser.py`.
- Success or failure is logged to the console.
- Detailed logs are generated for debugging, including any schema mismatches or parsing errors.

## Project Structure
- `agent.py`: Main script containing the `BankStatementParserAgent` class and workflow logic.
- `custom_parser/`: Directory where generated parser scripts are saved.
- `data/<bank_name>/`: Directory for input PDF and CSV files.
- `.env`: Environment file for storing the Gemini API key.

## Example
To generate a parser for an ICICI bank statement:
```bash
python agent.py --target icici
```
Ensure `data/icici/icici_sample.pdf` and `data/icici/icici_sample.csv` exist.

## Logging
Logs are output to the console with timestamps and levels (INFO, ERROR, DEBUG). Example:
```
2025-08-02 14:45:00,123 - INFO - Successfully generated parser for icici
```

## Limitations
- The agent assumes a consistent PDF format for a given bank.
- Maximum of three attempts to fix parser code.
- Requires a valid Gemini API key and internet access.
- Limited to PDF parsing with pdfplumber and schema validation against a provided CSV.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for bugs, feature requests, or improvements.

## License
This project is licensed under the MIT License.