# ai-agent-challenge
Coding agent challenge which write custom parsers for Bank statement PDF.
# AI-Powered Self-Fixing PDF Parser Agent

## Project Overview

This project demonstrates an **AI-powered autonomous agent** that can learn, generate code, and self-improve to solve complex PDF parsing challenges. The agent uses **Gemini AI** to dynamically generate and improve PDF parser code through iterative learning.

## 🎯 Main Goal

**The primary objective is NOT to achieve 100% perfect parsing accuracy, but to demonstrate AI learning and self-improvement capabilities.**

### What This Agent Does:
- **Generates PDF parsing code from scratch** using Gemini AI
- **Learns from test failures** and improves iteratively
- **Demonstrates true AI problem-solving** through code generation
- **Shows autonomous learning** without human intervention

### Why Tests May Fail:
- **PDF parsing is inherently complex** - even advanced AI struggles with malformed documents
- **Text extraction is imperfect** - minor differences cause test failures
- **Strict equality requirements** - `DataFrame.equals` is extremely demanding
- **The goal is learning, not perfection** - failures drive improvement

##  Project Structure

```
ai-agent-challenge/
├── agent.py              # Main AI agent orchestrator
├── custom_parsers/       # Generated parser modules
├── data/
│   └── icici/           # ICICI bank statement data
│       ├── ICICI_sample.pdf    # Malformed PDF to parse
│       └── result.csv          # Expected output (ground truth)
└── tests/                # Test suite for validation
```

### Key Files:
- **`agent.py`**: The AI agent that generates and improves parser code
- **`custom_parsers/`**: Where the agent writes generated parser code
- **`data/icici/`**: Contains the challenging PDF and expected results
- **`tests/`**: Validates the generated parser against ground truth

- ## Architecture Diagram

- <img width="2300" height="2089" alt="Flowcharts" src="https://github.com/user-attachments/assets/8a0bf635-93ac-4703-966c-cad71640f130" />

## 🔄 Workflow Flow explaining Diagram

### 1. **Initialization Phase**
```
User Command → Agent State → LangGraph Setup → First Node
```

### 2. **Code Generation Phase**
```
Gemini AI Prompt → Code Generation → File Writing → Next Node
```

### 3. **Testing Phase**
```
Generated Parser → Test Execution → Result Capture → Decision Point
```

### 4. **Learning Phase**
```
Test Failure → Error Analysis → Self-Fix → State Update → Loop Back
```

### 5. **Termination Phase**
```
Success OR Max Attempts → Final State → Output Generation → End

```
### 🚀 5-Step Run Instructions

### Step 1: Get Gemini API Key
- Visit: https://makersuite.google.com/app/apikey
- Create a new API key for Gemini AI
- Copy the key to use in Step 2

### Step 2: Set Up Environment
- Create a `.env` file in the project root
- Add your API key: `GOOGLE_API_KEY=your_api_key_here`
- Ensure Python 3.8+ is installed

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Agent
```bash
python agent.py --target <bank name ex- icici> -> python agent.py --target icici
```

### Step 5: Observe AI Learning
- Watch the agent generate code on Attempt 1
- See it learn from failures and improve on Attempt 2
- Observe advanced strategies on Attempt 3
- Celebrate the learning journey, not just the final result

## 🔍 Understanding the Results

### Success Indicators:
-  **Code Generation**: Agent creates working Python parsers
-  **Learning**: Each attempt shows improvement
-  **Problem Solving**: Sophisticated parsing strategies emerge
-  **Autonomy**: No human intervention needed

### Expected Outcomes:
- **Tests may fail** - this is normal and expected
- **Parsing accuracy improves** with each attempt
- **Code quality increases** through AI learning
- **Agent demonstrates resilience** and problem-solving

##  Example Output

```
---GENERATING CODE WITH GEMINI AI---
Gemini generated code for attempt 1
---TESTING CODE---
❌ Tests Failed!
---SELF-FIX: attempts left 1---
---GENERATING CODE WITH GEMINI AI---
Gemini generated code for attempt 2
---TESTING CODE---
✅ Tests Passed!
---AGENT FINISHED SUCCESSFULLY---
```

## 🔧 Technical Details

- **Framework**: LangGraph for workflow orchestration
- **AI Model**: Gemini 2.5 Pro for code generation
- **PDF Processing**: pypdf for text extraction
- **Testing**: pytest for validation
- **State Management**: TypedDict for agent state

##  Important Notes

- **This is a learning demonstration, not a production tool**
- **Test failures are expected and drive improvement**
- **The goal is AI learning, not perfect parsing**
- **Complex PDFs will always have parsing challenges**
- **Success is measured in learning, not accuracy**

please contact me for any improvement at abbasayan4167@gmail.com 
or my linkedin profile - https://www.linkedin.com/in/ayanabbasi/
