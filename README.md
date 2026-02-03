# Data Analaysis ReAct-style AI Agent

## Goal

Building an AI agent that is capable of performing the following tasks:
- Generating Python code for data analysis and visualization in a plot
- Executing the generated code in a sandbox: a isolated environment, in order to avoid execution of unreliable code
- Produce Plotly-based HTML visualizations

## Architecture

The suggested architecture is the Hexagonal one (also known as *Ports \& Adapters* architecture). Since the (optional Phase B) advanced features are not implemented, there was not necessary to adopt such architecture. However, with future perpesctive, the project was structured in order to be expandable for Ports and Adapters implementation (specifically suitable for the advanced features). Then the project present the following architecture:

- **Agent Layer**: This layer contains the implementation of the ReAct agent, including the state and the *ReAct* loop. The agent is responsible for:
    - Generating Python code for data analysis and visualization
    - Executing the generated code in a sandbox environment
    - Producing Plotly-based HTML visualizations
- **Tools Layer**: This layer contains the implementation of the tools used by the agent to perform its tasks, such as the code execution tool in a sandbox environment.

To implement this architecture, the project is structured in the following way:
```
agent-coding-assignment/
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py                # Agent state definition
│   │   └── simple_agent.py         # ReAct agent implementation 
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── sandbox.py              # CodeSandbox class for isolated execution
│   │   └── code_executor.py        # LangChain tool wrapper for code execution
│   │
│   ├── adapters/                   
│   │   └── __init__.py
│   │
│   ├──ports/
│   │   └── __init__.py
│   │
│   └── main.py                     # Main entry point to run the agent
│
├── output/                         # Directory for generated HTML plots
│   └── plot_*.html                 # Plotly visualizations (generated at runtime)
│
├── logs/
│   └── agent.log                   # Execution logs
│
├── tests/
│   ├── test_sandbox.py             # Tests for sandbox execution
│
├── docker-compose.yml              # Docker Compose configuration
├── Dockerfile                      # Dockerfile for containerizing the application
├── .env                            # Environment variables (GOOGLE_API_KEY)
├── .gitignore
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── example.ipynb                   # Learning/experimentation notebook
```

### Core Components

#### `src/agent/`
- **`state.py`**: Defines `AgentState` with messages and iteration counter
- **`react_agent.py`**: Implements `DataAnalysisAgent` class with:
  - ReAct loop (Reasoning → Action → Observation)
  - Agent nodes for reasoning and code generation
  - Conditional logic for iteration control

#### `src/tools/`
- **`sandbox.py`**: Implements `CodeSandbox` class for:
  - Subprocess-based code execution
  - Timeout enforcement (default: 30s)
  - Filesystem isolation using temporary directories
  - Plot extraction and storage
- **`code_executor.py`**: Defines `execute_python_code` LangChain tool for agent interaction

#### `src/main.py`
- Entry point that:
  - Loads environment variables
  - Initializes the agent
  - Runs example queries

### Directory Purposes

- **`src/`**: All source code following layered architecture
- **`outputs/`**: Runtime-generated HTML visualizations 
- **`logs/`**: Execution logs for debugging
- **`tests/`**: Unit and integration tests
- **Root files**: Configuration and documentation

## Technological Stack

- **ReAct Agent**: Implemented using LangGraph for structured reasoning and action flow
- **Language Model**: Google GenAI models via LangChain's wrapper
- **Sandboxing**: Custom subprocess-based sandbox for secure code execution using Python's `subprocess` and `tempfile` modules
- **Data Manipulation & Visualization**: Pandas and Plotly for data analysis and interactive plotting
- **

## Key Design Decisions
1. Subprocess over E2B: Simpler, meets requirements, reproducible
2. No Phase B: Focus on robust Phase A given time constraints
3. Gemini 2.5 Flash: Fast, good code generation, free tier available, and the last available model in the *Flash* category that is not in preview anymore (Gemini 3 Flash is in preview)
4. Hexagonal Architecture not fully implemented: Since the E2B sandbox and Chat Memory \& Persistence are not implemented, due the time constraints, it will be result unuseful, however it is 
structured for future expansion.
5. Since the LLM are inconsistent in wrapping the code within the response message, the code extraction method has been improved to handle multiple cases (```python ... ```, ``` ... ```, and no wrapping at all).
6. Given some constraints to the agent for performing the task:
    - Limited number of iterations (default: 5), both to avoid infinite loops and to not exceed the token limit of the LLM
    - Limited timeout for code execution (default: 30s), to prevent long-running or hanging processes
    - Filesystem restriction (temp directory only)  
7. Applied specific markers in the prompts, in order to guide the LLM to act as expected as possible.  



## Limitations: 
- Docker containerization for portable execution
- No network blocking at OS level (would require Docker network policies)
- No chat memory or persistence across sessions
- Limited error handling for complex code execution failures:
    - No error categorization (syntax vs runtime vs timeout)
    - No incremental fixes (agent regenerates entire code)
- E2B sandbox integration or Docker-based sandboxing with network isolation

## How to Run

1. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Set the `GOOGLE_API_KEY` environment variable in a `.env` file:
    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```
3. Run the main script:
    ```bash
    python -m src.main  
    ```