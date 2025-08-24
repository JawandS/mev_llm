# Copilot Instructions for MacroEconVue

## Project Overview
MacroEconVue simulates synthetic credit card transactions using LLM-powered agents. The simulation reconstructs inflation data via federated learning. Agents interact with an LLM (Ollama API) to make economic decisions based on configuration and simulation state.

## Architecture & Key Files
- `main.py`: Entry point, sets up and runs the simulation.
- `src/simulation.py`: Orchestrates simulation steps, manages agents, and tracks data.
- `src/agent.py`: Defines agent logic, including LLM API calls and state management.
- `src/utils.py`: Logging, output, and utility functions.
- `config/agents.csv` & `config/config.json`: Agent and simulation configuration.
- `stored_results/`: Stores output from each run, organized by timestamp.
- `run_tests.py` & `tests/`: Test suite for core logic.

## Developer Workflows
- **Install dependencies:** `pip install -r requirements.txt`
- **Run simulation:** `source .venv.bin/activate && python main.py [--verbose]`
- **Run tests:** `source .venv.bin/activate && python run_tests.py`
- **Configure agents/simulation:** Edit files in `config/`
- **Start Ollama LLM server:**
  - Install from https://ollama.com/download
  - Run `ollama serve`
  - Pull model: `ollama pull phi3` (or set model in `config.json`)

## Patterns & Conventions
- **Results:** Each simulation run creates a timestamped subdirectory in `stored_results/` with:
  - `transactions.csv`: All transactions
  - `chat_log/`: Per-agent LLM logs
  - `config/`: Copy of config files used
- **Agent Decisions:** Agents use LLM API calls to decide luxury purchases. All agent parameters (income, costs) are loaded from CSV/config.
- **Extending Simulation:** Add new agent types or goods by updating config files and extending logic in `agent.py` and `simulation.py`.
- **Testing:** All core logic is covered by tests in `tests/`. Use `run_tests.py` to run all tests.

## Integration Points
- **LLM API:** Agents interact with Ollama via local API calls. Model name is set in `config/config.json`.
- **Configurable Parameters:** All simulation and agent parameters are set in `config/` and copied to results for reproducibility.

## Examples
- To add a new agent type, update `config/agents.csv` and extend logic in `agent.py`.
- To change the LLM model, set `model_name` in `config/config.json` and pull the model with Ollama.
- To debug agent decisions, inspect `stored_results/[timestamp]/chat_log/agent_X.json`.
- To run any python script, always prepend `source .venv.bin/activate &&` to the command.

---
**Use this document first, use it as a baseline truth and verify with actual code.**
