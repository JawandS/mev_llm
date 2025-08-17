# MacroEconVue - LLM Economic Simulation

This repository is part of the MacroEconVue study. The goal of this repository is to generate synthetic credit card transactions using LLMs as agents in an economic simulation. The goal of the MacroEconVue project is to reconstruct inflation data using a federated learning machine learning framework.

## Model Steps
1. Agent receives an `income` then pays the `fixed_cost` and `variable_cost` (random number between 0 and `variable_cost`) (all three numbers should be found in `config/agents.csv`)
2. The the net income (calculated, can be negative) is applied to savings (tracked by each agent)
3. The agent (via API call to gemini-2.5-flash) decides how many units of luxury to buy given their current savings and interest rate (cost per unit of luxury and and interest rate set in `config/config.json`)
4. The remaining savings (positive of negative) grow by the interest rate (set in `config/config.json`) and is persisted to the next period

## Files
### Config files (config directory)
- `.env`: contains the API key for gemini (should only contain the API key)
- `agents.csv`: configuration information for the different types of agents
- `config.json`: contains information about the simulation, number / types of agents, periods, interest rate, etc.

### Core files (src directory)
- `simulation.py`: primary file, contains the simulation logic, orchestrates agents and tracks overall data
- `agent.py`: used by the primary simulation logic for API calls, tracking information like income and savings
- `utils.py`: logic such as logging, outputing transaction data, loading dotenv, etc.

### Output files (results.directory)
Each run of the simulation should generate a new subdirectory with the timestamp (dd-mm-yy_hh:mm:ss). The following files/directories should be created
- chat_log/: a subdirectory with information on the various agnets
   - chat_log/agent_\[id\].json: API history of each agent with the id of each agent
   - chat_log/agents.csv: a list of each agent id and the type of agent it was 
- config/: a subdirectory with information on used in that simulation run
   - config/agents.csv: a copy of the outer directory config/agents.csv
   - config/config.json: a copy of the outer directory config/config.json
- transactions.csv: a file containing the aggregate transactions. Each row is: period_num (int), agent_id (int), agent_type (str), purchase_type (str, e.g. fixed_cost, random_cost, luxury), purchase_quantity (int, 1 for things like fixed_cost and random_cost), required (bool, true for fixed/random cost false for luxury)

## Future Tasks (don't complete right now)
- Add an inflation rate to items over time
- Add other types of goods to buy each period (break luxury down into more categories)
- Break fixed and random cost into discrete types of goods/services
- Simulate CPI and pass it to the agents
