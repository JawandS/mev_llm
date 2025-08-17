"""
Utility functions for the MEV LLM Economic Simulation.

This module provides essential utilities for:
- Environment variable management
- Logging configuration
- Data persistence and output handling
- Configuration loading
"""

import os
import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dotenv import load_dotenv
import pandas as pd


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the simulation.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_environment() -> str:
    """
    Load environment variables from .env file.
    
    Returns:
        Google API key for Gemini
        
    Raises:
        ValueError: If API key is not found
    """
    load_dotenv(Path("config/.env"))
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Please create config/.env with your API key."
        )
    
    return api_key


def load_config() -> Dict[str, Any]:
    """
    Load simulation configuration from config.json.
    
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path("config/config.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def load_agent_types() -> pd.DataFrame:
    """
    Load agent type configurations from agents.csv.
    
    Returns:
        DataFrame with agent type configurations
        
    Raises:
        FileNotFoundError: If agents.csv doesn't exist
        pd.errors.EmptyDataError: If CSV is empty
    """
    agents_path = Path("config/agents.csv")
    
    if not agents_path.exists():
        raise FileNotFoundError(f"Agents configuration file not found: {agents_path}")
    
    try:
        agents_df = pd.read_csv(agents_path)
        required_columns = ['agent_type', 'income', 'fixed_cost', 'variable_cost']
        
        if not all(col in agents_df.columns for col in required_columns):
            raise ValueError(f"agents.csv must contain columns: {required_columns}")
        
        return agents_df
        
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("agents.csv is empty")


def create_results_directory() -> Path:
    """
    Create a timestamped results directory for simulation output.
    
    Returns:
        Path to the created results directory
    """
    timestamp = datetime.now().strftime("%d-%m-%y_%H:%M:%S")
    results_dir = Path(f"results/{timestamp}")
    
    # Create directory structure
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "chat_log").mkdir(exist_ok=True)
    (results_dir / "config").mkdir(exist_ok=True)
    
    return results_dir


def copy_config_files(results_dir: Path) -> None:
    """
    Copy configuration files to the results directory.
    
    Args:
        results_dir: Path to the results directory
    """
    import shutil
    
    config_dest = results_dir / "config"
    
    # Copy agents.csv
    shutil.copy2("config/agents.csv", config_dest / "agents.csv")
    
    # Copy config.json
    shutil.copy2("config/config.json", config_dest / "config.json")


def save_agent_chat_log(results_dir: Path, agent_id: int, chat_history: List[Dict[str, Any]]) -> None:
    """
    Save agent's chat history to JSON file.
    
    Args:
        results_dir: Path to the results directory
        agent_id: Unique identifier for the agent
        chat_history: List of chat messages and responses
    """
    chat_log_path = results_dir / "chat_log" / f"agent_{agent_id}.json"
    
    with open(chat_log_path, 'w') as f:
        json.dump(chat_history, f, indent=2, default=str)


def save_agents_summary(results_dir: Path, agents_info: List[Tuple[int, str]]) -> None:
    """
    Save summary of agents (ID and type) to CSV.
    
    Args:
        results_dir: Path to the results directory
        agents_info: List of (agent_id, agent_type) tuples
    """
    agents_summary_path = results_dir / "chat_log" / "agents.csv"
    
    with open(agents_summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['agent_id', 'agent_type'])
        writer.writerows(agents_info)


def save_transactions(results_dir: Path, transactions: List[Dict[str, Any]]) -> None:
    """
    Save transaction data to CSV file.
    
    Args:
        results_dir: Path to the results directory
        transactions: List of transaction dictionaries
    """
    transactions_path = results_dir / "transactions.csv"
    
    if not transactions:
        # Create empty file with headers
        with open(transactions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'period_num', 'agent_id', 'agent_type', 'purchase_type', 
                'purchase_quantity', 'required'
            ])
        return
    
    # Save transactions to CSV
    df = pd.DataFrame(transactions)
    df.to_csv(transactions_path, index=False)


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = {
        'simulation': ['periods', 'agents_per_type'],
        'economics': ['interest_rate', 'luxury_cost_per_unit'],
        'llm': ['model_name', 'temperature', 'max_tokens']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
        
        for key in keys:
            if key not in config[section]:
                raise ValueError(f"Missing configuration key: {section}.{key}")
    
    # Validate numeric ranges
    if config['simulation']['periods'] <= 0:
        raise ValueError("periods must be positive")
    
    if config['simulation']['agents_per_type'] <= 0:
        raise ValueError("agents_per_type must be positive")
    
    if not (0 <= config['economics']['interest_rate'] <= 1):
        raise ValueError("interest_rate must be between 0 and 1")
    
    if config['economics']['luxury_cost_per_unit'] <= 0:
        raise ValueError("luxury_cost_per_unit must be positive")
