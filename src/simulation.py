"""
Main simulation orchestrator for the MEV LLM Economic Simulation.

This module contains the Simulation class that orchestrates the entire
economic simulation, managing multiple agents across multiple periods
and collecting transaction data.
"""

import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd

from .agent import Agent
from .utils import (
    setup_logging, load_environment, load_config, load_agent_types,
    create_results_directory, copy_config_files, save_agent_chat_log,
    save_agents_summary, save_transactions, validate_config
)


class Simulation:
    """
    Main simulation orchestrator for the economic simulation.
    
    Manages multiple agents across multiple periods, coordinates their
    interactions, and handles data collection and output.
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """
        Initialize the simulation.
        
        Args:
            config_path: Path to the configuration file
        """
        self.logger = setup_logging()
        self.logger.info("Initializing MEV LLM Economic Simulation")
        
        # Load configuration and environment
        self.config = load_config()
        validate_config(self.config)
        self.api_key = load_environment()
        self.agent_types_df = load_agent_types()
        
        # Initialize simulation state
        self.agents: List[Agent] = []
        self.current_period = 0
        self.transactions: List[Dict[str, Any]] = []
        self.results_dir: Path = None
        
        # Extract config values for easy access
        self.num_periods = self.config['simulation']['periods']
        self.agents_per_type = self.config['simulation']['agents_per_type']
        self.interest_rate = self.config['economics']['interest_rate']
        self.luxury_cost_per_unit = self.config['economics']['luxury_cost_per_unit']
        
        self.logger.info(f"Simulation configured for {self.num_periods} periods")
        self.logger.info(f"Interest rate: {self.interest_rate:.1%}")
        self.logger.info(f"Luxury cost per unit: ${self.luxury_cost_per_unit:.2f}")
    
    def create_agents(self) -> None:
        """
        Create agents based on the configuration.
        
        Creates the specified number of agents for each agent type
        defined in agents.csv.
        """
        agent_id = 0
        
        for _, agent_type_row in self.agent_types_df.iterrows():
            agent_type = agent_type_row['agent_type']
            income = agent_type_row['income']
            fixed_cost = agent_type_row['fixed_cost']
            variable_cost = agent_type_row['variable_cost']
            
            # Create multiple agents of this type
            for i in range(self.agents_per_type):
                agent = Agent(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    income=income,
                    fixed_cost=fixed_cost,
                    variable_cost=variable_cost,
                    api_key=self.api_key,
                    model_name=self.config['llm']['model_name'],
                    temperature=self.config['llm']['temperature'],
                    max_tokens=self.config['llm']['max_tokens']
                )
                
                self.agents.append(agent)
                agent_id += 1
                
                self.logger.debug(
                    f"Created agent {agent_id-1}: {agent_type} "
                    f"(income=${income}, fixed=${fixed_cost}, variable=${variable_cost})"
                )
        
        self.logger.info(f"Created {len(self.agents)} agents across {len(self.agent_types_df)} types")
    
    def run_period(self, period_num: int) -> None:
        """
        Run a single period of the simulation.
        
        Args:
            period_num: The current period number (0-indexed)
        """
        self.logger.info(f"Starting period {period_num + 1}/{self.num_periods}")
        
        period_transactions = []
        
        # Process each agent for this period
        for agent in self.agents:
            try:
                agent_transactions = agent.process_period(
                    self.luxury_cost_per_unit,
                    self.interest_rate
                )
                period_transactions.extend(agent_transactions)
                
            except Exception as e:
                self.logger.error(
                    f"Error processing agent {agent.agent_id} in period {period_num}: {e}"
                )
                # Continue with other agents even if one fails
                continue
        
        # Add period transactions to total
        self.transactions.extend(period_transactions)
        
        # Log period summary
        period_luxury_purchases = sum(
            1 for t in period_transactions 
            if t['purchase_type'] == 'luxury' and t['purchase_quantity'] > 0
        )
        
        total_luxury_units = sum(
            t['purchase_quantity'] for t in period_transactions 
            if t['purchase_type'] == 'luxury'
        )
        
        self.logger.info(
            f"Period {period_num + 1} complete: "
            f"{period_luxury_purchases} agents made luxury purchases, "
            f"{total_luxury_units} total luxury units purchased"
        )
    
    def run_simulation(self) -> Path:
        """
        Run the complete simulation.
        
        Returns:
            Path to the results directory
        """
        self.logger.info("Starting complete simulation run")
        
        # Create results directory
        self.results_dir = create_results_directory()
        self.logger.info(f"Results will be saved to: {self.results_dir}")
        
        # Copy configuration files
        copy_config_files(self.results_dir)
        
        # Create agents
        self.create_agents()
        
        # Run all periods
        for period in range(self.num_periods):
            self.current_period = period
            self.run_period(period)
        
        # Save results
        self._save_results()
        
        self.logger.info("Simulation completed successfully")
        return self.results_dir
    
    def _save_results(self) -> None:
        """
        Save all simulation results to files.
        """
        self.logger.info("Saving simulation results")
        
        # Save transaction data
        save_transactions(self.results_dir, self.transactions)
        
        # Save agent chat histories
        for agent in self.agents:
            save_agent_chat_log(
                self.results_dir,
                agent.agent_id,
                agent.get_chat_history()
            )
        
        # Save agents summary
        agents_info = [(agent.agent_id, agent.agent_type) for agent in self.agents]
        save_agents_summary(self.results_dir, agents_info)
        
        self.logger.info(f"Results saved to {self.results_dir}")
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation results.
        
        Returns:
            Dictionary with simulation summary statistics
        """
        if not self.transactions:
            return {"error": "No transactions recorded"}
        
        df = pd.DataFrame(self.transactions)
        
        # Calculate summary statistics
        total_transactions = len(df)
        luxury_transactions = df[df['purchase_type'] == 'luxury']
        total_luxury_purchases = len(luxury_transactions)
        total_luxury_units = luxury_transactions['purchase_quantity'].sum()
        
        # Agent savings summary
        agent_savings = {
            agent.agent_id: agent.savings 
            for agent in self.agents
        }
        
        avg_savings = sum(agent_savings.values()) / len(agent_savings)
        
        # Luxury purchases by agent type
        luxury_by_type = (
            luxury_transactions.groupby('agent_type')['purchase_quantity']
            .sum()
            .to_dict()
        )
        
        summary = {
            'simulation_config': {
                'periods': self.num_periods,
                'total_agents': len(self.agents),
                'agents_per_type': self.agents_per_type,
                'interest_rate': self.interest_rate,
                'luxury_cost_per_unit': self.luxury_cost_per_unit
            },
            'transaction_summary': {
                'total_transactions': total_transactions,
                'luxury_transactions': total_luxury_purchases,
                'total_luxury_units': int(total_luxury_units),
                'luxury_purchase_rate': total_luxury_purchases / len(self.agents) / self.num_periods
            },
            'agent_summary': {
                'average_final_savings': avg_savings,
                'savings_by_agent': agent_savings
            },
            'luxury_by_agent_type': luxury_by_type,
            'results_directory': str(self.results_dir)
        }
        
        return summary


def main():
    """
    Main entry point for running the simulation.
    """
    try:
        # Create and run simulation
        simulation = Simulation()
        results_dir = simulation.run_simulation()
        
        # Print summary
        summary = simulation.get_simulation_summary()
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved to: {results_dir}")
        print(f"Total agents: {summary['simulation_config']['total_agents']}")
        print(f"Periods run: {summary['simulation_config']['periods']}")
        print(f"Total transactions: {summary['transaction_summary']['total_transactions']}")
        print(f"Luxury purchases: {summary['transaction_summary']['luxury_transactions']}")
        print(f"Total luxury units: {summary['transaction_summary']['total_luxury_units']}")
        print(f"Average final savings: ${summary['agent_summary']['average_final_savings']:.2f}")
        print("\nLuxury purchases by agent type:")
        for agent_type, units in summary['luxury_by_agent_type'].items():
            print(f"  {agent_type}: {units} units")
        print("="*60)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Simulation failed: {e}")
        raise


if __name__ == "__main__":
    main()
