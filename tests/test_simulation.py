"""
Tests for the simulation module.

This module contains unit tests for the Simulation class used in the
MEV LLM Economic Simulation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path
import pandas as pd
import sys
sys.path.append('../src')

from src.simulation import Simulation


class TestSimulation(unittest.TestCase):
    """Test cases for the Simulation class."""
    
    def setUp(self):
        """Set up test simulation instance with mocked dependencies."""
        # Mock configuration data
        self.mock_config = {
            "simulation": {"periods": 3, "agents_per_type": 2},
            "economics": {"interest_rate": 0.02, "luxury_cost_per_unit": 50.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        
        # Mock agent types DataFrame
        self.mock_agent_types = pd.DataFrame({
            'agent_type': ['young_professional', 'family'],
            'income': [1200.0, 1800.0],
            'fixed_cost': [400.0, 600.0],
            'variable_cost': [400.0, 600.0]
        })
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_simulation_initialization(self, mock_setup_logging, mock_validate, 
                                     mock_load_config, mock_load_agent_types
                                     ):
        """Test simulation initialization."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation
        simulation = Simulation()
        
        # Verify initialization
        assert simulation.num_periods == 3
        assert simulation.agents_per_type == 2
        assert simulation.interest_rate == 0.02
        assert simulation.luxury_cost_per_unit == 50.0
        assert len(simulation.agents) == 0  # No agents created yet
        assert len(simulation.transactions) == 0
        assert simulation.current_period == 0
        
        # Verify mocks were called
        mock_setup_logging.assert_called_once()
        mock_load_config.assert_called_once()
        mock_load_agent_types.assert_called_once()
        mock_validate.assert_called_once_with(self.mock_config)
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    @patch('src.simulation.Agent')
    def test_create_agents(self, mock_agent_class, mock_setup_logging, 
                          mock_validate, mock_load_config, mock_load_agent_types
                          ):
        """Test agent creation."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Mock Agent constructor
        mock_agents = []
        def create_mock_agent(*args, **kwargs):
            mock_agent = Mock()
            mock_agent.agent_id = kwargs.get('agent_id', len(mock_agents))
            mock_agent.agent_type = kwargs.get('agent_type', 'test')
            mock_agents.append(mock_agent)
            return mock_agent
        
        mock_agent_class.side_effect = create_mock_agent
        
        # Create simulation and agents
        simulation = Simulation()
        simulation.create_agents()
        
        # Verify correct number of agents created
        # 2 agent types × 2 agents per type = 4 total agents
        assert len(simulation.agents) == 4
        assert mock_agent_class.call_count == 4
        
        # Verify agent creation parameters
        calls = mock_agent_class.call_args_list
        assert calls[0][1]['agent_type'] == 'young_professional'
        assert calls[1][1]['agent_type'] == 'young_professional'
        assert calls[2][1]['agent_type'] == 'family'
        assert calls[3][1]['agent_type'] == 'family'
        
        # Verify agent IDs are sequential
        assert calls[0][1]['agent_id'] == 0
        assert calls[1][1]['agent_id'] == 1
        assert calls[2][1]['agent_id'] == 2
        assert calls[3][1]['agent_id'] == 3
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_run_period(self, mock_setup_logging, mock_validate, 
                       mock_load_config, mock_load_agent_types):
        """Test running a single period."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation
        simulation = Simulation()
        
        # Create mock agents
        mock_agent1 = Mock()
        mock_agent1.agent_id = 1
        mock_agent1.process_period.return_value = [
            {'period_num': 0, 'agent_id': 1, 'purchase_type': 'luxury', 'purchase_quantity': 2}
        ]
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = 2
        mock_agent2.process_period.return_value = [
            {'period_num': 0, 'agent_id': 2, 'purchase_type': 'fixed_cost', 'purchase_quantity': 1}
        ]
        
        simulation.agents = [mock_agent1, mock_agent2]
        
        # Run period
        simulation.run_period(0)
        
        # Verify agents processed period
        mock_agent1.process_period.assert_called_once_with(50.0, 0.02)
        mock_agent2.process_period.assert_called_once_with(50.0, 0.02)
        
        # Verify transactions collected
        assert len(simulation.transactions) == 2
        assert simulation.transactions[0]['agent_id'] == 1
        assert simulation.transactions[1]['agent_id'] == 2
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_run_period_with_agent_error(self, mock_setup_logging, mock_validate, 
                                        mock_load_config, mock_load_agent_types
                                        ):
        """Test running period when an agent encounters an error."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation
        simulation = Simulation()
        
        # Create mock agents - one that fails, one that succeeds
        mock_agent1 = Mock()
        mock_agent1.agent_id = 1
        mock_agent1.process_period.side_effect = Exception("Agent processing error")
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = 2
        mock_agent2.process_period.return_value = [
            {'period_num': 0, 'agent_id': 2, 'purchase_type': 'luxury', 'purchase_quantity': 1}
        ]
        
        simulation.agents = [mock_agent1, mock_agent2]
        
        # Run period
        simulation.run_period(0)
        
        # Verify simulation continued despite agent error
        assert len(simulation.transactions) == 1  # Only successful agent's transactions
        assert simulation.transactions[0]['agent_id'] == 2
    
    @patch('src.simulation.create_results_directory')
    @patch('src.simulation.copy_config_files')
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_run_simulation(self, mock_setup_logging, mock_validate, 
                           mock_load_config, mock_load_agent_types,
                           mock_copy_config, mock_create_results):
        """Test complete simulation run."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        mock_results_dir = Path("/tmp/test_results")
        mock_create_results.return_value = mock_results_dir
        
        # Create simulation
        simulation = Simulation()
        
        # Mock the create_agents and run_period methods
        with patch.object(simulation, 'create_agents') as mock_create_agents:
            with patch.object(simulation, 'run_period') as mock_run_period:
                with patch.object(simulation, '_save_results') as mock_save_results:
                    
                    # Run simulation
                    result_dir = simulation.run_simulation()
                    
                    # Verify methods called
                    mock_create_results.assert_called_once()
                    mock_copy_config.assert_called_once_with(mock_results_dir)
                    mock_create_agents.assert_called_once()
                    
                    # Verify run_period called for each period
                    assert mock_run_period.call_count == 3  # 3 periods
                    mock_run_period.assert_any_call(0)
                    mock_run_period.assert_any_call(1)
                    mock_run_period.assert_any_call(2)
                    
                    mock_save_results.assert_called_once()
                    
                    assert result_dir == mock_results_dir
    
    @patch('src.simulation.save_transactions')
    @patch('src.simulation.save_agent_chat_log')
    @patch('src.simulation.save_agents_summary')
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_save_results(self, mock_setup_logging, mock_validate, 
                         mock_load_config, mock_load_agent_types,
                         mock_save_agents_summary, mock_save_agent_chat_log,
                         mock_save_transactions):
        """Test saving simulation results."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation
        simulation = Simulation()
        simulation.results_dir = Path("/tmp/test_results")
        
        # Add mock agents and transactions
        mock_agent1 = Mock()
        mock_agent1.agent_id = 1
        mock_agent1.agent_type = 'young_professional'
        mock_agent1.get_chat_history.return_value = [{'test': 'data1'}]
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = 2
        mock_agent2.agent_type = 'family'
        mock_agent2.get_chat_history.return_value = [{'test': 'data2'}]
        
        simulation.agents = [mock_agent1, mock_agent2]
        simulation.transactions = [{'test': 'transaction'}]
        
        # Save results
        simulation._save_results()
        
        # Verify save functions called
        mock_save_transactions.assert_called_once_with(
            simulation.results_dir, simulation.transactions
        )
        
        assert mock_save_agent_chat_log.call_count == 2
        mock_save_agent_chat_log.assert_any_call(
            simulation.results_dir, 1, [{'test': 'data1'}]
        )
        mock_save_agent_chat_log.assert_any_call(
            simulation.results_dir, 2, [{'test': 'data2'}]
        )
        
        mock_save_agents_summary.assert_called_once_with(
            simulation.results_dir, [(1, 'young_professional'), (2, 'family')]
        )
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_get_simulation_summary(self, mock_setup_logging, mock_validate, 
                                   mock_load_config, mock_load_agent_types):
        """Test getting simulation summary."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation
        simulation = Simulation()
        simulation.results_dir = Path("/tmp/test_results")
        
        # Add mock agents
        mock_agent1 = Mock()
        mock_agent1.agent_id = 1
        mock_agent1.savings = 1000.0
        
        mock_agent2 = Mock()
        mock_agent2.agent_id = 2
        mock_agent2.savings = 1500.0
        
        simulation.agents = [mock_agent1, mock_agent2]
        
        # Add mock transactions
        simulation.transactions = [
            {
                'period_num': 0, 'agent_id': 1, 'agent_type': 'young_professional',
                'purchase_type': 'luxury', 'purchase_quantity': 2, 'required': False
            },
            {
                'period_num': 0, 'agent_id': 1, 'agent_type': 'young_professional',
                'purchase_type': 'fixed_cost', 'purchase_quantity': 1, 'required': True
            },
            {
                'period_num': 0, 'agent_id': 2, 'agent_type': 'family',
                'purchase_type': 'luxury', 'purchase_quantity': 1, 'required': False
            }
        ]
        
        # Get summary
        summary = simulation.get_simulation_summary()
        
        # Verify summary content
        assert summary['simulation_config']['periods'] == 3
        assert summary['simulation_config']['total_agents'] == 2
        assert summary['simulation_config']['interest_rate'] == 0.02
        
        assert summary['transaction_summary']['total_transactions'] == 3
        assert summary['transaction_summary']['luxury_transactions'] == 2
        assert summary['transaction_summary']['total_luxury_units'] == 3
        
        assert summary['agent_summary']['average_final_savings'] == 1250.0
        assert summary['agent_summary']['savings_by_agent'] == {1: 1000.0, 2: 1500.0}
        
        assert summary['luxury_by_agent_type']['young_professional'] == 2
        assert summary['luxury_by_agent_type']['family'] == 1
    
    @patch('src.simulation.load_agent_types')
    @patch('src.simulation.load_config')
    @patch('src.simulation.validate_config')
    @patch('src.simulation.setup_logging')
    def test_get_simulation_summary_no_transactions(self, mock_setup_logging, 
                                                   mock_validate, mock_load_config, 
                                                   mock_load_agent_types):
        """Test getting summary when no transactions exist."""
        # Setup mocks
        mock_logger = Mock()
        mock_setup_logging.return_value = mock_logger
        mock_load_config.return_value = self.mock_config
        mock_load_agent_types.return_value = self.mock_agent_types
        
        # Create simulation with no transactions
        simulation = Simulation()
        simulation.transactions = []
        
        # Get summary
        summary = simulation.get_simulation_summary()
        
        # Verify error response
        assert "error" in summary
        assert summary["error"] == "No transactions recorded"


if __name__ == "__main__":
    unittest.main()
