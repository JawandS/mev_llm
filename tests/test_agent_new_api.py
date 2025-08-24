"""
Updated tests for the Agent class with the new API.
This file contains tests that work with the current Agent implementation.
"""

import unittest
from unittest.mock import Mock, patch
import sys
sys.path.append('../src')

from src.agent import Agent
from src.utils import load_config, load_agent_types


class TestAgentNewAPI(unittest.TestCase):
    """Test cases for the Agent class using the new API."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test configuration."""
        # Load actual configuration values
        cls.config = load_config()
        cls.agent_types_df = load_agent_types()
        
        # Get young_professional data from config
        young_prof_data = cls.agent_types_df[
            cls.agent_types_df['agent_type'] == 'young_professional'
        ].iloc[0]
        cls.young_prof_income = float(young_prof_data['income'])
        
        # Build fixed and variable costs from config structure
        cls.fixed_costs = {}
        for cost_type in cls.config['economics']['fixed_costs']:
            cls.fixed_costs[cost_type] = float(young_prof_data[cost_type])
        
        cls.variable_costs = {}
        for cost_type in cls.config['economics']['variable_costs']:
            cls.variable_costs[cost_type] = float(young_prof_data[cost_type])
            
        # Get economic parameters
        cls.interest_rate = cls.config['economics']['interest_rate']
        cls.discretionary_goods = cls.config['economics']['discretionary_goods']
    
    def setUp(self):
        """Set up test agent instance."""
        self.agent = Agent(
            agent_id=1,
            agent_type="young_professional",
            income=self.young_prof_income,
            fixed_costs=self.fixed_costs,
            variable_costs=self.variable_costs,
            discretionary_goods=self.discretionary_goods
        )
    
    def test_discretionary_purchases_api(self):
        """Test the discretionary purchases API."""
        self.agent.savings = 100.0
        
        # Mock the LLM call to avoid network dependency
        with patch('src.agent.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "entertainment: 2\ntravel: 1"
            }
            mock_response.raise_for_status.return_value = None
            mock_requests.post.return_value = mock_response
            
            result = self.agent.decide_discretionary_purchases(interest_rate=0.04)
            
            # Should return a dictionary
            assert isinstance(result, dict)
            
            # Should have keys for available goods
            for good in self.discretionary_goods.keys():
                assert good in result
                assert isinstance(result[good], int)
                assert result[good] >= 0
    
    def test_process_period_complete_cycle(self):
        """Test a complete period processing cycle."""
        self.agent.savings = 50.0
        
        # Mock the LLM call
        with patch('src.agent.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "entertainment: 1\ntravel: 0"
            }
            mock_response.raise_for_status.return_value = None
            mock_requests.post.return_value = mock_response
            
            # Mock random.uniform for variable costs
            with patch('random.uniform') as mock_random:
                mock_random.side_effect = [50.0, 20.0]  # healthcare, repair
                
                transactions = self.agent.process_period(
                    interest_rate=0.04,
                    current_period=1
                )
                
                # Should return a list of transactions
                assert isinstance(transactions, list)
                assert len(transactions) > 0
                
                # Agent's period should be updated
                assert self.agent.period == 1
    
    def test_financial_constraints(self):
        """Test agent behavior with limited funds."""
        self.agent.savings = 1.0  # Very low savings
        
        # Mock the LLM call
        with patch('src.agent.requests') as mock_requests:
            mock_response = Mock()
            mock_response.json.return_value = {
                "response": "entertainment: 0\ntravel: 0"
            }
            mock_response.raise_for_status.return_value = None
            mock_requests.post.return_value = mock_response
            
            result = self.agent.decide_discretionary_purchases(interest_rate=0.04)
            
            # Should still return valid dictionary even with constraints
            assert isinstance(result, dict)
            for good in self.discretionary_goods.keys():
                assert good in result
                # With low savings, purchases should be conservative
                assert result[good] >= 0


if __name__ == '__main__':
    unittest.main()
