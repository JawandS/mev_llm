"""
Tests for the agent module.

This module contains unit tests for the Agent class used in the
MEV LLM Economic Simulation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
sys.path.append('../src')

from src.agent import Agent


class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
    def setUp(self):
        """Set up test agent instance."""
        self.agent = Agent(
            agent_id=1,
            agent_type="young_professional",
            income=1200.0,
            fixed_cost=400.0,
            variable_cost=400.0,
            api_key="test_api_key"
        )
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == 1
        assert self.agent.agent_type == "young_professional"
        assert self.agent.income == 1200.0
        assert self.agent.fixed_cost == 400.0
        assert self.agent.variable_cost == 400.0
        assert self.agent.savings == 0.0
        assert self.agent.period == 0
        assert len(self.agent.chat_history) == 0
    
    def test_calculate_net_income(self):
        """Test net income calculation."""
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 200.0  # Variable cost
            
            net_income, variable_cost = self.agent.calculate_net_income()
            
            expected_net = 1200.0 - 400.0 - 200.0  # income - fixed - variable
            assert net_income == expected_net
            assert variable_cost == 200.0
    
    def test_calculate_net_income_negative(self):
        """Test net income calculation resulting in negative value."""
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 400.0  # Maximum variable cost
            
            net_income, variable_cost = self.agent.calculate_net_income()
            
            expected_net = 1200.0 - 400.0 - 400.0  # = 400.0
            assert net_income == expected_net
            assert variable_cost == 400.0
    
    def test_parse_luxury_response_valid(self):
        """Test parsing valid luxury response."""
        response = "I would like to buy 3 luxury units."
        result = self.agent._parse_luxury_response(response)
        assert result == 3
    
    def test_parse_luxury_response_zero(self):
        """Test parsing response with zero."""
        response = "I don't want to buy any luxury items. 0"
        result = self.agent._parse_luxury_response(response)
        assert result == 0
    
    def test_parse_luxury_response_no_number(self):
        """Test parsing response with no numbers."""
        response = "I'm not sure what to buy."
        result = self.agent._parse_luxury_response(response)
        assert result == 0
    
    def test_parse_luxury_response_negative(self):
        """Test parsing response with negative number (should return 0)."""
        response = "I want to return -2 items."
        result = self.agent._parse_luxury_response(response)
        assert result == 0  # Negative numbers should be converted to 0
    
    def test_create_luxury_prompt(self):
        """Test luxury prompt creation."""
        self.agent.savings = 500.0
        self.agent.period = 2
        
        prompt = self.agent._create_luxury_prompt(50.0, 0.02)
        
        assert "young_professional" in prompt
        assert "$500.00" in prompt
        assert "period 3" in prompt
        assert "$50.00" in prompt
        assert "2.0%" in prompt
    
    @patch('src.agent.genai')
    def test_decide_luxury_purchases_success(self, mock_genai):
        """Test successful luxury purchase decision."""
        # Mock the generative model
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "I want to buy 2 luxury units."
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        
        self.agent.model = mock_model
        
        result = self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert result == 2
        assert len(self.agent.chat_history) == 1
        assert self.agent.chat_history[0]['response'] == "I want to buy 2 luxury units."
    
    @patch('src.agent.genai')
    def test_decide_luxury_purchases_api_error(self, mock_genai):
        """Test luxury purchase decision with API error."""
        # Mock the generative model to raise an exception
        mock_model = Mock()
        mock_model.generate_content.side_effect = Exception("API Error")
        mock_genai.GenerativeModel.return_value = mock_model
        
        self.agent.model = mock_model
        
        result = self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert result == 0  # Should return 0 on error
        assert len(self.agent.chat_history) == 1
        assert "ERROR" in self.agent.chat_history[0]['response']
    
    def test_process_period_complete(self):
        """Test complete period processing."""
        self.agent.savings = 1000.0
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 300.0  # Variable cost
            
            with patch.object(self.agent, 'decide_luxury_purchases') as mock_decide:
                mock_decide.return_value = 2  # Buy 2 luxury units
                
                transactions = self.agent.process_period(50.0, 0.02)
                
                # Check transactions
                assert len(transactions) == 3  # fixed, variable, luxury
                
                # Check fixed cost transaction
                fixed_tx = next(t for t in transactions if t['purchase_type'] == 'fixed_cost')
                assert fixed_tx['agent_id'] == 1
                assert fixed_tx['required'] == True
                assert fixed_tx['amount'] == 400.0
                
                # Check variable cost transaction
                variable_tx = next(t for t in transactions if t['purchase_type'] == 'variable_cost')
                assert variable_tx['required'] == True
                assert variable_tx['amount'] == 300.0
                
                # Check luxury transaction
                luxury_tx = next(t for t in transactions if t['purchase_type'] == 'luxury')
                assert luxury_tx['required'] == False
                assert luxury_tx['purchase_quantity'] == 2
                assert luxury_tx['amount'] == 100.0  # 2 * 50.0
                
                # Check savings calculation
                # Initial: 1000
                # + Net income: 1200 - 400 - 300 = 500
                # - Luxury: 100
                # + Interest: (1000 + 500 - 100) * 0.02 = 28
                expected_savings = 1000 + 500 - 100 + (1400 * 0.02)
                assert abs(self.agent.savings - expected_savings) < 0.01
                
                # Check period increment
                assert self.agent.period == 1
    
    def test_process_period_insufficient_funds(self):
        """Test period processing when agent can't afford luxury."""
        self.agent.savings = 50.0  # Low savings
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 300.0
            
            with patch.object(self.agent, 'decide_luxury_purchases') as mock_decide:
                mock_decide.return_value = 5  # Try to buy 5 units (250 cost) but only have ~550 savings
                
                transactions = self.agent.process_period(50.0, 0.02)
                
                # Should only have fixed and variable cost transactions, no luxury
                transaction_types = [t['purchase_type'] for t in transactions]
                assert 'fixed_cost' in transaction_types
                assert 'variable_cost' in transaction_types
                assert 'luxury' not in transaction_types  # Couldn't afford it
    
    def test_get_state(self):
        """Test getting agent state."""
        self.agent.savings = 750.0
        self.agent.period = 3
        
        state = self.agent.get_state()
        
        expected_state = {
            'agent_id': 1,
            'agent_type': 'young_professional',
            'period': 3,
            'savings': 750.0,
            'income': 1200.0,
            'fixed_cost': 400.0,
            'variable_cost': 400.0
        }
        
        assert state == expected_state
    
    def test_get_chat_history(self):
        """Test getting chat history."""
        # Add some mock chat history
        self.agent.chat_history = [
            {"period": 0, "prompt": "test", "response": "test response"}
        ]
        
        history = self.agent.get_chat_history()
        
        assert len(history) == 1
        assert history[0]['period'] == 0
        # Ensure it's a copy, not the original
        assert history is not self.agent.chat_history


class TestAgentEdgeCases(unittest.TestCase):
    """Test edge cases for the Agent class."""
    
    def test_agent_with_zero_income(self):
        """Test agent with zero income."""
        agent = Agent(
            agent_id=2,
            agent_type="unemployed",
            income=0.0,
            fixed_cost=100.0,
            variable_cost=50.0,
            api_key="test_key"
        )
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 25.0
            
            net_income, _ = agent.calculate_net_income()
            assert net_income == -125.0  # 0 - 100 - 25
    
    def test_agent_with_high_savings_and_low_income(self):
        """Test agent with high savings but low income."""
        agent = Agent(
            agent_id=3,
            agent_type="retiree",
            income=200.0,
            fixed_cost=300.0,
            variable_cost=100.0,
            api_key="test_key"
        )
        agent.savings = 10000.0  # High savings
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 50.0
            
            with patch.object(agent, 'decide_luxury_purchases') as mock_decide:
                mock_decide.return_value = 10  # Try to buy many luxury items
                
                transactions = agent.process_period(100.0, 0.05)
                
                # Should be able to afford luxury despite negative income
                luxury_transactions = [t for t in transactions if t['purchase_type'] == 'luxury']
                assert len(luxury_transactions) == 1
                assert luxury_transactions[0]['purchase_quantity'] == 10


if __name__ == "__main__":
    unittest.main()
