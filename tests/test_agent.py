"""
Tests for the agent module.

This module contains unit tests for the Agent class used in the
MEV LLM Economic Simulation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import pandas as pd
sys.path.append('../src')

from src.agent import Agent
from src.utils import load_config, load_agent_types


class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
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
            
        # Calculate totals for backward compatibility
        cls.young_prof_fixed = sum(cls.fixed_costs.values())
        cls.young_prof_variable = sum(cls.variable_costs.values())
        
        # Get economic parameters - config has nested structure
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
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        assert self.agent.agent_id == 1
        assert self.agent.agent_type == "young_professional"
        assert self.agent.income == self.young_prof_income
        assert self.agent.fixed_cost == self.young_prof_fixed
        assert self.agent.variable_cost == self.young_prof_variable
        assert self.agent.savings == 0.00
        assert self.agent.period == 0
        assert len(self.agent.chat_history) == 0
    
    def test_calculate_net_income(self):
        """Test net income calculation."""
        # Since there are 2 variable cost categories, mock both calls
        healthcare_cost = self.variable_costs['healthcare'] / 2
        repair_cost = self.variable_costs['repair'] / 2
        with patch('random.uniform') as mock_random:
            mock_random.side_effect = [healthcare_cost, repair_cost]
            
            net_income, total_variable_cost = self.agent.calculate_net_income()
            
            expected_variable_cost = healthcare_cost + repair_cost
            expected_net = self.young_prof_income - self.young_prof_fixed - expected_variable_cost
            assert net_income == expected_net
            assert total_variable_cost == expected_variable_cost
    
    def test_calculate_net_income_negative(self):
        """Test net income calculation resulting in negative value."""
        # Use maximum costs for both variable cost categories
        healthcare_cost = self.variable_costs['healthcare']
        repair_cost = self.variable_costs['repair']
        with patch('random.uniform') as mock_random:
            mock_random.side_effect = [healthcare_cost, repair_cost]
            
            net_income, total_variable_cost = self.agent.calculate_net_income()
            
            expected_variable_cost = healthcare_cost + repair_cost
            expected_net = self.young_prof_income - self.young_prof_fixed - expected_variable_cost
            assert net_income == expected_net
            assert total_variable_cost == expected_variable_cost
    
    # DISABLED - API changed from luxury_purchases to discretionary_purchases
    # def test_parse_luxury_response_with_validation_valid(self):
    #     """Test parsing valid luxury response with new validation method."""
    #     # Valid clean number
    #     result, success = self.agent._parse_luxury_response_with_validation("3", 5)
    #     assert result == 3
    #     assert success == True
    #     
    #     # Valid zero
    #     result, success = self.agent._parse_luxury_response_with_validation("0", 5)
    #     assert result == 0
    #     assert success == True
    #     
    #     # Valid number at max affordable
    #     result, success = self.agent._parse_luxury_response_with_validation("5", 5)
    #     assert result == 5
    #     assert success == True

    @patch('src.agent.requests')
    def test_new_discretionary_purchases_api(self, mock_requests):
        """Test the new discretionary purchases API."""
        self.agent.savings = 100.0
        
        # Mock successful LLM response in JSON format
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"entertainment": 2, "travel": 1}'
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        # Test that method exists and returns expected format
        result = self.agent.decide_discretionary_purchases(interest_rate=0.04)
        
        # Should return a dictionary of purchases
        assert isinstance(result, dict)
        
        # Should have keys for available discretionary goods
        for good in self.discretionary_goods.keys():
            assert good in result
            assert isinstance(result[good], int)
            assert result[good] >= 0
        
        # Should have the expected values from our mock
        assert result['entertainment'] == 2
        assert result['travel'] == 1

    # DISABLED - API changed, method signatures and return types different
    # def test_parse_luxury_response_with_validation_invalid(self):
    #     """Test parsing invalid responses that should trigger reprompt."""
    #     # Text with number should fail (strict parsing)
    #     result, success = self.agent._parse_luxury_response_with_validation("I want 3 items", 5)
    #     assert result == 0
    #     assert success == False
    #     ...

    @patch('src.agent.requests')
    def test_new_process_period_api(self, mock_requests):
        """Test the new process_period API."""
        self.agent.savings = 50.0
        
        # Mock successful LLM response in JSON format
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": '{"entertainment": 1, "travel": 0}'
        }
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        # Mock random.uniform for variable costs
        with patch('random.uniform') as mock_random:
            mock_random.side_effect = [50.0, 20.0]  # healthcare, repair costs
            
            # Test that method exists and returns expected format
            transactions = self.agent.process_period(
                interest_rate=0.04,
                current_period=1
            )
            
            # Should return a list of transaction dictionaries
            assert isinstance(transactions, list)
            
            # Should have some transactions (income, expenses, purchases)
            assert len(transactions) > 0
            
            # Each transaction should be a dictionary with expected keys
            for transaction in transactions:
                assert isinstance(transaction, dict)
                assert 'agent_id' in transaction
                assert 'amount' in transaction
                assert 'period_num' in transaction

    def test_create_simplified_prompt(self):
        """Test simplified prompt creation for retries."""
        self.agent.savings = 200.0
        
        # First retry prompt
        prompt1 = self.agent._create_simplified_prompt(12.0, 4, 1)
        assert "RETRY" in prompt1
        assert "0 to 4" in prompt1
        assert "Examples of correct responses" in prompt1
        
        # Second retry prompt (ultra-minimal)
        prompt2 = self.agent._create_simplified_prompt(12.0, 4, 2)
        assert len(prompt2) < len(prompt1)  # Should be shorter
        assert "0 to 4" in prompt2

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_retry_logic_success(self, mock_requests):
        """Test retry logic when first response is unparseable."""
        self.agent.savings = 200.0
        
        # Mock responses: first bad, second good
        mock_response1 = Mock()
        mock_response1.json.return_value = {"response": "I want to buy some items"}  # Unparseable
        mock_response1.raise_for_status.return_value = None
        
        mock_response2 = Mock()
        mock_response2.json.return_value = {"response": "2"}  # Clean number
        mock_response2.raise_for_status.return_value = None
        
        mock_requests.post.side_effect = [mock_response1, mock_response2]
        
        result = self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert result == 2
        assert len(self.agent.chat_history) == 1  # Only successful attempt logged
        assert mock_requests.post.call_count == 2  # Two API calls made
        # Check that it was a retry attempt (attempt 1 = second call)
        assert "_retry_1" in self.agent.chat_history[0]['decision_type']
        assert self.agent.chat_history[0]['attempt_number'] == 2

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_all_retries_fail(self, mock_requests):
        """Test when all retry attempts fail."""
        self.agent.savings = 200.0
        
        # Mock all responses as unparseable
        mock_response = Mock()
        mock_response.json.return_value = {"response": "I cannot decide"}
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        with self.assertRaises(RuntimeError) as context:
            self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert "Failed to get valid LLM response after 3 attempts" in str(context.exception)
        assert mock_requests.post.call_count == 3  # Three attempts made
        assert len(self.agent.chat_history) == 1  # Error logged

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_api_error_with_retries(self, mock_requests):
        """Test API errors with retry logic."""
        self.agent.savings = 200.0
        
        # Mock API errors for all attempts
        mock_requests.post.side_effect = Exception("Connection error")
        
        with self.assertRaises(RuntimeError) as context:
            self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert "Failed to get valid LLM response after 3 attempts" in str(context.exception)
        assert mock_requests.post.call_count == 3

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_number_out_of_range(self, mock_requests):
        """Test when LLM returns number outside valid range."""
        self.agent.savings = 100.0  # Can afford 2 items at 50 each
        
        # Mock response with number too high
        mock_response = Mock()
        mock_response.json.return_value = {"response": "10"}  # Can only afford 2
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        with self.assertRaises(RuntimeError):
            self.agent.decide_luxury_purchases(50.0, 0.02)
        
        # Should trigger retries because 10 > max_affordable (2)

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_temperature_adjustment(self, mock_requests):
        """Test that retry attempts use lower temperature."""
        self.agent.savings = 200.0
        self.agent.temperature = 0.7  # Initial temperature
        
        # Mock first response as unparseable, second as good
        mock_response1 = Mock()
        mock_response1.json.return_value = {"response": "I want some"}
        mock_response1.raise_for_status.return_value = None
        
        mock_response2 = Mock()
        mock_response2.json.return_value = {"response": "2"}
        mock_response2.raise_for_status.return_value = None
        
        mock_requests.post.side_effect = [mock_response1, mock_response2]
        
        self.agent.decide_luxury_purchases(50.0, 0.02)
        
        # Check that retry used lower temperature
        calls = mock_requests.post.call_args_list
        assert calls[0][1]['json']['options']['temperature'] == 0.7  # First call uses agent's temperature
        assert calls[1][1]['json']['options']['temperature'] == 0.1  # Retry uses lower temperature
    
    def test_create_luxury_prompt(self):
        """Test luxury prompt creation."""
        self.agent.savings = 500.0
        self.agent.period = 2
        
        prompt = self.agent._create_luxury_prompt(50.0, 0.02)
        
        assert prompt
    
    @patch('src.agent.requests')
    def test_decide_luxury_purchases_success(self, mock_requests):
        """Test successful luxury purchase decision."""
        # Set savings to positive amount so LLM call is made
        self.agent.savings = 200.0
        
        # Mock the Ollama API response with clean number
        mock_response = Mock()
        mock_response.json.return_value = {"response": "2"}  # Clean number response
        mock_response.raise_for_status.return_value = None
        mock_requests.post.return_value = mock_response
        
        result = self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert result == 2
        assert len(self.agent.chat_history) == 1
        assert self.agent.chat_history[0]['response'] == "2"

    @patch('src.agent.requests')
    def test_decide_luxury_purchases_api_error(self, mock_requests):
        """Test luxury purchase decision with API error."""
        # Set savings to positive amount so LLM call is attempted
        self.agent.savings = 100.0
        
        # Mock the Ollama API to raise an exception
        mock_requests.post.side_effect = Exception("API Error")
        
        # Should raise RuntimeError after retries
        with self.assertRaises(RuntimeError) as context:
            self.agent.decide_luxury_purchases(50.0, 0.02)
        
        assert "Failed to get valid LLM response" in str(context.exception)
    
    def test_process_period_complete(self):
        """Test complete period processing."""
        self.agent.savings = 1000.00
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = self.young_prof_variable * 0.8  # 80% of max variable cost
            
            with patch.object(self.agent, 'decide_luxury_purchases') as mock_decide:
                mock_decide.return_value = 2  # Buy 2 luxury units
                
                transactions = self.agent.process_period(self.luxury_cost, self.interest_rate, 0)
                
                # Check transactions
                assert len(transactions) == 3  # fixed, variable, luxury
                
                # Check fixed cost transaction
                fixed_tx = next(t for t in transactions if t['purchase_type'] == 'fixed_cost')
                assert fixed_tx['agent_id'] == 1
                assert fixed_tx['required'] == True
                assert fixed_tx['amount'] == self.young_prof_fixed
                
                # Check variable cost transaction
                variable_tx = next(t for t in transactions if t['purchase_type'] == 'variable_cost')
                assert variable_tx['required'] == True
                assert variable_tx['amount'] == self.young_prof_variable * 0.8
                
                # Check luxury transaction
                luxury_tx = next(t for t in transactions if t['purchase_type'] == 'luxury')
                assert luxury_tx['required'] == False
                assert luxury_tx['purchase_quantity'] == 2
                assert luxury_tx['amount'] == 2 * self.luxury_cost
                
                # Check savings calculation
                # Initial: 1000
                # + Net income: income - fixed - variable
                # - Luxury: 2 * luxury_cost
                # + Interest: 0 (period 0, no interest)
                net_income = self.young_prof_income - self.young_prof_fixed - (self.young_prof_variable * 0.8)
                expected_savings = 1000 + net_income - (2 * self.luxury_cost)
                assert abs(self.agent.savings - expected_savings) < 0.01
                
                # Check period increment
                assert self.agent.period == 1
    
    def test_process_period_insufficient_funds(self):
        """Test period processing when agent can't afford luxury."""
        self.agent.savings = 50.0  # Low savings
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = self.young_prof_variable  # Max variable cost
            
            with patch.object(self.agent, 'decide_luxury_purchases') as mock_decide:
                # Calculate how much they'll have after income and expenses
                net_income = self.young_prof_income - self.young_prof_fixed - self.young_prof_variable
                total_available = 50.0 + net_income
                max_affordable_units = int(total_available // self.luxury_cost)
                
                # Make them want to buy more than they can afford
                mock_decide.return_value = max_affordable_units + 10
                
                transactions = self.agent.process_period(self.luxury_cost, self.interest_rate, 0)
                
                # Should only have fixed and variable cost transactions, no luxury
                transaction_types = [t['purchase_type'] for t in transactions]
                assert 'fixed_cost' in transaction_types
                assert 'variable_cost' in transaction_types
                assert 'luxury' not in transaction_types  # Couldn't afford it
    
    def test_get_state(self):
        """Test getting agent state."""
        self.agent.savings = 750.00
        self.agent.period = 3
        
        state = self.agent.get_state()
        
        expected_state = {
            'agent_id': 1,
            'agent_type': 'young_professional',
            'period': 3,
            'savings': 750.00,
            'income': self.young_prof_income,
            'fixed_cost': self.young_prof_fixed,
            'variable_cost': self.young_prof_variable
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


class TestAgentEconomicDecisions(unittest.TestCase):
    """Test economic decision-making scenarios."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test configuration."""
        # Load actual configuration values
        cls.config = load_config()
        cls.agent_types_df = load_agent_types()
        
        # Get economic parameters
        cls.discretionary_goods = cls.config['economics']['discretionary_goods']
        
        # Build cost dictionaries for young professional
        young_prof_data = cls.agent_types_df[
            cls.agent_types_df['agent_type'] == 'young_professional'
        ].iloc[0]
        
        cls.young_prof_fixed_costs = {}
        for cost_type in cls.config['economics']['fixed_costs']:
            cls.young_prof_fixed_costs[cost_type] = young_prof_data[cost_type]
        
        cls.young_prof_variable_costs = {}
        for cost_type in cls.config['economics']['variable_costs']:
            cls.young_prof_variable_costs[cost_type] = young_prof_data[cost_type]
    
    def setUp(self):
        """Set up test agents for economic scenarios."""
        self.young_professional = Agent(
            agent_id=10,
            agent_type="young_professional",
            income=4800.0,
            fixed_costs={'housing': 800.0, 'insurance': 200.0},
            variable_costs={'healthcare': 150.0, 'repair': 50.0},
            discretionary_goods=self.discretionary_goods
        )
        
        self.low_income = Agent(
            agent_id=11,
            agent_type="low_income",
            income=1800.0,
            fixed_costs={'housing': 600.0, 'insurance': 250.0},
            variable_costs={'healthcare': 120.0, 'repair': 30.0},
            discretionary_goods=self.discretionary_goods
        )
    
    def test_financial_constraint_scenarios(self):
        """Test various financial constraint scenarios."""
        # Test agent with negative savings
        self.young_professional.savings = -100.0
        max_affordable = max(0, int(self.young_professional.savings // 50.0))
        assert max_affordable == 0
        
        # Test agent with just enough for one item
        self.young_professional.savings = 50.0
        max_affordable = max(0, int(self.young_professional.savings // 50.0))
        assert max_affordable == 1
        
        # Test agent with good savings
        self.young_professional.savings = 500.0
        max_affordable = max(0, int(self.young_professional.savings // 50.0))
        assert max_affordable == 10
    
    def test_prompt_financial_status_messages(self):
        """Test that prompts contain appropriate financial status messages."""
        # Test constrained scenario
        self.low_income.savings = 25.0  # Less than luxury cost
        self.low_income.actual_variable_cost = 250.0
        prompt = self.low_income._create_luxury_prompt(50.0, 0.04)
        assert "BUDGET CONSTRAINT" in prompt or "FINANCIAL STRESS" in prompt
        
        # Test comfortable scenario
        self.young_professional.savings = 1000.0
        self.young_professional.actual_variable_cost = 300.0
        prompt = self.young_professional._create_luxury_prompt(50.0, 0.04)
        assert "FINANCIAL FLEXIBILITY" in prompt
    
    def test_economic_metrics_in_prompt(self):
        """Test that economic metrics are correctly calculated in prompts."""
        self.young_professional.savings = 500.00
        self.young_professional.actual_variable_cost = 75.00
        
        luxury_cost = self.luxury_cost
        prompt = self.young_professional._create_luxury_prompt(luxury_cost, 0.00)
        
        # Check for key economic information
        assert "500.00" in prompt  # Savings amount
        assert f"{luxury_cost:.2f}" in prompt  # Luxury cost
        assert "0.000%" in prompt  # Weekly interest rate (0%/52)
        max_affordable = int(500.00 // luxury_cost)
        assert f"{max_affordable} units" in prompt  # Max affordable


class TestAgentEdgeCases(unittest.TestCase):
    """Test edge cases for the Agent class."""
    
    def test_agent_with_zero_income(self):
        """Test agent with zero income."""
        agent = Agent(
            agent_id=2,
            agent_type="unemployed",
            income=0.0,
            fixed_costs={'housing': 80.0, 'insurance': 20.0},
            variable_costs={'healthcare': 30.0, 'repair': 20.0},
            discretionary_goods={"entertainment": 8.0, "travel": 15.0}
        )
        
        with patch('random.uniform') as mock_random:
            # Mock both variable cost calls
            mock_random.side_effect = [15.0, 10.0]  # healthcare=15, repair=10
            
            net_income, total_variable_cost = agent.calculate_net_income()
            assert net_income == -125.0  # 0 - (80+20) - (15+10)
            assert total_variable_cost == 25.0
    
    def test_agent_with_high_savings_and_low_income(self):
        """Test agent with high savings but low income."""
        agent = Agent(
            agent_id=3,
            agent_type="retiree",
            income=200.0,
            fixed_costs={'housing': 200.0, 'insurance': 100.0},
            variable_costs={'healthcare': 80.0, 'repair': 20.0},
            discretionary_goods={"entertainment": 8.0, "travel": 15.0}
        )
        agent.savings = 10000.0  # High savings
        
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 50.0
            
            with patch.object(agent, 'decide_luxury_purchases') as mock_decide:
                mock_decide.return_value = 10  # Try to buy many luxury items
                
                transactions = agent.process_period(100.0, 0.05, 0)
                
                # Should be able to afford luxury despite negative income
                luxury_transactions = [t for t in transactions if t['purchase_type'] == 'luxury']
                assert len(luxury_transactions) == 1
                assert luxury_transactions[0]['purchase_quantity'] == 10


if __name__ == "__main__":
    unittest.main()
