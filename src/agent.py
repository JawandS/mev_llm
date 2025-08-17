"""
Agent implementation for the MEV LLM Economic Simulation.

This module provides the Agent class that represents individual economic actors
in the simulation. Each agent has financial characteristics and uses LLM-based
decision making for luxury purchases.
"""

import random
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai


class Agent:
    """
    Represents an individual economic agent in the simulation.
    
    Each agent has income, fixed costs, variable costs, and savings.
    Agents use LLM calls to make decisions about luxury purchases.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        agent_type: str, 
        income: float, 
        fixed_cost: float, 
        variable_cost: float,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize an economic agent.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., 'young_professional', 'family')
            income: Monthly income
            fixed_cost: Fixed monthly expenses
            variable_cost: Maximum variable monthly expenses
            api_key: Google API key for Gemini
            model_name: Name of the LLM model to use
            temperature: LLM temperature for response variability
            max_tokens: Maximum tokens for LLM responses
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.income = income
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        
        # Financial state
        self.savings = 0.0
        self.period = 0
        
        # Chat history for this agent
        self.chat_history: List[Dict[str, Any]] = []
        
        # Configure Gemini API
        genai.configure(api_key=api_key)
        
        # Configure safety settings to be more permissive for economic simulation
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name,
            safety_settings=safety_settings
        )
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        
    def calculate_net_income(self) -> float:
        """
        Calculate net income after fixed and variable costs.
        
        Returns:
            Net income (can be negative)
        """
        actual_variable_cost = random.uniform(0, self.variable_cost)
        net_income = self.income - self.fixed_cost - actual_variable_cost
        
        self.logger.debug(
            f"Period {self.period}: Income=${self.income:.2f}, "
            f"Fixed=${self.fixed_cost:.2f}, Variable=${actual_variable_cost:.2f}, "
            f"Net=${net_income:.2f}"
        )
        
        return net_income, actual_variable_cost
    
    def decide_luxury_purchases(
        self, 
        luxury_cost_per_unit: float, 
        interest_rate: float
    ) -> int:
        """
        Use LLM to decide how many luxury units to purchase.
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Current interest rate
            
        Returns:
            Number of luxury units to purchase
        """
        # Create prompt for the LLM
        prompt = self._create_luxury_prompt(luxury_cost_per_unit, interest_rate)
        
        try:
            # Make API call to Gemini
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            
            # Check if response is valid and has content
            if not response.candidates:
                self.logger.error("No candidates in response")
                raise ValueError("No candidates in response")
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY
                self.logger.error("Response was blocked by safety filters")
                raise ValueError("Response blocked by safety filters")
            elif candidate.finish_reason == 3:  # RECITATION
                self.logger.error("Response was blocked due to recitation")
                raise ValueError("Response blocked due to recitation")
            elif candidate.finish_reason not in [0, 1]:  # UNSPECIFIED or STOP
                self.logger.error(f"Response finished with reason: {candidate.finish_reason}")
                raise ValueError(f"Invalid finish reason: {candidate.finish_reason}")
            
            # Check if content exists and has parts
            if not candidate.content or not candidate.content.parts:
                self.logger.error("No content parts in response")
                raise ValueError("No content parts in response")
            
            response_text = response.text
            
            # Log the interaction
            interaction = {
                "period": self.period,
                "prompt": prompt,
                "response": response_text,
                "timestamp": str(self.period)
            }
            self.chat_history.append(interaction)
            
            # Parse the response to get number of units
            luxury_units = self._parse_luxury_response(response_text)
            
            self.logger.info(
                f"Agent {self.agent_id} decided to buy {luxury_units} luxury units"
            )
            
            return luxury_units
            
        except Exception as e:
            self.logger.error(f"Error in LLM call: {e}")
            # Log the failed interaction
            self.chat_history.append({
                "period": self.period,
                "prompt": prompt,
                "response": f"ERROR: {str(e)}",
                "fallback_decision": "SIMULATION_HALTED",
                "timestamp": str(self.period)
            })
            # Halt execution by re-raising the exception
            raise RuntimeError(f"LLM call failed for Agent {self.agent_id}: {e}") from e
    
    def _create_luxury_prompt(self, luxury_cost_per_unit: float, interest_rate: float) -> str:
        """
        Create a prompt for the LLM to decide on luxury purchases.
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Current interest rate
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Economic simulation analysis for {self.agent_type}.

Financial data:
- Savings: {self.savings:.2f}
- Income: {self.income:.2f}
- Fixed expenses: {self.fixed_cost:.2f}
- Variable expenses: {self.variable_cost:.2f}
- Interest rate: {interest_rate:.1%}
- Item cost: {luxury_cost_per_unit:.2f}

Simulation period: {self.period + 1}

Calculate optimal purchase quantity for discretionary items based on:
- Available funds
- Interest opportunity cost
- Consumer profile: {self.agent_type}
- Financial prudence

Output format: Single integer (0 or greater)."""
        
        return prompt
    
    def _parse_luxury_response(self, response_text: str) -> int:
        """
        Parse the LLM response to extract number of luxury units.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Number of luxury units (non-negative integer)
        """
        try:
            # Extract first number from response
            import re
            numbers = re.findall(r'\d+', response_text)
            
            if numbers:
                luxury_units = int(numbers[0])
                # Ensure non-negative
                return max(0, luxury_units)
            else:
                self.logger.warning(f"No number found in response: {response_text}")
                return 0
                
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Could not parse response '{response_text}': {e}")
            return 0
    
    def process_period(
        self, 
        luxury_cost_per_unit: float, 
        interest_rate: float
    ) -> List[Dict[str, Any]]:
        """
        Process a complete period for this agent.
        
        This includes:
        1. Calculating net income
        2. Adding to savings
        3. Deciding on luxury purchases
        4. Applying interest to remaining savings
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Interest rate per period
            
        Returns:
            List of transaction records for this period
        """
        transactions = []
        period_start_savings = self.savings
        
        # Step 1: Calculate net income and costs
        net_income, actual_variable_cost = self.calculate_net_income()
        
        # Step 2: Add net income to savings
        self.savings += net_income
        
        # Record fixed cost transaction
        transactions.append({
            'period_num': self.period,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'purchase_type': 'fixed_cost',
            'purchase_quantity': 1,
            'required': True,
            'amount': self.fixed_cost
        })
        
        # Record variable cost transaction
        transactions.append({
            'period_num': self.period,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'purchase_type': 'variable_cost',
            'purchase_quantity': 1,
            'required': True,
            'amount': actual_variable_cost
        })
        
        # Step 3: Decide on luxury purchases
        luxury_units = self.decide_luxury_purchases(luxury_cost_per_unit, interest_rate)
        
        # Check if agent can afford luxury purchases
        total_luxury_cost = luxury_units * luxury_cost_per_unit
        if total_luxury_cost <= self.savings:
            self.savings -= total_luxury_cost
            
            if luxury_units > 0:
                transactions.append({
                    'period_num': self.period,
                    'agent_id': self.agent_id,
                    'agent_type': self.agent_type,
                    'purchase_type': 'luxury',
                    'purchase_quantity': luxury_units,
                    'required': False,
                    'amount': total_luxury_cost
                })
                
                self.logger.info(
                    f"Agent {self.agent_id} purchased {luxury_units} luxury units "
                    f"for ${total_luxury_cost:.2f}"
                )
        else:
            self.logger.warning(
                f"Agent {self.agent_id} cannot afford {luxury_units} luxury units "
                f"(cost: ${total_luxury_cost:.2f}, savings: ${self.savings:.2f})"
            )
        
        # Step 4: Apply interest (deannualize) to remaining savings
        interest_earned = self.savings * (interest_rate / 12)
        self.savings += interest_earned
        
        self.logger.info(
            f"Period {self.period}: Agent {self.agent_id} "
            f"savings: ${period_start_savings:.2f} -> ${self.savings:.2f} "
            f"(interest: ${interest_earned:.2f})"
        )
        
        # Increment period counter
        self.period += 1
        
        return transactions
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current state of the agent.
        
        Returns:
            Dictionary with agent's current state
        """
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'period': self.period,
            'savings': self.savings,
            'income': self.income,
            'fixed_cost': self.fixed_cost,
            'variable_cost': self.variable_cost
        }
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete chat history for this agent.
        
        Returns:
            List of chat interactions
        """
        return self.chat_history.copy()
