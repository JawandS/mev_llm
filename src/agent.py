"""
Agent implementation for the MEV LLM Economic Simulation.

This module provides the Agent class that represents rational economic actors
in the simulation. Each agent follows microeconomic principles for household
decision-making:

- Receives periodic income and pays mandatory expenses
- Makes utility-maximizing decisions about discretionary consumption
- Considers opportunity costs, risk management, and future security
- Uses LLM-based reasoning to balance immediate gratification vs. saving

The simulation follows a coherent economic sequence each period:
1. Income receipt and mandatory expense payment
2. Rational luxury consumption decision (LLM-based)
3. Interest application to remaining savings
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple
import requests


class Agent:
    """
    Represents a rational economic agent in the simulation.
    
    Each agent follows standard microeconomic principles:
    1. Receives periodic income and pays mandatory expenses (fixed + variable costs)
    2. Makes utility-maximizing decisions about discretionary luxury consumption
    3. Considers opportunity costs (interest foregone) and risk management
    4. Maintains savings that grow at the prevailing interest rate
    
    The agent uses LLM-based decision making to balance immediate consumption
    utility against future financial security, following rational household
    economic theory.
    """
    
    def __init__(
        self, 
        agent_id: int, 
        agent_type: str, 
        income: float, 
        fixed_cost: float, 
        variable_cost: float,
        model_name: str = "llama3.2",
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
        self.actual_variable_cost = 0.0  # Initialize to 0
        
        # Chat history for this agent
        self.chat_history: List[Dict[str, Any]] = []
        
        # Ollama model configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
            
        self.logger = logging.getLogger(f"Agent_{agent_id}")
        
    def calculate_net_income(self) -> float:
        """
        Calculate net income after fixed and variable costs.
        
        Returns:
            Net income (can be negative)
        """
        self.actual_variable_cost = random.uniform(0, self.variable_cost)
        net_income = self.income - self.fixed_cost - self.actual_variable_cost
        
        self.logger.debug(
            f"Period {self.period}: Income=${self.income:.2f}, "
            f"Fixed=${self.fixed_cost:.2f}, Variable=${self.actual_variable_cost:.2f}, "
            f"Net=${net_income:.2f}"
        )
        
        return net_income, self.actual_variable_cost
    
    def decide_luxury_purchases(
        self, 
        luxury_cost_per_unit: float, 
        interest_rate: float
    ) -> int:
        """
        Use LLM to make rational luxury purchase decision based on economic optimization.
        
        The LLM is always consulted to maintain complete economic information,
        even when financially constrained, to enable learning and realistic modeling.
        
        If the initial response cannot be parsed, retry with a simplified prompt
        to ensure data integrity without fallback values.
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Annual interest rate (opportunity cost)
            
        Returns:
            Number of luxury units to purchase (optimal choice)
        """
        max_affordable = max(0, int(self.savings // luxury_cost_per_unit)) if luxury_cost_per_unit > 0 else 0
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Choose prompt type based on attempt number
                if attempt == 0:
                    # First attempt: Full economic analysis prompt
                    prompt = self._create_luxury_prompt(luxury_cost_per_unit, interest_rate)
                    prompt_type = "full_economic_analysis"
                else:
                    # Retry attempts: Simplified numeric-only prompt
                    prompt = self._create_simplified_prompt(luxury_cost_per_unit, max_affordable, attempt)
                    prompt_type = f"simplified_retry_{attempt}"
                
                # Call Ollama API for rational decision-making
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "options": {
                            "temperature": 0.1 if attempt > 0 else self.temperature,  # Lower temperature for retries
                            "num_predict": 50 if attempt > 0 else self.max_tokens  # Shorter responses for retries
                        },
                        "stream": False
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                response_text = data.get("response", "")
                
                # Try to parse the response
                luxury_quantity, parse_success = self._parse_luxury_response_with_validation(response_text, max_affordable)
                
                if parse_success:
                    # Successful parse - log and return result
                    can_afford_any = self.savings >= luxury_cost_per_unit
                    decision_type = "llm_rational_choice" if can_afford_any else "llm_constrained_choice"
                    if attempt > 0:
                        decision_type += f"_retry_{attempt}"
                    
                    interaction = {
                        "period": self.period,
                        "prompt": prompt,
                        "response": response_text,
                        "parsed_quantity": luxury_quantity,
                        "timestamp": str(self.period),
                        "decision_type": decision_type,
                        "prompt_type": prompt_type,
                        "attempt_number": attempt + 1,
                        "financially_constrained": not can_afford_any
                    }
                    self.chat_history.append(interaction)
                    
                    # Apply budget constraint if necessary
                    if luxury_quantity > max_affordable:
                        original_quantity = luxury_quantity
                        luxury_quantity = max_affordable
                        self.logger.info(
                            f"Agent {self.agent_id} chose {original_quantity} units but constrained to "
                            f"affordable maximum {max_affordable} (savings: ${self.savings:.2f})"
                        )
                        interaction["constraint_applied"] = f"Reduced from {original_quantity} to {max_affordable} (budget limit)"
                    
                    return luxury_quantity
                else:
                    # Parse failed - log and retry
                    self.logger.warning(f"Attempt {attempt + 1}: Could not parse LLM response: '{response_text}'")
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying with simplified prompt...")
                    
            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1}: Error in LLM call: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying after error...")
                
        # If all retries failed, this is a critical error - raise exception to halt simulation
        error_msg = f"Agent {self.agent_id}: Failed to get valid LLM response after {max_retries} attempts"
        self.logger.error(error_msg)
        
        # Log the complete failure
        self.chat_history.append({
            "period": self.period,
            "error": error_msg,
            "timestamp": str(self.period),
            "decision_type": "critical_failure",
            "attempts_made": max_retries
        })
        
        # Raise exception to halt simulation and preserve data integrity
        raise RuntimeError(error_msg)
    
    def _create_luxury_prompt(self, luxury_cost_per_unit: float, interest_rate: float) -> str:
        """
        Create a prompt for the LLM to decide on luxury purchases.
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Current interest rate
            
        Returns:
            Formatted prompt string
        """
        # Calculate key economic metrics for decision-making
        max_affordable = max(0, int(self.savings // luxury_cost_per_unit)) if luxury_cost_per_unit > 0 else 0
        monthly_interest_rate = interest_rate / 12
        
        # Calculate what savings would be worth next period if saved
        potential_savings_growth = self.savings * (1 + monthly_interest_rate)
        
        # Calculate expected net income for risk assessment
        expected_variable_cost = self.variable_cost / 2  # Average of 0 to variable_cost
        expected_net_income = self.income - self.fixed_cost - expected_variable_cost
        
        # Determine financial constraint status
        financial_status = ""
        if max_affordable == 0:
            if self.savings <= 0:
                financial_status = "⚠️  FINANCIAL STRESS: You have negative/zero savings and cannot afford any luxury purchases."
            else:
                financial_status = f"⚠️  BUDGET CONSTRAINT: Your savings (${self.savings:.2f}) are insufficient for even one luxury item (${luxury_cost_per_unit:.2f})."
        elif max_affordable == 1:
            financial_status = f"💰 TIGHT BUDGET: You can afford only {max_affordable} luxury item, requiring careful consideration."
        elif max_affordable <= 3:
            financial_status = f"💰 LIMITED OPTIONS: You can afford up to {max_affordable} luxury items - moderate financial flexibility."
        else:
            financial_status = f"💰 FINANCIAL FLEXIBILITY: You can afford up to {max_affordable} luxury items - good financial position."
        
        # Add savings buffer analysis
        if self.savings > 0:
            months_of_coverage = self.savings / (self.fixed_cost + expected_variable_cost)
            if months_of_coverage < 1:
                financial_status += f"\n⚠️  RISK WARNING: Current savings only cover {months_of_coverage:.1f} months of expenses."
            elif months_of_coverage < 3:
                financial_status += f"\n⚠️  LOW BUFFER: Current savings cover {months_of_coverage:.1f} months of expenses (recommended: 3+ months)."
        
        prompt = f"""You are a rational economic agent managing household finances for a '{self.agent_type}' profile. Your goal is to maximize utility by balancing immediate luxury consumption against future financial security.

ECONOMIC SITUATION (Period {self.period + 1}):
Financial Position:
- Available savings for spending: ${self.savings:.2f}
- This period's net income: ${self.income - self.fixed_cost - self.actual_variable_cost:.2f}
  (Income ${self.income:.2f} - Fixed costs ${self.fixed_cost:.2f} - Variable costs ${self.actual_variable_cost:.2f})

{financial_status}

Investment Opportunity:
- Luxury goods cost: ${luxury_cost_per_unit:.2f} per unit
- Maximum you can afford: {max_affordable} units
- Monthly interest rate: {monthly_interest_rate*100:.3f}% (Annual: {interest_rate*100:.2f}%)
- If you save instead: ${self.savings:.2f} grows to ${potential_savings_growth:.2f} next period

RATIONAL DECISION FRAMEWORK:
You must choose how many luxury units to purchase (0 to {max_affordable}) by considering:

1. UTILITY MAXIMIZATION: Each luxury unit provides immediate satisfaction
2. RISK MANAGEMENT: Maintaining savings provides security against:
   - Future income volatility (variable costs range ${0:.0f}-${self.variable_cost:.2f})
   - Unexpected expenses or economic downturns
   - Building recommended emergency fund (3+ months expenses)

3. OPPORTUNITY COST: Money spent on luxury cannot earn {monthly_interest_rate*100:.3f}% monthly interest

4. AGENT PROFILE CONSIDERATIONS:
   - {self.agent_type} households typically prioritize {'financial stability and family security' if self.agent_type == 'family' else 'building long-term wealth while enjoying some current consumption' if self.agent_type == 'young_professional' else 'preserving retirement savings and conservative spending' if self.agent_type == 'retiree' else 'minimizing expenses while building emergency funds' if self.agent_type == 'student' else 'essential expenses first, very cautious luxury spending' if self.agent_type == 'low_income' else 'moderate spending with financial security'}

FINANCIAL CONSTRAINT AWARENESS:
- Even if you cannot afford luxury now, this experience teaches you about financial planning
- Consider how your current financial constraints might influence future earning/saving strategies
- Recognize the value of delayed gratification when resources are limited

DECISION RULE:
As a rational agent, purchase luxury goods only when the immediate utility exceeds the combined value of:
- Interest earnings foregone
- Risk reduction from maintaining savings buffer
- Future consumption opportunities

Choose the optimal number of luxury units (0-{max_affordable}) that maximizes your expected utility while maintaining appropriate financial security for your profile.

CRITICAL: Your response must contain ONLY a single number from 0 to {max_affordable}. Do not include any words, explanations, punctuation, or formatting. Examples of correct responses: 0, 1, 2, 3. Examples of incorrect responses: "I choose 2", "0 units", "2.", "The answer is 2".

Your decision:"""
        
        return prompt
    
    def _create_simplified_prompt(self, luxury_cost_per_unit: float, max_affordable: int, attempt: int) -> str:
        """
        Create a simplified prompt for retry attempts when the main prompt fails.
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            max_affordable: Maximum number of units the agent can afford
            attempt: Retry attempt number (1, 2, etc.)
            
        Returns:
            Simplified prompt focused only on getting a number
        """
        if attempt == 1:
            # First retry: Clear and direct
            return f"""RETRY: You must respond with ONLY a number.

Available savings: ${self.savings:.2f}
Item cost: ${luxury_cost_per_unit:.2f}
Maximum affordable: {max_affordable}

How many items to buy? Enter only a number from 0 to {max_affordable}.

Examples of correct responses: 0, 1, 2, 3
Examples of incorrect responses: "I choose 2", "0 items", "two"

Answer:"""

        else:
            # Second retry: Ultra-minimal
            return f"""Enter a number from 0 to {max_affordable}:"""
    
    def _parse_luxury_response_with_validation(self, response_text: str, max_affordable: int) -> Tuple[int, bool]:
        """
        Parse LLM response with strict validation - only accept clean numbers.
        
        Args:
            response_text: Raw response from LLM
            max_affordable: Maximum affordable quantity for validation
            
        Returns:
            Tuple of (quantity, success_flag)
        """
        if not response_text:
            return 0, False
            
        # Clean the response - remove whitespace only
        cleaned_response = response_text.strip()
        
        # Strict validation: must be a pure number with no other text
        if cleaned_response.isdigit():
            quantity = int(cleaned_response)
            # Ensure it's within valid range
            if 0 <= quantity <= max_affordable:
                return quantity, True
            else:
                # Number is outside valid range - reprompt
                return 0, False
        
        # Any response that isn't a pure number should trigger a reprompt
        return 0, False
    
    def process_period(
        self, 
        luxury_cost_per_unit: float, 
        interest_rate: float
    ) -> List[Dict[str, Any]]:
        """
        Process a complete period for this agent following rational economic sequence:
        
        1. Receive income and pay mandatory expenses (fixed + variable costs)
        2. Calculate available savings for discretionary spending
        3. Make rational luxury purchase decision via LLM
        4. Execute purchase if affordable
        5. Apply interest to remaining savings
        
        Args:
            luxury_cost_per_unit: Cost per unit of luxury goods
            interest_rate: Annual interest rate
            
        Returns:
            List of transaction records for this period
        """
        transactions = []
        period_start_savings = self.savings
        
        self.logger.info(
            f"Agent {self.agent_id} ({self.agent_type}) starting period {self.period + 1} "
            f"with ${period_start_savings:.2f} savings"
        )
        
        # STEP 1: INCOME & MANDATORY EXPENSES
        # Calculate and pay mandatory costs (fixed + random variable)
        net_income, actual_variable_cost = self.calculate_net_income()
        
        # Update savings with net income (after mandatory expenses)
        self.savings += net_income
        
        self.logger.info(
            f"Period {self.period + 1}: Net income ${net_income:.2f} "
            f"(Income ${self.income:.2f} - Fixed ${self.fixed_cost:.2f} - Variable ${actual_variable_cost:.2f}). "
            f"Available for discretionary spending: ${self.savings:.2f}"
        )
        
        # Record mandatory expense transactions
        transactions.append({
            'period_num': self.period,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'purchase_type': 'fixed_cost',
            'purchase_quantity': 1,
            'required': True,
            'amount': self.fixed_cost
        })
        
        transactions.append({
            'period_num': self.period,
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'purchase_type': 'variable_cost',
            'purchase_quantity': 1,
            'required': True,
            'amount': self.actual_variable_cost
        })
        
        # STEP 2: RATIONAL LUXURY CONSUMPTION DECISION
        # Agent makes optimal decision based on current savings, opportunity cost, and risk preferences
        luxury_units = self.decide_luxury_purchases(luxury_cost_per_unit, interest_rate)
        
        # STEP 3: EXECUTE PURCHASE (if affordable and rational)
        total_luxury_cost = luxury_units * luxury_cost_per_unit
        if luxury_units > 0:
            if total_luxury_cost <= self.savings:
                self.savings -= total_luxury_cost
                
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
                    f"Agent {self.agent_id} rationally purchased {luxury_units} luxury units "
                    f"for ${total_luxury_cost:.2f}. Remaining savings: ${self.savings:.2f}"
                )
            else:
                self.logger.warning(
                    f"Agent {self.agent_id} attempted irrational purchase: {luxury_units} units "
                    f"(cost: ${total_luxury_cost:.2f} > savings: ${self.savings:.2f}). Purchase denied."
                )
                # Override irrational decision
                luxury_units = 0
        
        # STEP 4: APPLY INTEREST TO REMAINING SAVINGS
        # Savings grow at the risk-free rate (opportunity cost of consumption)
        monthly_interest_rate = interest_rate / 12
        interest_earned = self.savings * monthly_interest_rate
        self.savings += interest_earned
        
        self.logger.info(
            f"Period {self.period + 1} complete for Agent {self.agent_id}: "
            f"Savings ${period_start_savings:.2f} -> ${self.savings:.2f} "
            f"(Net income: ${net_income:.2f}, Luxury spending: ${total_luxury_cost:.2f}, "
            f"Interest earned: ${interest_earned:.2f})"
        )
        
        # Advance to next period
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
