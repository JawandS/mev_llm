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
        fixed_costs: Dict[str, float],
        variable_costs: Dict[str, float],
        discretionary_goods: Dict[str, float],
        model_name: str = "llama3.2",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        """
        Initialize an economic agent with dynamic cost structure.
        
        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of agent (e.g., 'young_professional', 'family')
            income: Weekly income
            fixed_costs: Dictionary of fixed cost categories and values
            variable_costs: Dictionary of variable cost categories and values (max amounts)
            discretionary_goods: Dictionary of discretionary goods and their prices
            model_name: Name of the LLM model to use
            temperature: LLM temperature for response variability
            max_tokens: Maximum tokens for LLM responses
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.income = income
        self.fixed_costs = fixed_costs
        self.variable_costs = variable_costs
        self.discretionary_goods = discretionary_goods
        
        # Calculate totals for compatibility and easy access
        self.fixed_cost = sum(fixed_costs.values())
        self.variable_cost = sum(variable_costs.values())
        
        # Store individual cost components for backward compatibility
        # This allows existing code to still access self.housing, self.insurance, etc.
        for cost_name, cost_value in fixed_costs.items():
            setattr(self, cost_name, cost_value)
        for cost_name, cost_value in variable_costs.items():
            setattr(self, cost_name, cost_value)
        
        # Financial state
        self.savings = 0.0
        self.period = 0
        
        # Initialize actual variable cost tracking dynamically
        self.actual_variable_costs = {cost_name: 0.0 for cost_name in variable_costs.keys()}
        self.actual_variable_cost = 0.0  # Total of all variable costs
        
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
        # Calculate actual variable costs as random values up to the maximum for each category
        total_actual_variable_cost = 0.0
        
        for cost_name, max_cost in self.variable_costs.items():
            actual_cost = random.uniform(0, max_cost)
            self.actual_variable_costs[cost_name] = actual_cost
            total_actual_variable_cost += actual_cost
            # Also set individual attributes for backward compatibility
            setattr(self, f"actual_{cost_name}_cost", actual_cost)
        
        self.actual_variable_cost = total_actual_variable_cost
        net_income = self.income - self.fixed_cost - self.actual_variable_cost
        
        # Create dynamic logging message
        fixed_summary = ", ".join([f"{cost}=${value:.2f}" for cost, value in self.fixed_costs.items()])
        variable_summary = ", ".join([f"{cost}=${self.actual_variable_costs[cost]:.2f}" for cost in self.variable_costs.keys()])
        
        self.logger.debug(
            f"Period {self.period}: Income=${self.income:.2f}, "
            f"Fixed costs: {fixed_summary}, "
            f"Variable costs: {variable_summary}, "
            f"Net=${net_income:.2f}"
        )
        
        return net_income, self.actual_variable_cost
    
    def decide_discretionary_purchases(
        self, 
        interest_rate: float
    ) -> Dict[str, int]:
        """
        Use LLM to make rational discretionary purchase decisions based on economic optimization.
        
        The LLM is always consulted to maintain complete economic information,
        even when financially constrained, to enable learning and realistic modeling.
        
        Args:
            interest_rate: Annual interest rate (opportunity cost)
            
        Returns:
            Dictionary mapping goods to quantities to purchase
        """
        # Calculate maximum affordable for each good
        max_affordable = {}
        for good, price in self.discretionary_goods.items():
            max_affordable[good] = max(0, int(self.savings // price)) if price > 0 else 0
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Choose prompt type based on attempt number
                if attempt == 0:
                    # First attempt: Full economic analysis prompt
                    prompt = self._create_discretionary_prompt(interest_rate)
                    prompt_type = "full_economic_analysis"
                else:
                    # Retry attempts: Simplified numeric-only prompt
                    prompt = self._create_simplified_discretionary_prompt(max_affordable, attempt)
                    prompt_type = f"simplified_retry_{attempt}"
                
                # Call Ollama API for rational decision-making
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "options": {
                            "temperature": 0.1 if attempt > 0 else self.temperature,  # Lower temperature for retries
                            "num_predict": 100 if attempt > 0 else self.max_tokens  # Shorter responses for retries
                        },
                        "stream": False
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                response_text = data.get("response", "")
                
                # Try to parse the response
                purchases, success = self._parse_discretionary_response(response_text, max_affordable)
                
                if success:
                    # Successful parse - log and return result
                    can_afford_any = any(self.savings >= price for price in self.discretionary_goods.values())
                    decision_type = "llm_rational_choice" if can_afford_any else "llm_constrained_choice"
                    if attempt > 0:
                        decision_type += f"_retry_{attempt}"
                    
                    interaction = {
                        "period": self.period,
                        "prompt": prompt,
                        "response": response_text,
                        "parsed_purchases": purchases,
                        "timestamp": str(self.period),
                        "decision_type": decision_type,
                        "prompt_type": prompt_type,
                        "attempt_number": attempt + 1,
                        "financially_constrained": not can_afford_any
                    }
                    self.chat_history.append(interaction)
                    
                    return purchases
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
    
    def _create_discretionary_prompt(self, interest_rate: float) -> str:
        """
        Create a prompt for the LLM to decide on entertainment and travel purchases.
        
        Args:
            interest_rate: Current interest rate
            
        Returns:
            Formatted prompt string
        """
        # Calculate maximum affordable for each good
        max_affordable = {}
        goods_info = []
        for good, price in self.discretionary_goods.items():
            max_affordable[good] = max(0, int(self.savings // price)) if price > 0 else 0
            goods_info.append(f"{good}: ${price:.2f} per unit (max affordable: {max_affordable[good]})")
        
        monthly_interest_rate = interest_rate / 12
        
        goods_list = "\n".join([f"- {info}" for info in goods_info])
        
        prompt = f"""You are a {self.agent_type} with ${self.savings:.2f} savings making discretionary spending decisions.

AVAILABLE DISCRETIONARY GOODS:
{goods_list}

ECONOMIC CONTEXT:
- Monthly interest rate: {monthly_interest_rate*100:.3f}% (Annual: {interest_rate*100:.2f}%)
- Current savings grow to ${self.savings * (1 + monthly_interest_rate):.2f} next month if saved
- This period income after fixed/variable costs: ${self.income - self.fixed_cost - self.actual_variable_cost:.2f}

DECISION FRAMEWORK:
Consider utility maximization vs. opportunity cost of foregone interest earnings.
Each good provides different satisfaction:
- Entertainment: Social experiences, relaxation, cultural activities  
- Travel: Exploration, adventure, new experiences

Your agent type ({self.agent_type}) should influence preferences and risk tolerance.

CRITICAL: Respond with ONLY a JSON object showing quantities for each good.
Format: {{"entertainment": 0, "travel": 0}}
Replace 0 with your chosen quantities (must be within affordable limits).

Examples:
{{"entertainment": 2, "travel": 1}}
{{"entertainment": 0, "travel": 3}}
{{"entertainment": 1, "travel": 0}}

Your decision:"""
        
        return prompt
    
    def process_period(
        self, 
        interest_rate: float,
        current_period: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Process a complete period for this agent following rational economic sequence:
        
        1. Receive income and pay mandatory expenses (fixed + variable costs)
        2. Calculate available savings for discretionary spending
        3. Make rational discretionary purchase decisions via LLM
        4. Execute purchases if affordable
        5. Apply interest to remaining savings/debt (monthly, every 4 periods)
        
        Args:
            interest_rate: Annual interest rate
            current_period: Current simulation period (0-indexed)
            
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
        
        # STEP 2: RATIONAL DISCRETIONARY CONSUMPTION DECISION
        # Agent makes optimal decision based on current savings, opportunity cost, and risk preferences
        purchases = self.decide_discretionary_purchases(interest_rate)
        
        # STEP 3: EXECUTE PURCHASES (if affordable and rational)
        total_discretionary_cost = 0
        purchase_costs = {}
        
        # Calculate total cost and individual costs
        for good, quantity in purchases.items():
            if quantity > 0:
                cost = quantity * self.discretionary_goods[good]
                purchase_costs[good] = cost
                total_discretionary_cost += cost
        
        if total_discretionary_cost > 0:
            if total_discretionary_cost <= self.savings:
                self.savings -= total_discretionary_cost
                
                # Create transaction record for each good purchased
                for good, quantity in purchases.items():
                    if quantity > 0:
                        transactions.append({
                            'period_num': self.period,
                            'agent_id': self.agent_id,
                            'agent_type': self.agent_type,
                            'purchase_type': good,
                            'purchase_quantity': quantity,
                            'required': False,
                            'amount': purchase_costs[good]
                        })
                
                purchase_summary = ", ".join([
                    f"{quantity} {good} units (${purchase_costs[good]:.2f})"
                    for good, quantity in purchases.items() if quantity > 0
                ])
                
                self.logger.info(
                    f"Agent {self.agent_id} rationally purchased {purchase_summary}. "
                    f"Total cost: ${total_discretionary_cost:.2f}. Remaining savings: ${self.savings:.2f}"
                )
            else:
                self.logger.warning(
                    f"Agent {self.agent_id} attempted irrational purchase: total cost ${total_discretionary_cost:.2f} "
                    f"> savings ${self.savings:.2f}. Purchase denied."
                )
                # Override irrational decision
                purchases = {good: 0 for good in purchases}
                total_discretionary_cost = 0
        
        # STEP 4: APPLY INTEREST TO REMAINING SAVINGS/DEBT (MONTHLY)
        # Interest is applied every 4 periods (monthly) to both savings and debt
        interest_earned = 0.0
        if (current_period + 1) % 4 == 0:  # Apply interest monthly (every 4 weeks)
            monthly_interest_rate = interest_rate / 12  # Convert annual to monthly rate
            interest_earned = self.savings * monthly_interest_rate
            self.savings += interest_earned
            
            interest_type = "earned" if interest_earned >= 0 else "charged"
            self.logger.debug(
                f"Monthly interest {interest_type}: ${abs(interest_earned):.2f} "
                f"(Rate: {monthly_interest_rate*100:.3f}% monthly)"
            )
        
        self.logger.info(
            f"Period {self.period + 1} complete for Agent {self.agent_id}: "
            f"Savings ${period_start_savings:.2f} -> ${self.savings:.2f} "
            f"(Net income: ${net_income:.2f}, Discretionary spending: ${total_discretionary_cost:.2f}, "
            f"Interest {('earned' if interest_earned >= 0 else 'charged')}: ${abs(interest_earned):.2f})"
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
    
    def _parse_discretionary_response(self, response_text: str, max_affordable: Dict[str, int]) -> Tuple[Dict[str, int], bool]:
        """
        Parse the LLM response for discretionary purchases.
        
        Args:
            response_text: Raw response from LLM
            max_affordable: Dictionary of maximum affordable quantities per good
            
        Returns:
            Tuple of (purchases_dict, success)
        """
        try:
            import json
            import re
            
            # Clean up the response text
            response_text = response_text.strip()
            
            # Try to extract JSON from the response
            # Look for JSON-like pattern
            json_match = re.search(r'\{[^}]*\}', response_text)
            if json_match:
                json_str = json_match.group()
                try:
                    purchases = json.loads(json_str)
                    
                    # Validate the purchases
                    if isinstance(purchases, dict):
                        valid_purchases = {}
                        for good, quantity in purchases.items():
                            if good in max_affordable:
                                try:
                                    qty = int(quantity)
                                    if 0 <= qty <= max_affordable[good]:
                                        valid_purchases[good] = qty
                                    else:
                                        valid_purchases[good] = min(max(0, qty), max_affordable[good])
                                except (ValueError, TypeError):
                                    valid_purchases[good] = 0
                        
                        # Ensure all goods are represented
                        for good in max_affordable:
                            if good not in valid_purchases:
                                valid_purchases[good] = 0
                        
                        return valid_purchases, True
                        
                except json.JSONDecodeError:
                    pass
            
            # Fallback: try to parse as comma-separated values (legacy format)
            if ',' in response_text:
                parts = response_text.split(',')
                if len(parts) >= 2:
                    try:
                        entertainment_num = int(parts[0].strip())
                        travel_num = int(parts[1].strip())
                        
                        purchases = {
                            'entertainment': min(max(0, entertainment_num), max_affordable.get('entertainment', 0)),
                            'travel': min(max(0, travel_num), max_affordable.get('travel', 0))
                        }
                        return purchases, True
                    except ValueError:
                        pass
            
            # Default fallback
            return {good: 0 for good in max_affordable}, False
            
        except Exception:
            return {good: 0 for good in max_affordable}, False

    def _create_simplified_discretionary_prompt(self, max_affordable: Dict[str, int], attempt: int) -> str:
        """
        Create a simplified prompt for retry attempts when the main prompt fails.
        
        Args:
            max_affordable: Dictionary of maximum affordable quantities per good
            attempt: Current attempt number
            
        Returns:
            Simplified prompt string
        """
        goods_info = []
        for good, price in self.discretionary_goods.items():
            max_qty = max_affordable.get(good, 0)
            goods_info.append(f"{good.title()} costs ${price:.2f} per unit (max: {max_qty})")
        
        goods_text = "\n".join(goods_info)
        
        return f"""You have ${self.savings:.2f} savings.
{goods_text}

Respond with ONLY a JSON object with your choices:
{{"entertainment": 0, "travel": 0}}

Replace 0 with your chosen quantities (within limits).
Your choice:"""
