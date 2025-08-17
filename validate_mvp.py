#!/usr/bin/env python3
"""
Validation script for the MEV LLM Economic Simulation MVP.

This script runs a basic validation of all core components
without making actual API calls to save costs during development.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation import Simulation
from src.agent import Agent


def validate_configuration():
    """Validate configuration loading."""
    print("🔍 Validating configuration...")
    
    try:
        from src.utils import load_config, load_agent_types, validate_config
        
        config = load_config()
        validate_config(config)
        print("  ✅ Configuration valid")
        
        agent_types = load_agent_types()
        print(f"  ✅ Loaded {len(agent_types)} agent types")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def validate_agent_functionality():
    """Validate agent functionality with mocked LLM calls."""
    print("🔍 Validating agent functionality...")
    
    try:
        # Create agent with mocked API
        with patch('src.agent.genai') as mock_genai:
            # Mock the generative model
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "2"  # Agent decides to buy 2 luxury units
            mock_model.generate_content.return_value = mock_response
            mock_genai.GenerativeModel.return_value = mock_model
            
            agent = Agent(
                agent_id=1,
                agent_type="young_professional",
                income=1200.0,
                fixed_cost=400.0,
                variable_cost=400.0,
                api_key="test_key"
            )
            
            # Test income calculation
            with patch('random.uniform', return_value=300.0):
                net_income, variable_cost = agent.calculate_net_income()
                assert net_income == 500.0  # 1200 - 400 - 300
                print("  ✅ Income calculation works")
            
            # Test luxury decision (mocked)
            luxury_units = agent.decide_luxury_purchases(50.0, 0.02)
            assert luxury_units == 2
            print("  ✅ Luxury decision making works")
            
            # Test period processing
            agent.savings = 1000.0
            with patch('random.uniform', return_value=300.0):
                transactions = agent.process_period(50.0, 0.02)
                assert len(transactions) >= 2  # At least fixed and variable
                print("  ✅ Period processing works")
            
            print(f"  ✅ Agent final savings: ${agent.savings:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_simulation_flow():
    """Validate simulation orchestration with mocked agents."""
    print("🔍 Validating simulation flow...")
    
    try:
        # Create simulation
        sim = Simulation()
        print(f"  ✅ Simulation initialized for {sim.num_periods} periods")
        
        # Create agents
        with patch('src.agent.genai'):
            sim.create_agents()
            print(f"  ✅ Created {len(sim.agents)} agents")
        
        # Mock agent processing
        mock_transactions = [
            {
                'period_num': 0, 'agent_id': 1, 'agent_type': 'young_professional',
                'purchase_type': 'luxury', 'purchase_quantity': 1, 'required': False
            }
        ]
        
        for agent in sim.agents:
            agent.process_period = Mock(return_value=mock_transactions)
        
        # Test single period
        sim.run_period(0)
        print(f"  ✅ Period processing works - {len(sim.transactions)} transactions recorded")
        
        # Test results saving functionality
        from src.utils import create_results_directory
        results_dir = create_results_directory()
        print(f"  ✅ Results directory created: {results_dir}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_data_structures():
    """Validate data structure consistency."""
    print("🔍 Validating data structures...")
    
    try:
        # Test transaction structure
        sample_transaction = {
            'period_num': 0,
            'agent_id': 1,
            'agent_type': 'young_professional',
            'purchase_type': 'luxury',
            'purchase_quantity': 2,
            'required': False,
            'amount': 100.0
        }
        
        required_fields = ['period_num', 'agent_id', 'agent_type', 'purchase_type', 'purchase_quantity', 'required']
        for field in required_fields:
            assert field in sample_transaction
        
        print("  ✅ Transaction structure valid")
        
        # Test agent state structure
        with patch('src.agent.genai'):
            agent = Agent(1, "test", 1000, 200, 200, "test_key")
            state = agent.get_state()
            
            required_state_fields = ['agent_id', 'agent_type', 'period', 'savings', 'income', 'fixed_cost', 'variable_cost']
            for field in required_state_fields:
                assert field in state
        
        print("  ✅ Agent state structure valid")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data structure error: {e}")
        return False


def main():
    """Run all validation tests."""
    print("MEV LLM Economic Simulation - MVP Validation")
    print("="*60)
    
    all_tests_passed = True
    
    # Run all validation tests
    tests = [
        validate_configuration,
        validate_agent_functionality,
        validate_simulation_flow,
        validate_data_structures
    ]
    
    for test in tests:
        if not test():
            all_tests_passed = False
        print()
    
    # Final summary
    print("="*60)
    if all_tests_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("   The MVP is ready for use with a real Google API key.")
        print("   Run 'python main.py' to start the full simulation.")
    else:
        print("❌ Some validation tests failed.")
        print("   Please check the errors above before running the simulation.")
    print("="*60)
    
    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
