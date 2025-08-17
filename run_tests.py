#!/usr/bin/env python3
"""
Test runner for the MEV LLM Economic Simulation.

This script runs all unit tests for the project and provides
a summary of test results.
"""

import unittest
import sys
from pathlib import Path

# Add src directory to path so tests can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import test modules
from tests.test_utils import *
from tests.test_agent import *
from tests.test_simulation import *


def run_tests():
    """
    Run all unit tests and return results.
    
    Returns:
        unittest.TestResult: Test results object
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    suite.addTests(loader.loadTestsFromName('tests.test_utils'))
    suite.addTests(loader.loadTestsFromName('tests.test_agent'))
    suite.addTests(loader.loadTestsFromName('tests.test_simulation'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def main():
    """Main entry point for test runner."""
    print("="*60)
    print("MEV LLM Economic Simulation - Test Suite")
    print("="*60)
    
    try:
        result = run_tests()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback.split(chr(10))[-2]}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback.split(chr(10))[-2]}")
        
        # Exit with appropriate code
        if result.failures or result.errors:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)
            
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
