#!/usr/bin/env python3
"""
Main entry point for the MEV LLM Economic Simulation.

This script provides an easy way to run the economic simulation
with proper error handling and user feedback.
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.simulation import main as run_simulation


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MEV LLM Economic Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run simulation with default config
  python main.py --verbose          # Run with verbose logging
  
Before running:
  1. Copy config/.env.example to config/.env
  2. Add your Google API key to config/.env
  3. Adjust config/config.json if needed
  4. Install dependencies: pip install -r requirements.txt
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config/config.json",
        help="Path to configuration file (default: config/config.json)"
    )
    
    return parser.parse_args()


def check_prerequisites():
    """
    Check that all prerequisites are met before running simulation.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    issues = []
    
    # Check if .env file exists
    env_file = Path("config/.env")
    if not env_file.exists():
        issues.append(
            "❌ config/.env file not found. "
            "Copy config/.env.example to config/.env and add your API key."
        )
    
    # Check if config files exist
    config_file = Path("config/config.json")
    if not config_file.exists():
        issues.append("❌ config/config.json not found")
    
    agents_file = Path("config/agents.csv")
    if not agents_file.exists():
        issues.append("❌ config/agents.csv not found")
    
    # Check if required Python packages are installed
    try:
        import google.generativeai
        import pandas
        import dotenv
    except ImportError as e:
        issues.append(f"❌ Missing required package: {e.name}. Run: pip install -r requirements.txt")
    
    if issues:
        print("⚠️  Prerequisites not met:")
        for issue in issues:
            print(f"   {issue}")
        print("\nPlease fix these issues before running the simulation.")
        return False
    
    return True


def main():
    """Main entry point."""
    print("MEV LLM Economic Simulation")
    print("="*50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    print("✅ All prerequisites met. Starting simulation...")
    print()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Run the simulation
        run_simulation()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Simulation failed with error: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print("   Run with --verbose for detailed error information")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
