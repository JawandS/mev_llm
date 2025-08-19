#!/usr/bin/env python3
"""
Main entry point for the MEV LLM Economic Simulation.

This script provides an easy way to run the economic simulation
with proper error handling and user feedback.
"""

import sys
import argparse
import subprocess
import time
import json
import requests
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

Features:
    - Automatically starts and stops Ollama server
    - Downloads required model if not present locally
    - Handles all prerequisites automatically

Before running:
    1. Install Ollama from https://ollama.com/download
    2. Adjust config/config.json if needed (set model_name to desired Ollama model)
    3. Install dependencies: pip install -r requirements.txt
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


def check_ollama_installed():
    """
    Check if Ollama is installed on the system.
    
    Returns:
        bool: True if Ollama is installed, False otherwise
    """
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_ollama_server():
    """
    Start the Ollama server in the background.
    
    Returns:
        subprocess.Popen: The server process, or None if failed
    """
    try:
        print("🚀 Starting Ollama server...")
        process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is responding
        for attempt in range(5):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama server started successfully")
                    return process
            except requests.RequestException:
                time.sleep(1)
        
        # If we get here, server didn't start properly
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Failed to start Ollama server: {e}")
        return None


def stop_ollama_server(process):
    """
    Stop the Ollama server process.
    
    Args:
        process: The server process to stop
    """
    if process:
        try:
            print("🛑 Stopping Ollama server...")
            process.terminate()
            process.wait(timeout=5)
            print("✅ Ollama server stopped")
        except subprocess.TimeoutExpired:
            print("⚠️  Force killing Ollama server...")
            process.kill()
        except Exception as e:
            print(f"⚠️  Error stopping Ollama server: {e}")


def get_required_model():
    """
    Get the required model name from config file.
    
    Returns:
        str: Model name, or None if not found
    """
    try:
        config_file = Path("config/config.json")
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config.get("llm", {}).get("model_name")
    except Exception as e:
        print(f"⚠️  Could not read model name from config: {e}")
        return None


def check_and_pull_model(model_name):
    """
    Check if model exists locally, and pull it if not.
    
    Args:
        model_name: Name of the model to check/pull
        
    Returns:
        bool: True if model is available, False otherwise
    """
    if not model_name:
        print("❌ No model name specified in config")
        return False
    
    try:
        # Check if model exists
        print(f"🔍 Checking for model: {model_name}")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if model_name in model_names:
                print(f"✅ Model {model_name} found locally")
                return True
            else:
                print(f"📥 Model {model_name} not found locally. Downloading...")
                
                # Pull the model
                pull_response = requests.post(
                    "http://localhost:11434/api/pull",
                    json={"name": model_name},
                    timeout=300  # 5 minutes timeout for download
                )
                
                if pull_response.status_code == 200:
                    print(f"✅ Model {model_name} downloaded successfully")
                    return True
                else:
                    print(f"❌ Failed to download model {model_name}")
                    return False
        else:
            print("❌ Could not connect to Ollama server")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Error checking/pulling model: {e}")
        return False


def check_prerequisites():
    """
    Check that all prerequisites are met before running simulation.
    
    Returns:
        bool: True if all prerequisites are met, False otherwise
    """
    issues = []
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        issues.append("❌ Ollama not found. Please install from https://ollama.com/download")
    
    # Check if config files exist
    config_file = Path("config/config.json")
    if not config_file.exists():
        issues.append("❌ config/config.json not found")

    agents_file = Path("config/agents.csv")
    if not agents_file.exists():
        issues.append("❌ config/agents.csv not found")

    # Check if required Python packages are installed
    try:
        import pandas
        import requests
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
    
    print("✅ All prerequisites met")
    
    # Start Ollama server
    ollama_process = start_ollama_server()
    if not ollama_process:
        print("❌ Failed to start Ollama server")
        sys.exit(1)
    
    try:
        # Get required model and ensure it's available
        model_name = get_required_model()
        if not check_and_pull_model(model_name):
            print("❌ Failed to ensure model availability")
            sys.exit(1)
        
        print("🎯 Starting simulation...")
        print()
        
        # Set logging level
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Run the simulation
        run_simulation()
        
        print("\n✅ Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Simulation failed with error: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print("   Run with --verbose for detailed error information")
        
        sys.exit(1)
        
    finally:
        # Always stop the Ollama server
        stop_ollama_server(ollama_process)


if __name__ == "__main__":
    main()
