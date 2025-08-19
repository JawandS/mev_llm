"""
Tests for the utils module.

This module contains unit tests for all utility functions used in the
MEV LLM Economic Simulation.
"""

import pytest
import tempfile
import json
import csv
from pathlib import Path
import pandas as pd
from unittest.mock import patch, mock_open
from datetime import datetime

import sys
sys.path.append('../src')

from src.utils import (
    setup_logging, load_config, load_agent_types, create_results_directory,
    save_agent_chat_log, save_agents_summary, save_transactions, validate_config
)


class TestLogging:
    """Test logging configuration."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        logger = setup_logging()
        assert logger.name == "src.utils"
    
    def test_setup_logging_custom_level(self):
        """Test logging setup with custom level."""
        logger = setup_logging("DEBUG")
        assert logger.name == "src.utils"


class TestConfigLoading:
    """Test configuration loading functions."""
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        config_data = {
            "simulation": {"periods": 12, "agents_per_type": 5},
            "economics": {"interest_rate": 0.02, "luxury_cost_per_unit": 50.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            with patch('src.utils.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                with patch('builtins.open', mock_open(read_data=json.dumps(config_data))):
                    config = load_config()
                    assert config == config_data
        finally:
            Path(temp_path).unlink()
    
    def test_load_config_file_not_found(self):
        """Test loading configuration when file doesn't exist."""
        with patch('src.utils.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            with pytest.raises(FileNotFoundError):
                load_config()
    
    def test_load_agent_types_valid(self):
        """Test loading valid agent types."""
        csv_data = "agent_type,income,fixed_cost,variable_cost\nyoung_professional,1200.0,400.0,400.0\n"
        
        with patch('src.utils.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('pandas.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({
                    'agent_type': ['young_professional'],
                    'income': [1200.0],
                    'fixed_cost': [400.0],
                    'variable_cost': [400.0]
                })
                mock_read_csv.return_value = mock_df
                
                result = load_agent_types()
                assert len(result) == 1
                assert result.iloc[0]['agent_type'] == 'young_professional'
    
    def test_load_agent_types_missing_columns(self):
        """Test loading agent types with missing required columns."""
        with patch('src.utils.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('pandas.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({'agent_type': ['test']})  # Missing required columns
                mock_read_csv.return_value = mock_df
                
                with pytest.raises(ValueError, match="must contain columns"):
                    load_agent_types()


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = {
            "simulation": {"periods": 12, "agents_per_type": 5},
            "economics": {"interest_rate": 0.02, "luxury_cost_per_unit": 50.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        # Should not raise any exception
        validate_config(config)
    
    def test_validate_config_missing_section(self):
        """Test validation with missing section."""
        config = {
            "simulation": {"periods": 12, "agents_per_type": 5},
            # Missing economics and llm sections
        }
        with pytest.raises(ValueError, match="Missing configuration section"):
            validate_config(config)
    
    def test_validate_config_missing_key(self):
        """Test validation with missing key."""
        config = {
            "simulation": {"periods": 12},  # Missing agents_per_type
            "economics": {"interest_rate": 0.02, "luxury_cost_per_unit": 50.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        with pytest.raises(ValueError, match="Missing configuration key"):
            validate_config(config)
    
    def test_validate_config_invalid_values(self):
        """Test validation with invalid values."""
        # Test negative periods
        config = {
            "simulation": {"periods": -1, "agents_per_type": 5},
            "economics": {"interest_rate": 0.02, "luxury_cost_per_unit": 50.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        with pytest.raises(ValueError, match="periods must be positive"):
            validate_config(config)
        
        # Test invalid interest rate
        config["simulation"]["periods"] = 12
        config["economics"]["interest_rate"] = 1.5  # > 1
        with pytest.raises(ValueError, match="interest_rate must be between 0 and 1"):
            validate_config(config)


class TestResultsHandling:
    """Test results directory and file handling."""
    
    def test_create_results_directory(self):
        """Test creation of results directory."""
        with patch('src.utils.datetime') as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "01-01-25_12:00:00"
            
            result = create_results_directory()
            
            # Check that the result contains the expected path structure
            assert "results/01-01-25_12:00:00" in str(result)
            # The directory should exist (this will create it for real)
            assert result.exists()
    
    def test_save_agent_chat_log(self):
        """Test saving agent chat log."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            chat_log_dir = results_dir / "chat_log"
            chat_log_dir.mkdir()
            
            chat_history = [
                {"period": 0, "prompt": "test prompt", "response": "test response"}
            ]
            
            save_agent_chat_log(results_dir, 1, chat_history)
            
            # Check file was created
            log_file = chat_log_dir / "agent_1.json"
            assert log_file.exists()
            
            # Check content
            with open(log_file, 'r') as f:
                saved_data = json.load(f)
                assert saved_data == chat_history
    
    def test_save_agents_summary(self):
        """Test saving agents summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            chat_log_dir = results_dir / "chat_log"
            chat_log_dir.mkdir()
            
            agents_info = [(1, "young_professional"), (2, "family")]
            
            save_agents_summary(results_dir, agents_info)
            
            # Check file was created
            summary_file = chat_log_dir / "agents.csv"
            assert summary_file.exists()
            
            # Check content
            with open(summary_file, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert rows[0] == ['agent_id', 'agent_type']
                assert rows[1] == ['1', 'young_professional']
                assert rows[2] == ['2', 'family']
    
    def test_save_transactions_empty(self):
        """Test saving empty transactions list."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            
            save_transactions(results_dir, [])
            
            # Check file was created with headers
            transactions_file = results_dir / "transactions.csv"
            assert transactions_file.exists()
            
            with open(transactions_file, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)
                assert headers == [
                    'period_num', 'agent_id', 'agent_type', 'purchase_type', 
                    'purchase_quantity', 'required'
                ]
    
    def test_save_transactions_with_data(self):
        """Test saving transactions with data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            
            transactions = [
                {
                    'period_num': 0,
                    'agent_id': 1,
                    'agent_type': 'young_professional',
                    'purchase_type': 'luxury',
                    'purchase_quantity': 2,
                    'required': False
                }
            ]
            
            save_transactions(results_dir, transactions)
            
            # Check file content
            transactions_file = results_dir / "transactions.csv"
            df = pd.read_csv(transactions_file)
            
            assert len(df) == 1
            assert df.iloc[0]['agent_id'] == 1
            assert df.iloc[0]['purchase_type'] == 'luxury'


if __name__ == "__main__":
    pytest.main([__file__])
