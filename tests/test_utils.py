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
    
    @classmethod
    def setup_class(cls):
        """Load actual configuration for testing."""
        cls.actual_config = load_config()
        cls.actual_agent_types = load_agent_types()
    
    def test_load_config_valid(self):
        """Test loading valid configuration."""
        # Use actual config values for testing
        config_data = {
            "simulation": {"periods": self.actual_config['simulation']['periods'], "agents_per_type": self.actual_config['simulation']['agents_per_type']},
            "economics": {"interest_rate": self.actual_config['economics']['interest_rate'], "luxury_cost_per_unit": self.actual_config['economics']['luxury_cost_per_unit']},
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
        # Get first row from actual data for testing
        first_agent = self.actual_agent_types.iloc[0]
        csv_data = f"agent_type,income,fixed_cost,variable_cost\n{first_agent['agent_type']},{first_agent['income']},{first_agent['fixed_cost']},{first_agent['variable_cost']}\n"
        
        with patch('src.utils.Path') as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch('pandas.read_csv') as mock_read_csv:
                mock_df = pd.DataFrame({
                    'agent_type': [first_agent['agent_type']],
                    'income': [first_agent['income']],
                    'fixed_cost': [first_agent['fixed_cost']],
                    'variable_cost': [first_agent['variable_cost']]
                })
                mock_read_csv.return_value = mock_df
                
                result = load_agent_types()
                assert len(result) == 1
                assert result.iloc[0]['agent_type'] == first_agent['agent_type']
    
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
    
    @classmethod
    def setup_class(cls):
        """Load actual configuration for testing."""
        cls.actual_config = load_config()
    
    def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = {
            "simulation": {"periods": self.actual_config['simulation']['periods'], "agents_per_type": self.actual_config['simulation']['agents_per_type']},
            "economics": {"interest_rate": self.actual_config['economics']['interest_rate'], "luxury_cost_per_unit": self.actual_config['economics']['luxury_cost_per_unit']},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        # Should not raise any exception
        validate_config(config)
    
    def test_validate_config_missing_section(self):
        """Test validation with missing section."""
        config = {
            "simulation": {"periods": self.actual_config['simulation']['periods'], "agents_per_type": self.actual_config['simulation']['agents_per_type']},
            # Missing economics and llm sections
        }
        with pytest.raises(ValueError, match="Missing configuration section"):
            validate_config(config)
    
    def test_validate_config_missing_key(self):
        """Test validation with missing key."""
        config = {
            "simulation": {"periods": self.actual_config['simulation']['periods']},  # Missing agents_per_type
            "economics": {"interest_rate": self.actual_config['economics']['interest_rate'], "luxury_cost_per_unit": self.actual_config['economics']['luxury_cost_per_unit']},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        with pytest.raises(ValueError, match="Missing configuration key"):
            validate_config(config)
    
    def test_validate_config_invalid_values(self):
        """Test validation with invalid values."""
        # Test negative periods
        config = {
            "simulation": {"periods": -1, "agents_per_type": self.actual_config['simulation']['agents_per_type']},
            "economics": {"interest_rate": self.actual_config['economics']['interest_rate'], "luxury_cost_per_unit": self.actual_config['economics']['luxury_cost_per_unit']},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        with pytest.raises(ValueError, match="periods must be positive"):
            validate_config(config)
        
        # Test invalid interest rate
        config["simulation"]["periods"] = 12
        config["economics"]["interest_rate"] = 1.5  # > 1
        with pytest.raises(ValueError, match="interest_rate must be between 0 and 1"):
            validate_config(config)

    def test_validate_config_agents_per_type_formats(self):
        """Test validation of different agents_per_type formats."""
        base_config = {
            "simulation": {"periods": 12},
            "economics": {"interest_rate": 0.04, "luxury_cost_per_unit": 12.0},
            "llm": {"model_name": "gemini-2.0-flash-exp", "temperature": 0.7, "max_tokens": 1000}
        }
        
        # Test valid integer format
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = 2
        validate_config(config)  # Should not raise
        
        # Test valid dictionary format
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = {"young_professional": 2, "family": 1}
        validate_config(config)  # Should not raise
        
        # Test negative integer
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = -1
        with pytest.raises(ValueError, match="agents_per_type must be positive"):
            validate_config(config)
        
        # Test zero integer (should pass)
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = 0
        with pytest.raises(ValueError, match="agents_per_type must be positive"):
            validate_config(config)
        
        # Test empty dictionary
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = {}
        with pytest.raises(ValueError, match="agents_per_type dictionary cannot be empty"):
            validate_config(config)
        
        # Test dictionary with negative values
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = {"young_professional": -1}
        with pytest.raises(ValueError, match="agents_per_type\\[young_professional\\] must be a non-negative integer"):
            validate_config(config)
        
        # Test dictionary with non-integer values
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = {"young_professional": "invalid"}
        with pytest.raises(ValueError, match="agents_per_type\\[young_professional\\] must be a non-negative integer"):
            validate_config(config)
        
        # Test dictionary with zero values (should pass)
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = {"young_professional": 0, "family": 1}
        validate_config(config)  # Should not raise
        
        # Test invalid type (neither int nor dict)
        config = base_config.copy()
        config["simulation"]["agents_per_type"] = "invalid"
        with pytest.raises(ValueError, match="agents_per_type must be either an integer or a dictionary"):
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
