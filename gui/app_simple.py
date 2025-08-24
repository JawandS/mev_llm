#!/usr/bin/env python3
"""
Simple Flask web interface for MacroEconVue configuration management.

This module provides a web-based interface for:
- Editing simulation configuration (config.json)
- Managing agent types and their parameters (agents.csv)
"""

import json
import csv
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = 'mev_llm_config_key_2025'

# Configuration file paths
CONFIG_DIR = Path(__file__).parent.parent / "config"
CONFIG_JSON_PATH = CONFIG_DIR / "config.json"
AGENTS_CSV_PATH = CONFIG_DIR / "agents.csv"

class ConfigManager:
    """Handles reading and writing configuration files."""
    
    @staticmethod
    def load_config():
        """Load configuration from JSON file."""
        try:
            with open(CONFIG_JSON_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config_data):
        """Save configuration to JSON file."""
        try:
            with open(CONFIG_JSON_PATH, 'w') as f:
                json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    @staticmethod
    def load_agents():
        """Load agents from CSV file."""
        agents = []
        try:
            with open(AGENTS_CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    agents.append(row)
        except Exception as e:
            print(f"Error loading agents: {e}")
        return agents
    
    @staticmethod
    def save_agents(agents_data):
        """Save agents to CSV file."""
        try:
            if not agents_data:
                return False
            
            fieldnames = agents_data[0].keys()
            with open(AGENTS_CSV_PATH, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(agents_data)
            return True
        except Exception as e:
            print(f"Error saving agents: {e}")
            return False

@app.route('/')
def index():
    """Main configuration page."""
    config = ConfigManager.load_config()
    agents = ConfigManager.load_agents()
    return render_template('config.html', config=config, agents=agents)

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """API endpoint for configuration management."""
    if request.method == 'GET':
        return jsonify(ConfigManager.load_config())
    
    elif request.method == 'POST':
        try:
            config_data = request.get_json()
            if ConfigManager.save_config(config_data):
                return jsonify({'success': True, 'message': 'Configuration saved successfully'})
            else:
                return jsonify({'success': False, 'message': 'Failed to save configuration'}), 500
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/agents', methods=['GET', 'POST'])
def api_agents():
    """API endpoint for agent management."""
    if request.method == 'GET':
        return jsonify(ConfigManager.load_agents())
    
    elif request.method == 'POST':
        try:
            agents_data = request.get_json()
            if ConfigManager.save_agents(agents_data):
                return jsonify({'success': True, 'message': 'Agents saved successfully'})
            else:
                return jsonify({'success': False, 'message': 'Failed to save agents'}), 500
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 400

if __name__ == '__main__':
    print("🌐 MacroEconVue Configuration Interface")
    print("📝 Open your browser and go to: http://localhost:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)
