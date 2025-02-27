import json
import yaml
import os
import logging
from .logger import logger

# Define paths for configuration files
CONFIG_JSON_FILE = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_YAML_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

class Config:
    def __init__(self):
        self.config = {}
        self.load_config()

    def load_config(self):
        """Loads configuration from JSON and YAML files."""
        # Load JSON config if available
        if os.path.exists(CONFIG_JSON_FILE):
            try:
                with open(CONFIG_JSON_FILE, "r") as f:
                    json_config = json.load(f)
                self.config.update(json_config)
                logger.info("Config JSON file loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading config.json: {e}")
        
        # Load YAML config if available
        if os.path.exists(CONFIG_YAML_FILE):
            try:
                with open(CONFIG_YAML_FILE, "r") as f:
                    yaml_config = yaml.safe_load(f)
                if yaml_config:
                    self.config.update(yaml_config)
                logger.info("Config YAML file loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading config.yaml: {e}")

    def get(self, key, default=None):
        """Gets a configuration value with a fallback default."""
        return self.config.get(key, default)

# Create a config instance
config = Config()
