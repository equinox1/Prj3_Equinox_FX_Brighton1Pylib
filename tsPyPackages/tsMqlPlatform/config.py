import json
import os
from .logger import logger

# Load configuration from a JSON file
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

class Config:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        """Loads configuration from a JSON file."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config_data = json.load(f)
                logger.info("Config file loaded successfully.")
                return config_data
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
        return {}

    def get(self, key, default=None):
        """Gets a configuration value with a fallback default."""
        return self.config.get(key, default)

# Create a config instance
config = Config()
