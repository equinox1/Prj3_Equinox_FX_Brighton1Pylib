# __config__.py
import json
import yaml
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default paths
CONFIG_JSON_FILE = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_YAML_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

class BaseConfig:
    def __init__(self, config_file=None):
        self.config_file = config_file
        self.config = {}
        self.load_config(config_file)

    def load_config(self, config_file=None):
        """Loads configuration from a specified file, or from defaults."""
        if config_file:
            ext = Path(config_file).suffix.lower()
            try:
                with open(config_file, 'r') as f:
                    if ext == '.json':
                        self.config.update(json.load(f))
                    elif ext in ('.yaml', '.yml'):
                        self.config.update(yaml.safe_load(f))
                logger.info(f"Loaded config from {config_file}")
            except Exception as e:
                logger.error(f"Failed to load config from {config_file}: {e}")
        
        # Load default JSON config if no file or fallback
        if os.path.exists(CONFIG_JSON_FILE):
            try:
                with open(CONFIG_JSON_FILE, "r") as f:
                    json_config = json.load(f)
                self.config.update(json_config)
                logger.info("Config JSON file loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading config.json: {e}")

        # Load default YAML config
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

# Instantiate default config
config = BaseConfig()