# tsMqlPlatform/config.py

import json
import yaml
import os
import logging

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONFIG_JSON_FILE = os.path.join(os.path.dirname(__file__), "config.json")
CONFIG_YAML_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")

class PlatConfig:
    def __init__(self):
        self.config = {}
        self.load_config()

    def load_config(self):
        logger.info(f"Platform: Looking for CONFIG_JSON_FILE: {CONFIG_JSON_FILE}")
        logger.info(f"Platform: Looking for CONFIG_YAML_FILE: {CONFIG_YAML_FILE}")
        
        if os.path.exists(CONFIG_JSON_FILE):
            try:
                with open(CONFIG_JSON_FILE, "r") as f:
                    json_config = json.load(f)
                self.config.update(json_config)
                logger.info("Config JSON file loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading config.json: {e}")

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
        return self.config.get(key, default)

# Singleton instance
_config_instance = None

def get_config():
    global _config_instance
    if _config_instance is None:
        _config_instance = PlatConfig()
    return _config_instance
