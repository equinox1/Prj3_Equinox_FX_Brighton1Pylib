import json
import yaml
import os
import logging
from pathlib import Path
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
        logger.info(f"BaseParams: Looking for CONFIG_JSON_FILE: {CONFIG_JSON_FILE}")
        logger.info(f"BaseParams: Looking for CONFIG_YAML_FILE: {CONFIG_YAML_FILE}")
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

# Global configuration defaults (can be overridden by the config files or keyword arguments)
mp_glob_sub_win_drive = 'C://'
mp_glob_sub_wsl_drive = '/mnt/c/'
mp_glob_sub_lin_drive = '/home'
mp_glob_mac_drive = '/Users/'
mp_glob_sub_user = 'shepa'
mp_glob_sub_netdrive = 'OneDrive'
mp_glob_sub_dir1 = '8.0 Projects'
mp_glob_sub_dir2 = '8.3 ProjectModelsEquinox'
mp_glob_sub_data = 'Mql5Data'
mp_glob_sub_ml_src_lib = 'PythonLib'
mp_glob_sub_ml_src_modeldata = 'tsModelData'
mp_glob_sub_ml_src_configdata = 'tsConfigData'
mp_glob_sub_ml_directory = 'tshybrid_ensemble_tuning_prod'
mp_glob_sub_ml_model_name = 'prjEquinox1_prod.keras'
mp_glob_sub_ml_baseuniq = '1'
mp_glob_sub_mql_dir1 = 'Mql5'
mp_glob_sub_mql_dir2 = 'Files'
mp_glob_project_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox'
mp_glob_full_platform_win_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN'
mp_glob_full_platform_lin_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUNLIN'
mp_glob_full_platform_mac_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUNLMAC'
mp_glob_config_data = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\ConfigData'
mp_glob_full_python_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib'
mp_glob_full_model_data = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\tsModelData'
mp_glob_full_model_project = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\tsModelData\tshybrid_ensemble_tuning_prod'
mp_glob_full_model_sequence = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\tsModelData\tshybrid_ensemble_tuning_prod\1'
mp_glob_full_model_logs = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\PythonLib\tsModelData\tshybrid_ensemble_tuning_prod\1\logs'
mp_glob_full_mql5_base = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\MQL5\Brokers\Metaquotes\MQL5'
mp_glob_full_mql5_files = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\MQL5\Brokers\Metaquotes\MQL5\Files'
mp_glob_full_mql5_include = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\MQL5\Brokers\Metaquotes\MQL5\Include'
mp_glob_full_mql5_lib = r'C:\Users\shepa\OneDrive\8.0 Projects\8.3 ProjectModelsEquinox\EQUINRUN\MQL5\Brokers\Metaquotes\MQL5\Libraries'
