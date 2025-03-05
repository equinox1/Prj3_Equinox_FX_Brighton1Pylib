import json
import os
import logging
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

    def get_platform_path(self):
        """Sets the path to the MetaTrader platform."""
        # environment settings
        self.config_platform = self.get("default_platform", "Windows")
        if self.config_platform == "Windows":
            mp_pl_platform_base = 'equinrun'
        elif self.config_platform == "Linux":
            mp_pl_platform_base = 'equinrunlin'
        else:
            self.config_platform = "MacOS"
            mp_pl_platform_base = 'equinrunmac'

        mp_pl_src_base = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/"
        mp_pl_platform_base = mp_pl_platform_base
        mp_pl_src_lib = r"PythonLib"
        mp_pl_src_data = r"tsModelData"
        mp_pl_directory = r"tshybrid_ensemble_tuning_prod"
        mp_pl_baseuniq = str(1)  # str(mp_random)
        mp_pl_logdir = r"logs"
        mp_pl_num_base_path = os.path.join(mp_pl_src_base, mp_pl_platform_base, mp_pl_src_lib, mp_pl_src_data, mp_pl_directory, mp_pl_baseuniq)
        LOG_DIR = os.path.join(mp_pl_num_base_path, mp_pl_logdir)
        return LOG_DIR

# Create a config instance
config = Config()
LOG_DIR = config.get_platform_path()
# Create a logs directory if it doesn’t exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging

LOG_FILE = os.path.join(LOG_DIR, "platform_helper.log")

# Delete the log file if it exists
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Create logger instance
logger = logging.getLogger("MqlPlatform")

