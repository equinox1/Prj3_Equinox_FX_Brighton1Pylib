import logging
import os
from .config import config

# environment settings
self.config_platform = config.get("default_platform", "Windows")
if self.config_platform == "Windows":
    mp_pl_platform_base = 'equinrun'
elif self.config_platform == "Linux":
    mp_pl_platform_base = 'equinrunlin'
else:
    self.config_platform = "MacOS"
    mp_pl_platform_base = 'equinrunmac'

mp_pl_src_base = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/"
mp_pl_src_lib = r"PythonLib"
mp_pl_src_data = r"tsModelData"
mp_pl_directory = r"tshybrid_ensemble_tuning_prod"
mp_pl_baseuniq = str(1)  # str(mp_random)
mp_pl_logdir = r"logs"
mp_pl_num_base_path = os.path.join(mp_pl_src_base, mp_pl_platform_base, mp_pl_src_lib, mp_pl_src_data, mp_pl_directory, mp_pl_baseuniq)
LOG_DIR = os.path.join(mp_pl_num_base_path, mp_pl_logdir)

# Create a logs directory if it doesn’t exist
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging
LOG_FILE = os.path.join(LOG_DIR, "platform_helper.log")

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# Create logger instance
logger = logging.getLogger("PlatformHelper")
