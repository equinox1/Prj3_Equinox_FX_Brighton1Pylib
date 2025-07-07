import os
import sys
import time
from pathlib import Path
import threading
import logging
import warnings
import inspect

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

# Local imports
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlOverrides import CMqlOverrides

# Corrected: Assign the logger instance to the 'logger' variable
logger = logging.getLogger(__name__)

# Load configuration (needed before logging setup if logging depends on params)
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

# Log the loaded tune_params for debugging
logger.info(f"[oracle_server_main] Loaded tune_params: {tune_params}")


# Server network configuration
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)

# Suppress deprecated warnings
warnings.filterwarnings(
    "ignore",
    message="The `tune_new_entries` and `allow_new_entries` arguments are deprecated."
)

# Oracle setup
import random
# Use the user-specified base log path for oracle data
oracle_base_dir = Path(base_params.get('mp_glob_base_log_path')) / "oracle_server_data" # Create a subdirectory for oracle server data
oracle_base_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

model_name = os.environ.get('ML_MODEL_NAME', tune_params.get('ml_model_name', 'default_model'))
project_id = os.environ.get('ML_PROJECT_ID', base_params.get('mp_glob_sub_ml_baseuniq', random.randrange(1, 1024)))
project_name = f"{model_name}_{project_id}"

oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Oracle data directory: {oracle_full_path}")

logger.debug(f"CustomOracle class loaded from: {inspect.getfile(CustomOracle)}")

# Determine the 'overwrite' value. Prioritize environment variable.
env_overwrite_str = os.environ.get('MLTUNE_OVERWRITE')

# Log the raw environment variable value for debugging
logger.info(f"[oracle_server_main] Raw MLTUNE_OVERWRITE from environment: '{env_overwrite_str}'")

# Force overwrite to True if the environment variable is explicitly 'True' (case-insensitive)
# Otherwise, use the value from tune_params or default to False.
overwrite_value = False
if env_overwrite_str and env_overwrite_str.lower() == 'true':
    overwrite_value = True
else:
    overwrite_value = tune_params.get('overwrite', False)

logger.info(f"[oracle_server_main] tune_params.get('overwrite'): {tune_params.get('overwrite')}")
logger.info(f"[oracle_server_main] Final overwrite value passed to CustomOracle: {overwrite_value} (type: {type(overwrite_value)})")


oracle_instance = CustomOracle(
    objective=tune_params.get('objective', "val_loss"),
    max_trials=tune_params.get('num_trials', 50),
    directory=str(oracle_base_dir), # Pass the base directory for Oracle
    project_name=project_name, # Pass the project name
    seed=tune_params.get('seed', 42),
    overwrite=overwrite_value # Pass the determined overwrite value
)
logger.info("CustomOracle instance created.")

oracle_server = OracleServer(oracle_instance=oracle_instance, tuner_id="oracle_main_server")
app = oracle_server.app
logger.info("FastAPI app obtained from OracleServer instance.")

# ----------------------------
# Server Runner (Patched)
# ----------------------------
if __name__ == "__main__":
    import asyncio
    import platform

    # ✅ Fix for Windows asyncio event loop bug
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logger.info(f"🚀 Attempting to start Oracle Server at http://{xerces_server}:{xerces_port}")

    try:
        oracle_server.start(host=xerces_server, port=xerces_port)
        logger.info("✅ Oracle Server start method called. Server running in background thread.")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("👋 Oracle Server received KeyboardInterrupt. Shutting down gracefully.")
        oracle_server.stop()
        sys.exit(0)
    except Exception as e:
        logger.critical(f"❌ Oracle Server crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1)
