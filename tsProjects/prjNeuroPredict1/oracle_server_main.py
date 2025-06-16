import os
import sys
import time
from pathlib import Path
import threading
import logging
import warnings
import inspect # Import inspect for debugging module path

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

# Local imports
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlOverrides import CMqlOverrides


# Import the logging setup service directly
from tsMqlLogService import CMqlLogService # Use CMqlLogService, not CMqlSetup for raw logging init

# Load environment variables and app parameters using CMqlOverrides early
# This needs to be done *before* initializing the logger if logger depends on these params
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

# Extract backend for logging path - crucial for correct log file path
# This will be passed to initialize_logging. It can also be obtained from env if passed by launcher.
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__,
    loglevel='INFO',
    # Explicitly set the logfile name to ensure consistency
    logfile='tsneuropredict_app.log',
    # Pass the determined backend so logging goes into the correct subdirectory
    backend=backend_for_log # Pass the backend to the logging setup
)


# Server network configuration from app_params
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)


# Suppress specific KerasTuner warnings that are not relevant to the server's operation
warnings.filterwarnings(
    "ignore",
    message="The `tune_new_entries` and `allow_new_entries` arguments are deprecated."
)

# ----------------------------
# Oracle Initialization
# ----------------------------
import random # Already imported, but ensuring it's available for random.randrange

# Determine the full path for the Oracle directory
# This should be consistent with how base_path is determined in CMqlSetup or multiworker_launcher
oracle_base_dir = Path(base_params.get('mp_glob_sub_ml_src_modeldata', Path(__file__).parent / 'oracle_data'))
model_name = tune_params.get('ml_model_name', 'default_model')
project_id = base_params.get('mp_glob_sub_ml_baseuniq', random.randrange(1, 1024)) # Use mp_glob_sub_ml_baseuniq if available
project_name = f"{model_name}_{project_id}"

oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
logger.info(f"Oracle data directory: {oracle_full_path}")


# DEBUG: Print the path of the CustomOracle module being loaded
logger.debug(f"CustomOracle class loaded from: {inspect.getfile(CustomOracle)}")

# Instantiate CustomOracle
logger.info(f"Initializing CustomOracle with directory: {oracle_full_path.parent}, project_name: {oracle_full_path.name}")
oracle_instance = CustomOracle(
    objective=tune_params.get('objective', "val_loss"),
    max_trials=tune_params.get('num_trials', 50),
    directory=str(oracle_full_path.parent), # directory is the parent of project_name
    project_name=oracle_full_path.name, # project_name is the last part of the path
    seed=tune_params.get('seed', 42),
    overwrite=tune_params.get('overwrite', False) # Pass overwrite flag
)
logger.info("CustomOracle instance created.")


# Instantiate OracleServer and use its FastAPI app
oracle_server = OracleServer(oracle_instance=oracle_instance, tuner_id="oracle_main_server")
app = oracle_server.app # Get the FastAPI app from the OracleServer instance
logger.info("FastAPI app obtained from OracleServer instance.")


# ----------------------------
# Server Runner
# ----------------------------
if __name__ == "__main__":
    # The host and port for the Uvicorn server, typically from configuration
    server_host = xerces_server
    server_port = xerces_port

    logger.info(f"🚀 Attempting to start Oracle Server at http://{server_host}:{server_port}")

    try:
        # Use the OracleServer's start method to run Uvicorn in a separate thread
        oracle_server.start(host=server_host, port=server_port)
        logger.info("✅ Oracle Server start method called. Server running in background thread.")

        # Keep the main thread alive while the server (daemon thread) runs
        # The multiworker_launcher.py will terminate this process when it's done.
        while True:
            time.sleep(1) # Sleep to prevent busy-waiting
    except KeyboardInterrupt:
        logger.info("👋 Oracle Server received KeyboardInterrupt. Shutting down gracefully.")
        oracle_server.stop() # Call stop method if it has clean shutdown logic
        sys.exit(0)
    except Exception as e:
        # Log any unexpected crashes and exit with an error code
        logger.critical(f"❌ Oracle Server crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero status code to signal failure