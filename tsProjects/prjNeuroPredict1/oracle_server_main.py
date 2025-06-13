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
# CORRECTED: Import OracleServer from tsMqlMLOracleServer
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlOverrides import CMqlOverrides
from tsMqlSetup import CMqlSetup
import uvicorn # Ensure uvicorn is imported if running directly

# ----------------------------
# Global Configuration & Logger Setup
# ----------------------------

# Load environment variables and app parameters using CMqlOverrides early
mql_overrides = CMqlOverrides()
# Define all_params once
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {}) # Now 'all_params' is defined

# Server network configuration from app_params
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)

# Extract backend for logging path
backend = tune_params.get('backend', 'pytorch') # Assuming PyTorch as default for this context

# Retrieve global log file and directory paths from environment variables.
# These variables are expected to be set by the multiworker_launcher.
final_logdir = os.environ.get('GLOBAL_LOGDIR_PATH')
final_logfile_path = os.environ.get('GLOBAL_LOGFILE_PATH')

# Critical check: Ensure the global log directory is set
if not final_logdir:
    # Fallback or exit if essential environment variables are not set
    print("CRITICAL ERROR: GLOBAL_LOGDIR_PATH environment variable not set. Exiting.")
    sys.exit(1)

# Configure logging for the Oracle server
# This setup ensures logs go to the designated global log file
log_file_path = Path(final_logdir) / xerces_servername / backend / app_params.get('xerces_logfile', 'tsneuropredict_app.log')
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Basic logging configuration for the Oracle server
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Also log to console
    ]
)
logger = logging.getLogger(__name__)
logger.info("Oracle Server Logging initialized.")


# Suppress specific KerasTuner warnings that are not relevant to the server's operation
warnings.filterwarnings(
    "ignore",
    message="The `tune_new_entries` and `allow_new_entries` arguments are deprecated."
)

# ----------------------------
# KerasTuner Oracle Instance
# ----------------------------
import random
# Create a unique project directory for the Oracle based on parameters
# This ensures that different runs/backends don't collide with Oracle's internal files
model_name = tune_params.get('ml_model_name', 'default_model') # Default model name if not specified
# Project ID should ideally come from configuration or be dynamically generated for distinct runs
project_id = random.randrange(1,1024) # Use method from env manager
project_name = f"{model_name}_{project_id}" # Combined project name

# Determine the Oracle's directory based on the global model data path
oracle_base_dir = base_params.get('mp_glob_sub_ml_src_modeldata', Path(final_logdir) / 'oracle_data')
oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

# DEBUG: Print the path of the CustomOracle module being loaded
logger.info(f"DEBUG: CustomOracle class loaded from: {inspect.getfile(CustomOracle)}")

# Instantiate CustomOracle
logger.info(f"Initializing CustomOracle with directory: {oracle_full_path}, project_name: {project_name}")
oracle_instance = CustomOracle(
    objective="val_loss",
    max_trials=tune_params.get('num_trials', 50),
    directory=str(oracle_full_path), # Pass as string
    project_name=project_name,
    seed=tune_params.get('seed', 42),
    overwrite=tune_params.get('overwrite', False) # Pass overwrite flag
)
logger.info("CustomOracle instance created.")


# CORRECTED: Instantiate OracleServer and use its FastAPI app
# This replaces the direct FastAPI app creation and route definitions in oracle_server_main.py
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
        # This will block and run the Uvicorn server until it's manually stopped (e.g., Ctrl+C)
        # Use the 'app' object retrieved from oracle_server.app
        uvicorn.run(app, host=server_host, port=server_port)
        logger.info("✅ Oracle Server shut down gracefully.")
    except Exception as e:
        # Log any unexpected crashes and exit with an error code
        logger.critical(f"❌ Oracle Server crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero status code to signal failure