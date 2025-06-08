# ✅ oracle_server_main.py

import os
import sys
import time
from pathlib import Path
import threading
import logging
import warnings
from rich.logging import RichHandler # Keep this import for RichHandler if still used
import uvicorn
from datetime import datetime, date

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Optional
import time

from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlOverrides import CMqlOverrides
from tsMqlSetup import CMqlSetup

# Initialize CMqlSetup to ensure logging is configured correctly for this process
# Retrieve global logfile path and logdir from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
GLOBAL_LOGDIR_PATH = os.environ.get('GLOBAL_LOGDIR_PATH') # Also get logdir for CustomOracle

clientlog_config = CMqlSetup()
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for OracleServer. Using default logging.")

logger = logging.getLogger(__name__)

# -- Suppress ONNX Windows version warning --
warnings.filterwarnings("ignore", message="Unsupported Windows version")

# -- Load environment variables first --
# Get parameters from environment or defaults
env_trials = int(os.environ.get("MLTUNE_TRIALS", 128))
# Retrieve backend and tuner_type from environment set by multiworker_launcher
backend = os.environ.get('MLTUNE_BACKEND', 'tensorflow').lower()
tuner_model = os.environ.get('TUNER_TYPE', 'hyperband') # Get tuner_type from env as well

# -- Apply overrides from mql_overrides.env --
mql_overrides = CMqlOverrides()
# No need to override backend/tuner_type here if they are passed via environment vars from launcher
# If there are other mltune or app parameters you want to override from env, do it here.
# For now, let's just make sure mql_overrides's internal state reflects the environment ones.
mql_overrides.env.override_params({
    "mltune": {
        "backend": backend, # Ensure mql_overrides knows the current backend
        "num_trials": env_trials,
        "tuner_type": tuner_model, # Ensure mql_overrides knows the current tuner_type
        "reset_trials": mql_overrides.env.all_params().get("mltune", {}).get("reset_trials", True), # Keep existing if set
        "overwrite": mql_overrides.env.all_params().get("mltune", {}).get("overwrite", True),
        "tuner_id": os.environ.get('TUNER_ID', 'chief') # Use the tuner_id from environment
    },
})

# Re-fetch potentially overridden params for local use
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})

# Use parameters for OracleServer setup
num_trials = tune_params.get('num_trials', 128)
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = int(app_params.get('xerces_port', 9000))
tuner_id_from_env = os.environ.get('TUNER_ID', 'oracle_default') # Get tuner_id from environment

# Define the directory for the Oracle's internal state
# Use GLOBAL_LOGDIR_PATH for consistency
if GLOBAL_LOGDIR_PATH:
    oracle_dir = Path(GLOBAL_LOGDIR_PATH) / f"oracle_state_{backend}" / tuner_id_from_env
else:
    # Fallback if GLOBAL_LOGDIR_PATH is not set (should not happen if launcher sets it)
    oracle_dir = Path(os.getcwd()) / "oracle_state" / tuner_id_from_env

oracle_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Oracle state directory: {oracle_dir}")

# -- Uvicorn runner --
def run_uvicorn(app, host, port):
    """Function to run uvicorn in a separate thread."""
    try:
        # Uvicorn's log_config should ideally use the same FileHandler from CMqlSetup
        # For simplicity and to avoid Uvicorn's default formatting interfering,
        # we disable Uvicorn's default access log and let the root logger handle it.
        uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    except Exception as e:
        logger.exception(f"❌ Uvicorn failed to start: {e}")

def clean_stale_trials_from_oracle(logdir, project_name):
    import json
    from pathlib import Path

    # The oracle.json file is inside the project_name directory within the oracle_dir
    oracle_file = Path(logdir) / project_name / "oracle.json"
    
    if not oracle_file.exists():
        logger.warning(f"No oracle.json found at {oracle_file}")
        return

    try:
        with open(oracle_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        trials = data.get("trials", {})
        updated = False

        for trial_id, trial_data in trials.items():
            if trial_data.get("status") == "RUNNING":
                logger.info(f"Marking stale trial {trial_id} as FAILED")
                trial_data["status"] = "FAILED"
                updated = True

        if updated:
            with open(oracle_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Stale trials have been marked as FAILED.")
        else:
            logger.info("No stale trials found to clean.")
    except Exception as e:
        logger.exception(f"Failed to clean stale trials: {e}")

# -- Main launcher --
def main():
    logger.info(f"ServerMain: Using GTuner model: {tuner_model}") # Log updated tuner model
    logger.info(f"ServerMain: Using backend: {backend}") # Log updated backend

    # -- Clean up stale trials from previous runs --
    if tune_params.get("reset_trials", True):
        # Oracle directory for KerasTuner is `directory/project_name`
        oracle_project_path = oracle_dir / tuner_id_from_env
        if oracle_project_path.exists():
            logger.info(f"🗑️ Deleting existing Oracle project directory at {oracle_project_path} for a fresh start.")
            try:
                import shutil
                shutil.rmtree(oracle_project_path)
            except Exception as e:
                logger.error(f"❌ Failed to delete Oracle project directory: {e}")
    else:
        # Only clean stale trials within the specific project's oracle directory
        clean_stale_trials_from_oracle(oracle_dir, tuner_id_from_env)

    try:
        logger.info("🧠 Creating CustomOracle...")
        num_trials_for_oracle = tune_params.get("num_trials", 50)
        
        # Create the Oracle instance
        logger.info(f"Creating CustomOracle with max_trials={num_trials_for_oracle}, directory={oracle_dir}, project_name={tuner_id_from_env}")

        oracle = CustomOracle(
            objective="val_loss",
            max_trials=num_trials_for_oracle,
            directory=str(oracle_dir), # Pass as string
            project_name=tuner_id_from_env,
            reset_trials=tune_params.get('reset_trials', False),
            seed=tune_params.get('seed', 42),
        )

        # Prepopulate trials - only if reset_trials is True or if no trials exist in the oracle
        # The `oracle.trials` property (inherited from KerasTuner's Oracle) will load existing trials if `reset_trials` is False.
        if tune_params.get('reset_trials', False) or not oracle.trials:
            logger.info(f"🧪 Pre-populating {oracle.max_trials} trials in Oracle...")
            for _ in range(oracle.max_trials):
                oracle.create_trial(f"chief_{os.getpid()}") # Use process ID for uniqueness
            oracle.save() # Save the initial state of the Oracle
            logger.info(f"✅ Trial population complete. Oracle now has {len(oracle.trials)} trials.")
        else:
            logger.info(f"⏩ Not resetting trials. Oracle already has {len(oracle.trials)} trials.")


        logger.info(f"🚀 Starting OracleServer at http://{xerces_server}:{xerces_port}")
        server = OracleServer(oracle, tuner_id=tuner_id_from_env) # Pass the Oracle instance and tuner_id

        thread = threading.Thread(target=run_uvicorn, args=(server.app, xerces_server, xerces_port), daemon=True)
        thread.start()

        time.sleep(1)
        if not thread.is_alive():
            logger.error("❌ Uvicorn server thread died immediately after starting. Check configuration or port.")
            raise RuntimeError("Uvicorn failed to start. Check configuration or port.")

        logger.info("[OK] OracleServer is now running and awaiting requests.")
        while True:
            time.sleep(60)

    except Exception as e:
        logger.critical(f"❌ Failed to start OracleServer: {e}", exc_info=True)
        logger.info("⚙️ Cleaning up resources (if any)...")
        # Add any cleanup logic here if necessary
        logger.info("[OK] Cleanup completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

