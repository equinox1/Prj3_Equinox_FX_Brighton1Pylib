import os
import sys
import time
from pathlib import Path
import threading
import logging
import warnings

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Optional

# Local imports
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlOverrides import CMqlOverrides
from tsMqlSetup import CMqlSetup
import uvicorn # Ensure uvicorn is imported if running directly

# ----------------------------
# Global Configuration & Logger Setup
# ----------------------------

# Load environment variables and app parameters using CMqlOverrides early
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})

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

# Critical check: Ensure the log paths are available from environment variables.
if not final_logdir or not final_logfile_path:
    print(f"CRITICAL ERROR: GLOBAL_LOGDIR_PATH ({final_logdir}) or GLOBAL_LOGFILE_PATH ({final_logfile_path}) "
          f"environment variables are not set. Ensure multiworker_launcher.py is run correctly.", file=sys.stderr)
    sys.exit(1)

# Configure logging only if the root logger has no handlers already.
# This prevents repeated "Logging initialized" messages and duplicate handlers,
# while ensuring the logger is set up if it hasn't been by another module.
root_logger = logging.getLogger()
logger_initial_print_message = "" # Initialize message for the log
if not root_logger.handlers:
    try:
        # Use a new CMqlSetup instance for actual logging setup if needed (good practice).
        clientlog_config_actual = CMqlSetup()
        clientlog_config_actual.setup_logging(logfile=final_logfile_path) # Use the determined path
        logger_initial_print_message = f"Logging initialized. Logfile: {final_logfile_path}"
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to configure logging via CMqlSetup: {e}", file=sys.stderr)
        sys.exit(1)
else:
    # If handlers already exist, logging was configured by another module.
    logger_initial_print_message = "Logging already configured. Skipping re-initialization."

# Get the logger instance for this specific module.
# This should always be done AFTER the root logger has been configured.
logger = logging.getLogger(__name__)
# Log the initial message, confirming whether logging was set up or already existed.
logger.info(logger_initial_print_message)

# Suppress ONNX Windows version warning, if applicable
warnings.filterwarnings("ignore", message="Unsupported Windows version")

# Extract necessary configuration for CustomOracle from tune_params
max_trials = tune_params.get('num_trials', int(os.environ.get("MLTUNE_TRIALS", 128)))
project_name = app_params.get('project_name', 'PyTorchTuningProject')
objective_name = tune_params.get('objective', 'val_loss')

# The Oracle's state directory will be a sub-directory within the final_logdir.
# 'final_logdir' is guaranteed to be set at this point by retrieving from env vars.
oracle_directory = Path(final_logdir) / "oracle_state"

# Ensure the Oracle's state directory exists before initializing the Oracle.
try:
    oracle_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Oracle state directory ensured: {oracle_directory}")
except Exception as e:
    logger.critical(f"❌ Failed to create Oracle state directory {oracle_directory}: {e}")
    sys.exit(1) # Exit if the necessary directory cannot be created

# ----------------------------
# FastAPI Application and Oracle Instance
# ----------------------------
app = FastAPI()

# Initialize CustomOracle with all required parameters.
# The 'directory' parameter must be a string, which oracle_directory is (after Path conversion).
oracle = CustomOracle(
    objective=objective_name,
    max_trials=max_trials,
    directory=str(oracle_directory), # Pass the resolved path as a string
    project_name=project_name,
    seed=None, # Set a specific seed (e.g., 42) for reproducible hyperparameter search if needed
    reset_trials=True # Set to False if you want to resume previous tuning runs from disk
)

# Initialize a threading.Lock for thread-safe access to the Oracle instance,
# as FastAPI handlers might be executed concurrently.
lock = threading.Lock()

# ----------------------------
# Pydantic Request/Response Models
# ----------------------------
# Define Pydantic models for request body validation and response serialization.

class TrialRequest(BaseModel):
    tuner_id: str # Identifier for the tuner requesting a trial

class TrialScore(BaseModel):
    trial_id: str
    score: float # Numerical score (e.g., validation loss, accuracy)

class TrialResult(BaseModel):
    trial_id: str
    # 'result' should be a dictionary of metrics, compatible with KerasTuner's update_trial
    result: Dict[str, float] # Example: {'val_loss': 0.05, 'val_accuracy': 0.92}

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str # Status of the trial (e.g., "RUNNING", "COMPLETED", "STOPPED", "INVALID")

# ----------------------------
# FastAPI Endpoints
# ----------------------------

@app.post("/request_trial")
def handle_request_trial(request: TrialRequest):
    """
    Endpoint for a tuner to request a new trial (hyperparameter configuration).
    The Oracle generates a new set of hyperparameters based on its search algorithm.
    """
    with lock: # Acquire lock for thread-safe access to the Oracle
        logger.info(f"Request for new trial from tuner_id: {request.tuner_id}")
        # KerasTuner's Oracle.create_trial handles the population of hyperparameters internally
        trial = oracle.create_trial(request.tuner_id)
        
        if trial is None:
            # If no trial is available (e.g., max_trials reached, or no more configurations to explore)
            logger.info("No trial available (e.g., max_trials reached or no more configurations to explore).")
            return {"trial_id": None, "hyperparameters": {}}

        logger.info(f"Returning trial {trial.trial_id} with HPs: {trial.hyperparameters.values}")
        return {
            "trial_id": trial.trial_id,
            "hyperparameters": trial.hyperparameters.values
        }

@app.post("/report_result")
def handle_report_result(data: TrialResult):
    """
    Endpoint for a tuner to report the full results (metrics) of a completed trial.
    This updates the Oracle's record for the specific trial and marks it as COMPLETED.
    """
    with lock: # Acquire lock for thread-safe access to the Oracle
        logger.info(f"Reporting result for trial {data.trial_id} with metrics: {data.result}")
        # KerasTuner's Oracle.update_trial method accepts metrics as a dictionary
        oracle.update_trial(
            trial_id=data.trial_id,
            metrics=data.result,
            status='COMPLETED' # Explicitly mark as completed when results are reported
        )
        return {"status": "ok", "message": f"Trial {data.trial_id} results reported and updated."}

@app.post("/update_status")
def handle_update_status(data: TrialStatusUpdate):
    """
    Endpoint for a tuner to update the status of an ongoing or pending trial.
    This allows monitoring the lifecycle of trials (e.g., FAILED, RUNNING, STOPPED).
    """
    with lock: # Acquire lock for thread-safe access to the Oracle
        logger.info(f"Updating status for trial {data.trial_id} to: {data.status}")
        oracle.update_trial(
            trial_id=data.trial_id,
            status=data.status
        )
        return {"status": "ok", "message": f"Trial {data.trial_id} status updated to {data.status}."}

@app.get("/list_trials")
def list_trials():
    """
    Endpoint to list all trials currently known and managed by the Oracle.
    This is used by clients (like the dashboard) to retrieve and display trial progress.
    """
    with lock: # Acquire lock for thread-safe access to the Oracle
        trials_list = []
        # Iterate through the Oracle's internal trials dictionary
        for trial_id, trial_obj in oracle.trials.items():
            trial_info = {
                "trial_id": trial_id,
                "hyperparameters": trial_obj.hyperparameters.values,
                "score": trial_obj.score if hasattr(trial_obj, 'score') else None, # Include score if available
                "status": trial_obj.status
            }
            trials_list.append(trial_info)
        logger.info(f"Returning {len(trials_list)} trials.")
        return {"trials": trials_list}

@app.get("/status")
def health_check():
    """
    Basic health check endpoint for the Oracle server.
    Provides current status, maximum trials configured, objective being optimized,
    and number of active (ongoing) trials.
    """
    logger.info("Health check received.")
    with lock: # Acquire lock for thread-safe access to Oracle properties
        return {"status": "Oracle Server is running.",
                "max_trials": oracle.max_trials,
                "objective": oracle.objective.name, # Access the name of the optimization objective
                "active_trials": len(oracle.ongoing_trials)}

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
        uvicorn.run(app, host=server_host, port=server_port)
        logger.info("✅ Oracle Server shut down gracefully.")
    except Exception as e:
        # Log any unexpected crashes and exit with an error code
        logger.critical(f"❌ Oracle Server crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1) # Exit with a non-zero status code to signal failure
