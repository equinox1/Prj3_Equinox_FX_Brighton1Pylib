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

# Initialize CMqlSetup
clientlog_config = CMqlSetup()

# Set the global log directory and file path using CMqlSetup's method
# This is crucial to ensure the FileHandler has a valid, existing directory.
try:
    # Pass necessary parameters to set_log_dir for proper path construction
    final_logdir, final_logfile_path = clientlog_config.set_log_dir(
        logdir=app_params.get('LOGDIR'), # Use LOGDIR from app_params if available
        logfile=app_params.get('LOGFILE', 'tsneuropredict_app.log'), # Use LOGFILE from app_params or default
        servername=xerces_servername,
        backend=backend
    )
    logger_initial_print_message = f"Logging directory set to: {final_logdir}, file: {final_logfile_path}"
except Exception as e:
    # Fallback if set_log_dir fails early (use print as logger might not be fully configured yet)
    print(f"CRITICAL ERROR: Failed to set up log directories via CMqlSetup: {e}")
    sys.exit(1)

# Now, configure the actual logging using CMqlSetup's setup_logging method,
# passing the determined logfile path.
clientlog_config.setup_logging(logfile=final_logfile_path)

# Get the logger instance after setup_logging has been called.
# Using __name__ is best practice for module-specific loggers.
logger = logging.getLogger(__name__)
logger.info(logger_initial_print_message) # Log the path setup now that logger is ready

# Suppress ONNX Windows version warning, if applicable
warnings.filterwarnings("ignore", message="Unsupported Windows version")

# Extract necessary configuration for CustomOracle from tune_params
max_trials = tune_params.get('num_trials', int(os.environ.get("MLTUNE_TRIALS", 128)))
project_name = app_params.get('project_name', 'PyTorchTuningProject')
objective_name = tune_params.get('objective', 'val_loss')

# The Oracle's state directory will be a sub-directory within the final_logdir
# This should match how tsMqlMLCustomOracle.py expects its directory.
oracle_directory = Path(final_logdir) / "oracle_state" # Placing it directly under the final_logdir

# Ensure the Oracle's directory exists before initialization
try:
    oracle_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Oracle state directory ensured: {oracle_directory}")
except Exception as e:
    logger.critical(f"❌ Failed to create Oracle state directory {oracle_directory}: {e}")
    sys.exit(1) # Exit if we cannot create the necessary directory

# ----------------------------
# FastAPI Application and Oracle Instance
# ----------------------------
app = FastAPI()

# Initialize CustomOracle with all required parameters.
# Make sure the directory passed is a string, as KerasTuner expects it.
oracle = CustomOracle(
    objective=objective_name,
    max_trials=max_trials,
    directory=str(oracle_directory), # Pass as string
    project_name=project_name,
    seed=None, # Set a specific seed for reproducibility if needed (e.g., 42)
    reset_trials=True # Set to False if you want to resume previous tuning runs
)

# Initialize a threading.Lock for thread-safe access to the Oracle
lock = threading.Lock()

# ----------------------------
# Pydantic Request/Response Models
# ----------------------------
class TrialRequest(BaseModel):
    tuner_id: str

class TrialScore(BaseModel):
    trial_id: str
    score: float # For a simple score, e.g., validation loss

class TrialResult(BaseModel):
    trial_id: str
    # 'result' should be a dictionary of metrics, compatible with KerasTuner's update_trial
    result: Dict[str, float] # e.g., {'val_loss': 0.05, 'val_accuracy': 0.92}

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str # e.g., "RUNNING", "COMPLETED", "STOPPED", "INVALID"

# ----------------------------
# FastAPI Endpoints
# ----------------------------

@app.post("/request_trial")
def handle_request_trial(request: TrialRequest):
    """
    Endpoint for a tuner to request a new trial (hyperparameter configuration).
    """
    with lock:
        logger.info(f"Request for new trial from tuner_id: {request.tuner_id}")
        # KerasTuner's Oracle.create_trial handles the population of hyperparameters
        trial = oracle.create_trial(request.tuner_id)
        
        if trial is None:
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
    This aligns with the `report_trial_result` in `tsMqlMLOracleClient.py`.
    """
    with lock:
        logger.info(f"Reporting result for trial {data.trial_id} with metrics: {data.result}")
        # KerasTuner's Oracle.update_trial takes `metrics` as a dictionary
        oracle.update_trial(
            trial_id=data.trial_id,
            metrics=data.result,
            status='COMPLETED' # Mark as completed when results are reported
        )
        return {"status": "ok", "message": f"Trial {data.trial_id} results reported and updated."}

@app.post("/update_status")
def handle_update_status(data: TrialStatusUpdate):
    """
    Endpoint for a tuner to update the status of a trial (e.g., FAILED, RUNNING).
    This aligns with the `update_trial_status` in `tsMqlMLOracleClient.py`.
    """
    with lock:
        logger.info(f"Updating status for trial {data.trial_id} to: {data.status}")
        oracle.update_trial(
            trial_id=data.trial_id,
            status=data.status
        )
        return {"status": "ok", "message": f"Trial {data.trial_id} status updated to {data.status}."}

@app.get("/list_trials")
def list_trials():
    """
    Endpoint to list all trials known by the Oracle.
    Used by OracleClient.get_best_trial to retrieve all trials and find the best one.
    """
    with lock:
        trials_list = []
        for trial_id, trial_obj in oracle.trials.items():
            trial_info = {
                "trial_id": trial_id,
                "hyperparameters": trial_obj.hyperparameters.values,
                "score": trial_obj.score if hasattr(trial_obj, 'score') else None,
                "status": trial_obj.status
            }
            trials_list.append(trial_info)
        logger.info(f"Returning {len(trials_list)} trials.")
        return {"trials": trials_list}


@app.get("/status")
def health_check():
    """
    Basic health check endpoint for the server.
    """
    logger.info("Health check received.")
    return {"status": "Oracle Server is running.", "max_trials": oracle.max_trials, "active_trials": len(oracle.ongoing_trials)}

# ----------------------------
# Server Runner
# ----------------------------
if __name__ == "__main__":
    # The port value should come from your configuration.
    server_host = xerces_server
    server_port = xerces_port

    logger.info(f"🚀 Attempting to start Oracle Server at http://{server_host}:{server_port}")

    try:
        # This will block and run the Uvicorn server until it's manually stopped (e.g., Ctrl+C)
        uvicorn.run(app, host=server_host, port=server_port)
        logger.info("✅ Oracle Server shut down gracefully.")
    except Exception as e:
        logger.critical(f"❌ Oracle Server crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1) # Exit with an error code to signal failure
