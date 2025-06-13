import os
import sys
import time
from pathlib import Path
import threading
import logging
import warnings
import inspect # Import inspect for debugging module path

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

# Create a unique project directory for the Oracle based on parameters
# This ensures that different runs/backends don't collide with Oracle's internal files
model_name = mql_overrides.env.ml_model_name()
# Project ID should ideally come from configuration or be dynamically generated for distinct runs
project_id = mql_overrides.env.ml_project_unique_id() # Use method from env manager
project_name = f"{model_name}_{project_id}" # Combined project name

# Determine the Oracle's directory based on the global model data path
oracle_base_dir = mql_overrides.env.ml_model_data_path()
oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

# DEBUG: Print the path of the CustomOracle module being loaded
logger.info(f"DEBUG: CustomOracle class loaded from: {inspect.getfile(CustomOracle)}")

# Instantiate CustomOracle
logger.info(f"Initializing CustomOracle with directory: {oracle_full_path}, project_name: {project_name}")
# Use a simple objective string as CustomOracle is designed for a single objective
oracle = CustomOracle(
    objective="val_loss",
    max_trials=tune_params.get('num_trials', 50),
    directory=str(oracle_full_path), # Pass as string
    project_name=project_name,
    seed=tune_params.get('seed', 42),
    overwrite=tune_params.get('overwrite', False) # Pass overwrite flag
)
logger.info("CustomOracle instance created.")

# Thread-safe lock for Oracle access
lock = threading.Lock()
logger.info("Thread lock for Oracle access initialized.")


# ----------------------------
# FastAPI Application
# ----------------------------
app = FastAPI(title="KerasTuner Oracle Server")

# Pydantic models for request bodies
class TrialRequest(BaseModel):
    tuner_id: str

class TrialScore(BaseModel):
    trial_id: str
    score: float

class TrialResult(BaseModel):
    trial_id: str
    metrics: Dict[str, float]

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str
    score: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


@app.post("/request_trial")
async def request_trial(req: TrialRequest):
    logger.info(f"Request for new trial from tuner_id: {req.tuner_id}")
    with lock: # Acquire lock for thread-safe trial creation
        # Create a new trial using the Oracle's logic
        # This calls CustomOracle's overridden `create_trial` method
        trial = oracle.create_trial(req.tuner_id)
        if trial:
            logger.info(f"Issued trial {trial.trial_id} to tuner {req.tuner_id}. Hyperparameters: {trial.hyperparameters.values}")
            return {"trial_id": trial.trial_id, "hyperparameters": trial.hyperparameters.values}
        else:
            logger.warning(f"No new trial could be created for tuner_id: {req.tuner_id}. Max trials reached or no more combinations.")
            return {"trial_id": None, "hyperparameters": {}}

@app.post("/report_result")
async def report_result(req: TrialResult):
    logger.info(f"Reporting result for trial {req.trial_id}: {req.metrics}")
    with lock: # Acquire lock for thread-safe trial update
        # Update the trial with its final score
        # This calls CustomOracle's overridden `update_trial` method
        score_value = req.metrics.get(oracle.objective.name, None) # Use objective name for score key
        if score_value is not None:
            oracle.update_trial(
                trial_id=req.trial_id,
                status="COMPLETED", # Mark as completed upon receiving result
                score=score_value,
                metrics=req.metrics
            )
            logger.info(f"Trial {req.trial_id} completed with score: {score_value}")
        else:
            logger.warning(f"Trial {req.trial_id} reported result but missing objective metric '{oracle.objective.name}'. Metrics: {req.metrics}")
            oracle.update_trial(
                trial_id=req.trial_id,
                status="FAILED", # Mark as failed if objective score is missing
                metrics=req.metrics
            )
    return {"message": "Result reported."}


@app.post("/update_status")
async def update_status(req: TrialStatusUpdate):
    logger.info(f"Updating status for trial {req.trial_id} to {req.status}")
    with lock: # Acquire lock for thread-safe status update
        # This calls CustomOracle's overridden `update_trial` method
        oracle.update_trial(
            trial_id=req.trial_id,
            status=req.status,
            score=req.score,
            metrics=req.metrics
        )
    return {"message": f"Status for trial {req.trial_id} updated to {req.status}."}

@app.get("/list_trials")
async def list_trials():
    logger.info("Request to list trials.")
    with lock: # Acquire lock for thread-safe access to trials
        # Access the trials dictionary directly from the Oracle instance
        trials_list = [{"trial_id": t.trial_id, "status": t.status, "hyperparameters": t.hyperparameters.values, "score": t.score, "metrics": t.metrics} for t in oracle.trials.values()]
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
        objective_name = "N/A"
        # Safely try to get the objective name, accounting for KerasTuner's structure
        if hasattr(oracle, 'objective') and oracle.objective:
            if isinstance(oracle.objective, list) and len(oracle.objective) > 0:
                # KerasTuner's Oracle.objective is typically a list of Objective objects
                objective_name = oracle.objective[0].name
            elif hasattr(oracle.objective, 'name'): # Fallback if it's a single Objective object directly
                objective_name = oracle.objective.name
        
        return {"status": "Oracle Server is running.",
                "max_trials": oracle.max_trials,
                "objective": objective_name, # Use the safely retrieved objective name
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
