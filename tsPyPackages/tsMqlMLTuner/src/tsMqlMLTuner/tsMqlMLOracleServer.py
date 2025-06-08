import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging # Ensure logging is imported
import os # Import os to access environment variables

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    # Fallback or error if not set (should be set by multiworker_launcher.py)
    clientlog_config.setup_logging() # Use default if not provided via env
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for OracleServer. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
# -- end of logging setup ----

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

# Tuner and backend parameters from environment or defaults
tuner_model = tune_params.get('tuner_type', 'hyperband')
backend = tune_params.get('backend', 'tensorflow')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# Custom imports
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from keras_tuner.engine.trial import TrialStatus

# Pydantic models for request bodies
class TrialUpdate(BaseModel):
    trial_id: str
    status: str

class TrialResult(BaseModel):
    trial_id: str
    result: dict

class OracleServer:
    def __init__(self, oracle: CustomOracle, tuner_id: str):
        self.oracle = oracle
        self.tuner_id = tuner_id
        self.app = FastAPI(title="Oracle Server API", version="1.0.0")
        self._server_thread = None
        self._setup_routes()
        logger.info(f"[OracleServer] Initialized for tuner_id: {self.tuner_id}")
        logger.debug(f"[OracleServer] Backend: {backend}, Tuner Model: {tuner_model}")

    def _setup_routes(self):
        @self.app.get("/")
        async def root():
            logger.info("[OracleServer] Root endpoint accessed.")
            return {"message": "Oracle Server is running!"}

        @self.app.post("/get_trial")
        async def get_trial():
            logger.info("[OracleServer] Request to get a new trial.")
            try:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial:
                    logger.info(f"[OracleServer] Assigned new trial_id: {trial.trial_id} with status: {trial.status}")
                    return JSONResponse(content={
                        "trial_id": trial.trial_id,
                        "hyperparameters": trial.hyperparameters.values,
                        "status": trial.status
                    })
                logger.warning("[OracleServer] No new trials available. Max trials reached or no trials created.")
                return JSONResponse(content={"trial_id": None, "message": "No new trials available"}, status_code=404)
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[OracleServer] Exception in get_trial:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Failed to get trial: {str(e)}")

        @self.app.post("/report_result")
        async def report_result(update: TrialResult):
            logger.info(f"[OracleServer] Reporting result for trial_id: {update.trial_id}, result: {update.result}")
            try:
                if update.trial_id in self.oracle._trials:
                    trial = self.oracle._trials[update.trial_id]
                    # Assuming 'score' is the metric reported for the trial
                    score = update.result.get("score")
                    if score is not None:
                        trial.score = score
                        trial.status = TrialStatus.COMPLETED # Mark as completed
                        self.oracle.update_trial(trial.trial_id, score, trial.hyperparameters) # Update oracle's internal state
                        logger.info(f"[OracleServer] Trial {update.trial_id} completed with score: {score}")
                        return {"status": "success", "trial_id": update.trial_id, "score": score}
                    else:
                        logger.warning(f"[OracleServer] Result for trial {update.trial_id} is missing 'score'.")
                        return {"status": "failed", "trial_id": update.trial_id, "message": "Result missing 'score'"}
                logger.warning(f"[OracleServer] Trial {update.trial_id} not found for result reporting.")
                return {"status": "not_found", "trial_id": update.trial_id}
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[OracleServer] Exception in report_result for trial {update.trial_id}:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Report failed: {str(e)}")

        @self.app.post("/update_status")
        async def update_status(update: TrialUpdate):
            logger.info(f"[OracleServer] Updating status for trial_id: {update.trial_id} to {update.status}")
            try:
                if update.trial_id in self.oracle._trials:
                    self.oracle._trials[update.trial_id].status = update.status
                    logger.info(f"[OracleServer] Status updated for trial {update.trial_id}.")
                    return {"status": "updated", "trial_id": update.trial_id}
                logger.warning(f"[OracleServer] Trial {update.trial_id} not found for status update.")
                return {"status": "not_found", "trial_id": update.trial_id}
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"[OracleServer] Exception in update_status for trial {update.trial_id}:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

        @self.app.get("/list_trials")
        async def list_trials():
            logger.info("[OracleServer] Listing all trials.")
            trials = []
            for trial_id, trial in self.oracle._trials.items():
                trials.append({
                    "trial_id": trial_id,
                    "status": trial.status.name if hasattr(trial.status, 'name') else trial.status, # Handle enum or string status
                    "score": getattr(trial, "score", None),
                    "hyperparameters": trial.hyperparameters.values
                })
            logger.debug(f"[OracleServer] Retrieved {len(trials)} trials.")
            return {"trials": trials}

    def start(self, host="0.0.0.0", port=9000):
        if self._server_thread is not None:
            logger.warning("Oracle Server already running.")
            return

        def run_server():
            # Use reload=False in production to avoid issues with multiple instances
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"[OracleServer] Server thread started on {host}:{port}.")

    def stop(self):
        if self._server_thread is not None:
            # Uvicorn doesn't have a direct stop method for its server programmatically from outside its loop.
            # For a graceful shutdown in a real application, you'd send a signal or use a shared event.
            # For this example, stopping the daemon thread will exit with the main program.
            logger.info("Oracle Server stop requested. Daemon thread will exit with main process.")
            self._server_thread = None # Mark as stopped
        else:
            logger.info("Oracle Server is not running.")
