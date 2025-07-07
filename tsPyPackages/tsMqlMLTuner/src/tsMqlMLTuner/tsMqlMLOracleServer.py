import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
import os # Import os to access environment variables
from typing import Dict, Optional # <--- ADDED: Import Dict and Optional from typing
import errno # Import errno for specific error codes

from tsMqlOverrides import CMqlOverrides

# Removed: from tsMqlLogService import CMqlLogService # This import is not needed here
# Removed: from tsMqlLogService import CMLogServiceSetup # This import is not needed here

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

# Corrected: Get the logger instance, assuming it's configured by the main application
logger = logging.getLogger(__name__)


# Pydantic models for request bodies
class TrialRequest(BaseModel):
    tuner_id: str

class TrialResult(BaseModel):
    trial_id: str
    score: float
    status: Optional[str] = "COMPLETED"

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str

class OracleServer:
    def __init__(self, oracle_instance, tuner_id="oracle_server"):
        self.app = FastAPI()
        self.oracle = oracle_instance
        self.tuner_id = tuner_id
        self._server_thread = None
        self._setup_routes()
        logger.info(f"[OracleServer] Initialized for tuner_id: {self.tuner_id}")

    def _setup_routes(self):
        @self.app.post("/request_trial")
        async def request_trial(request: TrialRequest):
            logger.info(f"[OracleServer] Request for new trial from tuner_id: {request.tuner_id}")
            trial = self.oracle.get_trial(request.tuner_id)
            if trial:
                logger.info(f"[OracleServer] Returning trial {trial['trial_id']} to tuner {request.tuner_id}")
                return JSONResponse(trial)
            else:
                logger.info(f"[OracleServer] No trial available (e.g., max_trials reached).")
                raise HTTPException(status_code=200, detail="No more trials available or active.")

        @self.app.post("/report_result")
        async def report_result(result: TrialResult):
            logger.info(f"[OracleServer] Received result for trial {result.trial_id}: score={result.score}, status={result.status}")
            try:
                self.oracle.report_trial_result(result.trial_id, result.score, result.status)
                return JSONResponse({"message": "Result reported successfully."})
            except Exception as e:
                logger.error(f"[OracleServer] Error reporting result for trial {result.trial_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/update_status")
        async def update_status(update: TrialStatusUpdate):
            logger.info(f"[OracleServer] Received status update for trial {update.trial_id}: status={update.status}")
            try:
                self.oracle.update_trial_status(update.trial_id, update.status)
                return JSONResponse({"message": "Status updated successfully."})
            except Exception as e:
                logger.error(f"[OracleServer] Error updating status for trial {update.trial_id}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/status")
        async def health_check():
            logger.info("[OracleServer] Health check received.")
            return JSONResponse({
                "status": "Oracle Server is running.",
                "max_trials": self.oracle.max_trials,
                "active_trials": len(self.oracle.ongoing_trials) if hasattr(self.oracle, 'ongoing_trials') else 0
            })

    def start(self, host="0.0.0.0", port=9000):
        if self._server_thread is not None and self._server_thread.is_alive():
            logger.info("Oracle Server already running.")
            return

        def run_server():
            try:
                # Use '127.0.0.1' for local testing if '0.0.0.0' causes issues
                uvicorn.run(self.app, host=host, port=port, log_level="info")
            except OSError as e:
                if e.errno == errno.EADDRINUSE: # Error code for "address already in use"
                    logger.critical(f"❌ Failed to start Oracle Server: Address {host}:{port} already in use. Please ensure no other instance is running.")
                else:
                    logger.critical(f"❌ Failed to start Oracle Server due to OS error: {e}", exc_info=True)
                # Exit the thread if binding fails
                os._exit(1) # Use os._exit to terminate the thread immediately
            except Exception as e:
                logger.critical(f"❌ Oracle Server crashed unexpectedly in run_server thread: {e}", exc_info=True)
                os._exit(1) # Ensure thread terminates on unexpected errors


        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"[OracleServer] Started server thread on {host}:{port}")

    def stop(self):
        logger.info("[OracleServer] Attempting to stop server (daemon thread will terminate with main process).")
        if self._server_thread and self._server_thread.is_alive():
            # In a real-world scenario, you'd need a more robust shutdown mechanism
            # for uvicorn, but for a daemon thread, it will exit with the main process.
            # If explicit shutdown is needed, uvicorn.Server.shutdown() would be used.
            pass
