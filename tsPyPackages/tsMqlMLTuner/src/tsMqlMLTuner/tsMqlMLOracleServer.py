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

# -- Set up global logging (from tsMqlSetup) --
# This ensures consistency across all modules.
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# --- Logging setup ---
# This script now *only* gets a logger. The root logger is configured by multiworker_launcher.py.
# This prevents repeated "Logging initialized" messages and ensures a consistent log file.
logger = logging.getLogger(__name__)
# -- end of logging setup ----




# Pydantic models for request bodies
class TrialRequest(BaseModel):
    tuner_id: str

class TrialScore(BaseModel):
    trial_id: str
    score: float

class TrialResult(BaseModel):
    trial_id: str
    result: Dict[str, float]

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str


class OracleServer:
    def __init__(self, oracle_instance, tuner_id=None):
        self.app = FastAPI()
        self.oracle = oracle_instance
        self.lock = threading.Lock()
        self.tuner_id = tuner_id
        self._server_thread = None # For managing the Uvicorn thread
        logger.info(f"[OracleServer] Initialized with tuner_id: {self.tuner_id}")

        self._setup_routes()

    def _setup_routes(self):
        @self.app.post("/request_trial")
        async def request_trial(request: TrialRequest):
            with self.lock:
                logger.info(f"[OracleServer] Request for new trial from tuner_id: {request.tuner_id}")
                try:
                    trial = self.oracle.create_trial(request.tuner_id)
                    if trial is None:
                        logger.info("[OracleServer] No trial available (e.g., max_trials reached).")
                        return JSONResponse({"trial_id": None, "hyperparameters": {}}, status_code=200)
                    
                    # Ensure hyperparameters.values is a dictionary
                    hps_values = trial.hyperparameters.values if trial.hyperparameters else {}
                    logger.info(f"[OracleServer] Returning trial {trial.trial_id} with HPs: {hps_values}")
                    return JSONResponse({
                        "trial_id": trial.trial_id,
                        "hyperparameters": hps_values
                    })
                except Exception as e:
                    logger.error(f"[OracleServer] Error requesting trial: {e}", exc_info=True)
                    return JSONResponse({"detail": str(e)}, status_code=500)

        @self.app.post("/report_result")
        async def report_result(result_data: TrialResult):
            with self.lock:
                logger.info(f"[OracleServer] Reporting result for trial {result_data.trial_id} with result: {result_data.result}")
                try:
                    # KerasTuner's Oracle.update_trial takes 'metrics' as a dictionary
                    self.oracle.update_trial(
                        trial_id=result_data.trial_id,
                        metrics=result_data.result,
                        status='COMPLETED' # Mark as completed when results are reported
                    )
                    self.oracle.save() # Persist state after updating a trial
                    return JSONResponse({"status": "ok", "message": f"Trial {result_data.trial_id} results reported and updated."})
                except Exception as e:
                    logger.error(f"[OracleServer] Error reporting result for trial {result_data.trial_id}: {e}", exc_info=True)
                    return JSONResponse({"detail": str(e)}, status_code=500)

        @self.app.post("/update_status")
        async def update_status(status_data: TrialStatusUpdate):
            with self.lock:
                logger.info(f"[OracleServer] Updating status for trial {status_data.trial_id} to: {status_data.status}")
                try:
                    self.oracle.update_trial(
                        trial_id=status_data.trial_id,
                        status=status_data.status
                    )
                    self.oracle.save() # Persist state after status update
                    return JSONResponse({"status": "ok", "message": f"Trial {status_data.trial_id} status updated to {status_data.status}."})
                except Exception as e:
                    logger.error(f"[OracleServer] Error updating status for trial {status_data.trial_id}: {e}", exc_info=True)
                    return JSONResponse({"detail": str(e)}, status_code=500)

        @self.app.get("/list_trials")
        async def list_trials():
            with self.lock:
                trials_list = []
                # Ensure self.oracle.trials is accessed safely and is populated
                if hasattr(self.oracle, 'trials') and self.oracle.trials:
                    for trial_id, trial_obj in self.oracle.trials.items():
                        trial_info = {
                            "trial_id": trial_id,
                            "hyperparameters": trial_obj.hyperparameters.values if trial_obj.hyperparameters else {},
                            "score": trial_obj.score if hasattr(trial_obj, 'score') else None,
                            "status": trial_obj.status
                        }
                        trials_list.append(trial_info)
                logger.info(f"[OracleServer] Returning {len(trials_list)} trials.")
                return JSONResponse({"trials": trials_list})


        @self.app.get("/get_best_trial")
        async def get_best_trial():
            with self.lock:
                logger.info("[OracleServer] Request to get best trial.")
                try:
                    # You might need to refine this based on how KerasTuner's Oracle stores best trials
                    # For a simple approach, sort by objective score
                    trials_with_scores = [t for t in self.oracle.trials.values() if t.status == 'COMPLETED' and hasattr(t, 'score') and t.score is not None]
                    if not trials_with_scores:
                        logger.info("[OracleServer] No completed trials with scores found for best trial determination.")
                        return JSONResponse({"best_trial": None})

                    # Assuming 'val_loss' is objective and lower is better
                    best_trial_obj = min(trials_with_scores, key=lambda t: t.score)
                    
                    best_trial_info = {
                        "trial_id": best_trial_obj.trial_id,
                        "hyperparameters": best_trial_obj.hyperparameters.values,
                        "score": best_trial_obj.score,
                        "status": best_trial_obj.status
                    }
                    logger.info(f"[OracleServer] Best trial found: {best_trial_info['trial_id']} with score: {best_trial_info['score']}")
                    return JSONResponse({"best_trial": best_trial_info})
                except Exception as e:
                    logger.error(f"[OracleServer] Error getting best trial: {e}", exc_info=True)
                    return JSONResponse({"detail": str(e)}, status_code=500)

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
            # Use '127.0.0.1' for local testing if '0.0.0.0' causes issues
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"[OracleServer] Started server thread on {host}:{port}")

    def stop(self):
        logger.info("[OracleServer] Attempting to stop server (daemon thread will terminate with main process).")
        if self._server_thread and self._server_thread.is_alive():
            # In a real-world scenario, you'd need a more robust shutdown mechanism
            # for the uvicorn server, possibly involving uvicorn.Server and its stop() method.
            # For a simple daemon thread, exiting the main process is often sufficient.
            pass
