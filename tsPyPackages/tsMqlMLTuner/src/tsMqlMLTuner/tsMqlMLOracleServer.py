import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
<<<<<<< HEAD
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
=======

# -- start of logging setup --
from tsMqlSetup import CMqlSetup
# ✅ Logger and Logdir Setup
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
l
logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
# -- end of logging setup ----

class TrialRequest(BaseModel):
    trial_id: str
    hyperparameters: dict

class ResultReport(BaseModel):
    trial_id: str
    result: float

class UpdateStatusModel(BaseModel):
    trial_id: str
    status: str

class OracleServer:
    def __init__(self, oracle, tuner_id="chief"):
        self.oracle = oracle
        self.tuner_id = tuner_id
        self.app = FastAPI()
        self._server_thread = None
        self._configure_routes()

    def _configure_routes(self):
        @self.app.get("/heartbeat")
        def heartbeat():
            return {"status": "alive", "timestamp": datetime.datetime.utcnow().isoformat()}

        @self.app.get("/get_trial")
        def get_trial():
            try:
                trial = self.oracle.create_trial(self.tuner_id)
                if trial is None:
                    return JSONResponse(status_code=200, content={"trial": None})

                hp_values = getattr(trial.hyperparameters, 'values', {})
                trial_dict = {
                    "trial_id": trial.trial_id,
                    "hyperparameters": hp_values,
                    "status": trial.status,
                    "score": getattr(trial, "score", None),
                }
                return JSONResponse(status_code=200, content=trial_dict)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[OracleServer] 🔥 Exception in get_trial:\n{tb}")
                return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})

        @self.app.post("/report_result")
        def report_result(report: ResultReport):
            try:
                self.oracle.score_trial(report.trial_id, report.result)
                return {"message": "Result received."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error scoring trial: {e}")

        @self.app.post("/update_status")
        def update_status(update: UpdateStatusModel):
            try:
                if update.trial_id in self.oracle._trials:
                    self.oracle._trials[update.trial_id].status = update.status
                    return {"status": "updated", "trial_id": update.trial_id}
                return {"status": "not_found", "trial_id": update.trial_id}
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[OracleServer] 🔥 Exception in update_status:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

        @self.app.get("/list_trials")
        def list_trials():
            trials = []
            for trial_id, trial in self.oracle._trials.items():
                trials.append({
                    "trial_id": trial_id,
                    "status": trial.status,
                    "score": getattr(trial, "score", None),
                    "hyperparameters": trial.hyperparameters.values
                })
            return {"trials": trials}

    def start(self, host="0.0.0.0", port=9000):
        if self._server_thread is not None:
            print("Oracle Server already running.")
            return

        def run_server():
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
<<<<<<< HEAD
        logger.info(f"[OracleServer] Started server thread on {host}:{port}")

    def stop(self):
        logger.info("[OracleServer] Attempting to stop server (daemon thread will terminate with main process).")
        if self._server_thread and self._server_thread.is_alive():
            # In a real-world scenario, you'd need a more robust shutdown mechanism
            # for the uvicorn server, possibly involving uvicorn.Server and its stop() method.
            # For a simple daemon thread, exiting the main process is often sufficient.
            pass
=======
        time.sleep(1)
        print(f"Oracle Server started at http://{host}:{port}")
>>>>>>> 57ddb757d2636855e085392350ea7a26f8ad05f2
