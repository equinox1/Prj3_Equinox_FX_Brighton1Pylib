import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
# -- Set up global logging --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()
clientlog_config.setup_logging()  # Ensure logging is configured before getting the logger
logger = logging.getLogger(__name__)
# -- end of logging setup ----


from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile') # This should be the same as above.

class TrialRequest(BaseModel):
    tuner_id: str

class ReportResultRequest(BaseModel):
    trial_id: str
    result: dict

class UpdateStatusRequest(BaseModel):
    trial_id: str
    status: str

class OracleServer:
    def __init__(self, oracle, tuner_id="chief"):
        self.oracle = oracle
        self.tuner_id = tuner_id
        self.app = FastAPI()
        self._server_thread = None
        self._setup_routes()
        logger.info(f"[OracleServer] Initialized for tuner_id: {self.tuner_id}")
        logger.debug(f"[OracleServer] Backend: {backend}, Tuner Model: {gtuner_model}")


    def _setup_routes(self):
        @self.app.get("/heartbeat")
        async def heartbeat():
            logger.info("[OracleServer] Heartbeat received.")
            return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}

        @self.app.post("/request_trial")
        async def request_trial(request: TrialRequest):
            logger.info(f"[OracleServer] Request for new trial from tuner_id: {request.tuner_id}")
            trial = self.oracle.create_trial(tuner_id=request.tuner_id)
            if trial:
                hyperparameters = {hp.name: hp.current_value for hp in trial.hyperparameters.space}
                logger.info(f"[OracleServer] Returning trial {trial.trial_id} with HPs: {hyperparameters}")
                return {
                    "trial_id": trial.trial_id,
                    "hyperparameters": hyperparameters,
                    "status": trial.status,
                }
            logger.warning(f"[OracleServer] No new trial available for tuner_id: {request.tuner_id}")
            raise HTTPException(status_code=404, detail="No more trials available.")

        @self.app.post("/report_result")
        async def report_result(update: ReportResultRequest):
            logger.info(f"[OracleServer] Reporting result for trial {update.trial_id}: {update.result}")
            try:
                # Assuming result contains 'score' and other metrics
                score = update.result.get('score')
                status = update.result.get('status', 'COMPLETED') # Default to COMPLETED
                self.oracle.update_trial(
                    trial_id=update.trial_id,
                    status=status,
                    score=score,
                    hyperparameters=update.result.get('hyperparameters', None) # Pass HPs if available
                )
                return {"status": "received", "trial_id": update.trial_id}
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[OracleServer] Exception in report_result:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Result reporting failed: {str(e)}")

        @self.app.post("/update_status")
        async def update_status(update: UpdateStatusRequest):
            logger.info(f"[OracleServer] Updating status for trial {update.trial_id} to {update.status}")
            try:
                # Use the update_trial method with only status
                self.oracle.update_trial(trial_id=update.trial_id, status=update.status)
                return {"status": "updated", "trial_id": update.trial_id}
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[OracleServer] Exception in update_status:\n{tb}")
                raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")

        @self.app.get("/list_trials")
        async def list_trials():
            logger.info("[OracleServer] Listing all trials.")
            trials_list = []
            # CORRECTED: Access self.oracle.trials directly, which is a dictionary managed by KerasTuner Oracle
            for trial_id, trial in self.oracle.trials.items():
                trials_list.append({
                    "trial_id": trial_id,
                    "status": trial.status,
                    "score": getattr(trial, "score", None),
                    "hyperparameters": getattr(trial, "hyperparameters", {}).values
                })
            return {"trials": trials_list}

    def start(self, host="0.0.0.0", port=9000):
        if self._server_thread is not None and self._server_thread.is_alive():
            print("Oracle Server already running.")
            return

        def run_server():
            # Use '127.0.0.1' for local testing if '0.0.0.0' causes issues,
            # but '0.0.0.0' is generally preferred for broader network access.
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info(f"[OracleServer] Started server thread on {host}:{port}")

    def stop(self):
        # Stopping Uvicorn gracefully is not straightforward when run in a separate thread.
        # For simple cases, marking the thread as daemon and allowing the main program to exit works.
        # For more complex shutdowns, Uvicorn's Server class would be needed.
        # Here, we just log a message and let the daemon thread terminate with the main process.
        logger.info("[OracleServer] Attempting to stop server (daemon thread will terminate with main process).")
        if self._server_thread and self._server_thread.is_alive():
            # In a real-world scenario, you'd need a more robust shutdown mechanism
            # for the uvicorn server, e.g., by using `uvicorn.Server` directly.
            pass # Daemon thread will exit on main program termination.

