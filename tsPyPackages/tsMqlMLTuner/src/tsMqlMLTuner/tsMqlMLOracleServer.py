import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging

# Setup logger
logger = logging.getLogger(__name__)

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
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        time.sleep(1)
        print(f"Oracle Server started at http://{host}:{port}")
