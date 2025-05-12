import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import traceback

# Get a logger for this module
logger = logging.getLogger(__name__)

class TrialRequest(BaseModel):
    trial_id: str
    hyperparameters: dict

class ResultReport(BaseModel):
    trial_id: str
    result: float

class StatusUpdate(BaseModel):
    trial_id: str
    status: str

class OracleServer:
    def __init__(self, oracle):
        self.oracle = oracle
        self.app = FastAPI()
        self._server_thread = None
        self._configure_routes()

    def _configure_routes(self):
        @self.app.get("/get_trial")
        def get_trial():
            try:
                tuner_id = self.tuner_id if hasattr(self, "tuner_id") else "chief"
                trial = self.oracle.create_trial(tuner_id)

                if trial is None:
                    return JSONResponse(status_code=200, content={"trial": None})
                return trial
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[OracleServer] 🔥 Exception in get_trial:\n{tb}")
                return JSONResponse(status_code=500, content={
                    "error": str(e),
                    "traceback": tb,
                })

        @self.app.post("/report_result")
        def report_result(report: ResultReport):
            try:
                self.oracle.score_trial(report.trial_id, report.result)
                return {"message": "Result received."}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error scoring trial: {e}")

        @self.app.post("/update_status")
        def update_status(update: StatusUpdate):
            trial = self.oracle.get_trial(update.trial_id)
            if trial:
                trial.status = update.status
                return {"message": f"Status updated to {update.status}"}
            raise HTTPException(status_code=404, detail="Trial not found")

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

        @self.app.get("/heartbeat")
        def heartbeat():
            return {"status": "alive", "timestamp": datetime.datetime.utcnow().isoformat()}

    def start(self, host="192.168.1.103", port=9000):
        if self._server_thread is not None:
            print("Oracle Server already running.")
            return

        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        time.sleep(1)
        print(f"Oracle Server started at http://{host}:{port}")