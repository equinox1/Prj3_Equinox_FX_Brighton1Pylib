import threading
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
            trial = self.oracle.create_trial(tuner_id="chief")
            return {
                "trial_id": trial.trial_id,
                "hyperparameters": trial.hyperparameters.values
            }

        @self.app.post("/report_result")
        def report_result(report: ResultReport):
            self.oracle.update_trial(trial_id=report.trial_id, score=report.result)
            return {"message": "Result received."}

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

   