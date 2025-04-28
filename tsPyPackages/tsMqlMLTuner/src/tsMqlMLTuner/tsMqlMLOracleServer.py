# tsoracle_server.py

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

class OracleServer:
    def __init__(self, oracle):
        self.oracle = oracle
        self.app = FastAPI()
        self._configure_routes()
        self._server_thread = None

    def _configure_routes(self):
        @self.app.get("/trial")
        def get_trial():
            trial = self.oracle.create_trial()
            if trial:
                return {"trial_id": trial.trial_id, "hyperparameters": trial.hyperparameters.values}
            else:
                raise HTTPException(status_code=404, detail="No trial available")

        @self.app.post("/result")
        def report_result(report: ResultReport):
            trial_id = report.trial_id
            result = report.result
            self.oracle.update_trial(trial_id=trial_id, score=result)
            return {"message": "Result received"}

        @self.app.post("/upload_model")
        def upload_model(trial: TrialRequest):
            # Optional endpoint: expand this if you want workers to upload models
            return {"message": "Upload received (not implemented)"}

    def start(self, host="0.0.0.0", port=9000):
        if self._server_thread is not None:
            print("Oracle Server already running.")
            return

        def run_server():
            uvicorn.run(self.app, host=host, port=port, log_level="info")

        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        time.sleep(1)  # Give the server a second to start
        print(f"Oracle Server started at http://{host}:{port}")