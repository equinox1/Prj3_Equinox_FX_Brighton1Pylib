import os
import threading
import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlOverrides import CMqlOverrides
from tsMqlSetup import CMqlSetup
# ----------------------------
# Logger Setup
# ----------------------------
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile') # This should be the same as above.
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')


log_dir = global_logfile if global_logfile else global_logdir
log_file_path = os.path.join(log_dir, global_logdir)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("OracleServer")

# ----------------------------
# FastAPI and Oracle Setup
# ----------------------------
app = FastAPI()
oracle = CustomOracle()
lock = threading.Lock()

# ----------------------------
# Request/Response Models
# ----------------------------
class TrialRequest(BaseModel):
    tuner_id: str

class TrialScore(BaseModel):
    trial_id: str
    score: float

# ----------------------------
# Trial Request Endpoint
# ----------------------------
@app.post("/request_trial")
def handle_request_trial(request: TrialRequest):
    with lock:
        logger.info(f"Request for new trial from tuner_id: {request.tuner_id}")
        trial = oracle.create_trial(request.tuner_id)
        if trial is None:
            logger.info("No trial available.")
            return {"trial_id": None, "hyperparameters": {}}

        logger.info(f"Returning trial {trial.trial_id} with HPs: {trial.hyperparameters.values}")
        return {
            "trial_id": trial.trial_id,
            "hyperparameters": trial.hyperparameters.values
        }

# ----------------------------
# Score Submission Endpoint
# ----------------------------
@app.post("/score_trial")
def handle_score_trial(score_data: TrialScore):
    with lock:
        logger.info(f"Scoring trial {score_data.trial_id} with score: {score_data.score}")
        oracle.score_trial(score_data.trial_id, score_data.score)
        return {"status": "ok"}

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/status")
def health_check():
    logger.info("Health check received.")
    return {"status": "Oracle Server is running."}

# ----------------------------
# Server Runner
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get(xerces_port, 9000))
    logger.info(f"Starting Oracle Server on port {port}...")
    logger.info(f"Logging to file: {log_file_path}")
    uvicorn.run(app, host=xerces_server, port=xerces_port)
