# filename: tsMqlMLOracleServer.py
import threading
import time
import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback
import logging
import os
from typing import Dict, Optional
import errno
from pathlib import Path # Import Path

from tsMqlOverrides import CMqlOverrides
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle # Import CustomOracle

# Global oracle instance injected from oracle_server_main
oracle_instance: Optional[CustomOracle] = None

def set_oracle_instance(instance: CustomOracle):
    global oracle_instance
    oracle_instance = instance

mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

logger = logging.getLogger("OracleServer")

class TrialRequest(BaseModel):
    tuner_id: str

class TrialResult(BaseModel):
    trial_id: str
    score: float
    status: Optional[str] = "COMPLETED"

class TrialStatusUpdate(BaseModel):
    trial_id: str
    status: str

app = FastAPI()

@app.post("/get_trial")
async def get_trial(request: Request):
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    try:
        req_body = await request.json()
        tuner_id = req_body.get("tuner_id")
        if not tuner_id:
            raise HTTPException(status_code=400, detail="tuner_id is required.")
        trial = oracle_instance.get_trial(tuner_id)
        if trial:
            return JSONResponse(content=trial)
        else:
            return JSONResponse(content=None) # Indicate no more trials
    except Exception as e:
        logger.error(f"[OracleServer] Error in /get_trial: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report_result")
async def report_result(result: TrialResult):
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    try:
        oracle_instance.report_trial_result(result.trial_id, result.score, result.status)
        return {"message": "Trial result reported successfully."}
    except Exception as e:
        logger.error(f"[OracleServer] Error in /report_result: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_status")
async def update_status(update: TrialStatusUpdate):
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    try:
        oracle_instance.update_trial_status(update.trial_id, update.status)
        return {"message": "Trial status updated successfully."}
    except Exception as e:
        logger.error(f"[OracleServer] Error in /update_status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_best_trial")
async def get_best_trial():
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    try:
        best_trial = oracle_instance.get_best_trial()
        if best_trial:
            return JSONResponse(content=best_trial)
        else:
            return JSONResponse(content=None)
    except Exception as e:
        logger.error(f"[OracleServer] Error in /get_best_trial: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def run_uvicorn_server(host: str, port: int):
    """Runs the Uvicorn server in a separate thread."""
    try:
        # We no longer need to pass `app` here, it's global and imported.
        uvicorn.run(app, host=host, port=port, log_level="info")
    except Exception as e:
        logger.critical(f"[OracleServer] Uvicorn server failed to start: {e}", exc_info=True)
        # Attempt to create a dummy file to indicate failure to the parent process
        failure_indicator_path = Path(os.getcwd()) / "oracle_server_startup_failed.txt"
        with open(failure_indicator_path, "w") as f:
            f.write(f"Server failed to start at {datetime.datetime.now()}: {e}")
    finally:
        logger.info(f"[OracleServer] Server thread for {host}:{port} terminated.")

def start_oracle_server(host: str, port: int):
    """Starts the Oracle server in a new thread."""
    # This function is now simplified as the Uvicorn server runs the FastAPI app directly.
    thread = threading.Thread(target=run_uvicorn_server, args=(host, port))
    thread.daemon = True  # Allow the main program to exit even if the thread is still running
    thread.start()
    logger.info(f"[OracleServer] Started background server on {host}:{port}")
    time.sleep(3)


@app.post("/reset_trials")
async def reset_trials():
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    oracle_instance.trials.clear()
    oracle_instance.ongoing_trials.clear()
    oracle_instance.trials_created = 0
    oracle_instance.trials_completed = 0
    for f in os.listdir(oracle_instance.directory):
        if f.endswith(".json"):
            try:
                os.remove(os.path.join(oracle_instance.directory, f))
            except Exception as e:
                logger.warning(f"[OracleServer] Failed to delete {f}: {e}")
    logger.info("[OracleServer] Trials reset.")
    return {"message": "All trials have been reset."}

@app.get("/export_trials")
async def export_trials():
    if oracle_instance is None:
        raise HTTPException(status_code=500, detail="Oracle instance not initialized.")
    try:
        export_path = oracle_instance.directory / "trials_export.json"
        with export_path.open("w", encoding="utf-8") as f:
            import json
            json.dump(list(oracle_instance.trials.values()), f, indent=2)
        return {"message": "Trials exported.", "path": str(export_path)}
    except Exception as e:
        logger.error(f"[OracleServer] Error in /export_trials: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def health_check():
    if oracle_instance is None:
        return {"status": "Oracle Server is running, but Oracle instance is not initialized.",
                "max_trials": None,
                "active_trials": None,
                "completed_trials": None}
    logger.info("🔍 /status endpoint called - health check passed.")
    return {
        "status": "Oracle Server is running.",
        "max_trials": getattr(oracle_instance, 'max_trials', None),
        "trials_created": getattr(oracle_instance, 'trials_created', None),
        "trials_completed": getattr(oracle_instance, 'trials_completed', None),
        "active_trials": len(getattr(oracle_instance, 'ongoing_trials', []))
    }
