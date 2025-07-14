# Rewritten: patched_oracle_server_main.py
import os
import sys
import time
from pathlib import Path
import logging
import warnings
import inspect
import socket
from fastapi import FastAPI

from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlMLTuner.tsMqlMLOracleServer import app as oracle_app, start_oracle_server, set_oracle_instance
from tsMqlMLTuner.tsMqlMLOracleServer import app

import uvicorn
from tsMqlOverrides import CMqlOverrides

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Network config
xerces_server = app_params.get('xerces_server', "127.0.0.1")
xerces_port = app_params.get('xerces_port', 9000)

# Suppress deprecated warnings
warnings.filterwarnings("ignore", message="The `tune_new_entries` and `allow_new_entries` arguments are deprecated.")

# Directory setup
import random
oracle_base_dir = Path(base_params.get('mp_glob_base_log_path')) / "oracle_server_data"
oracle_base_dir.mkdir(parents=True, exist_ok=True)

model_name = os.environ.get('ML_MODEL_NAME', tune_params.get('ml_model_name', 'default_model'))
project_id = os.environ.get('ML_PROJECT_ID', base_params.get('mp_glob_sub_ml_baseuniq', random.randrange(1, 1024)))
project_name = f"{model_name}_{project_id}"

oracle_full_path = oracle_base_dir / project_name
oracle_full_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Oracle data directory: {oracle_full_path}")

# Check port availability
if not port_available(xerces_port, xerces_server):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False

if not port_available(xerces_port, "127.0.0.1"):
    logger.critical(f"❌ Port {xerces_port} already in use on host 127.0.0.1. Exiting.")
    sys.exit(1)

# Initialize Oracle
logger.debug(f"CustomOracle class loaded from: {inspect.getfile(CustomOracle)}")

oracle_instance = CustomOracle(
    objective=tune_params.get('objective', "val_loss"),
    max_trials=tune_params.get('num_trials', 50),
    directory=str(oracle_base_dir),
    project_name=project_name,
    seed=tune_params.get('seed', 42),
    overwrite=tune_params.get('overwrite', False)
)
logger.info("CustomOracle instance created.")

set_oracle_instance(oracle_instance)
app = oracle_app

@app.get("/status")
async def health_check():
    logger.info("🔍 /status endpoint called - health check passed.")
    return {
        "status": "Oracle Server is running.",
        "max_trials": getattr(oracle_instance, 'max_trials', None),
        "active_trials": len(getattr(oracle_instance, 'ongoing_trials', []))
    }

if __name__ == "__main__":
    import asyncio
    import platform

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logger.info(f"✅ Attempting to start Oracle Server at http://{xerces_server}:{xerces_port}")

    try:
        uvicorn.run(app, host="127.0.0.1", port=xerces_port, log_level="info")
    except KeyboardInterrupt:
        logger.info("👋 Oracle Server received KeyboardInterrupt. Shutting down gracefully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Exception occurred while starting Oracle Server: {e}", exc_info=True)
        sys.exit(1)
