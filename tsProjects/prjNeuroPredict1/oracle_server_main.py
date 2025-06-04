# ✅ oracle_server_main.py

import os
import time
from pathlib import Path
import threading
import logging
import warnings
from rich.logging import RichHandler
import uvicorn
from datetime import datetime, date

from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Dict, Optional
import time

from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlOverrides import CMqlOverrides
from tsMqlSetup import CMqlSetup

# -- Suppress ONNX Windows version warning --
warnings.filterwarnings("ignore", message="Unsupported Windows version")

# -- Load environment variables first --
env_backend = os.environ.get("MLTUNE_BACKEND", "tensorflow")
env_gtuner = os.environ.get("GTUNER_MODEL", env_backend)
env_trials = int(os.environ.get("MLTUNE_TRIALS", 128))
env_tuner_type = os.environ.get("TUNER_TYPE", "hyperband")

# -- Apply overrides before config extraction --
mql_overrides = CMqlOverrides()
mql_overrides.env.override_params({
    "mltune": {
        "backend": env_backend,
        "num_trials": env_trials,
        "tuner_type": env_tuner_type,
        "reset_trials": True,        # 👈 ensures all previous trials are cleared
        "overwrite": True,           # 👈 allows tuner to recreate directory/files
        "tuner_id": "chief"          # 👈 ensures a clean session per run
    },
    "app": {
        "gtuner_model": env_gtuner
    }
})

print(f"Num trials: {env_trials}")
num_trials = mql_overrides.env.all_params().get("mltune", {}).get("num_trials", 50)


# -- Extract config after overrides are in place --
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})

gtuner_model = app_params.get('gtuner_model', env_backend)
backend = tune_params.get("backend", env_backend)
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = int(app_params.get('xerces_port', 9000))
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# -- Setup logging --
setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=8,
    num_threads=1
)

global_logdir, global_logfile = setup_config.set_log_dir(
    logdir=None,
    logfile=xerces_logfile,
    servername=xerces_servername,
    ltuner=gtuner_model
)
logger = setup_config.setup_global_logger(global_logfile, force_reset=True)



# -- Logging headers --
logger.info(f"Chief Using GTuner model: {gtuner_model}")
logger.info(f"Chief Using backend: {backend}")
print(f"Global logdir: {global_logdir}")
print(f"Global logfile: {global_logfile}")

# -- Uvicorn runner --
def run_uvicorn(app, host, port):
    try:
        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s",
                },
            },
            "handlers": {
                "file": {
                    "class": "logging.FileHandler",
                    "filename": global_logfile,
                    "formatter": "default",
                    "level": "DEBUG",
                },
            },
            "root": {
                "handlers": ["file"],
                "level": "DEBUG",
            },
        }
        uvicorn.run(app, host=host, port=port, log_level="debug", log_config=log_config)
    except Exception as e:
        logger.exception(f"❌ Uvicorn failed to start: {e}")

def clean_stale_trials_from_oracle(logdir):
    import json
    from pathlib import Path

    oracle_file = Path(logdir) / "oracle.json"
    if not oracle_file.exists():
        logger.warning(f"No oracle.json found at {oracle_file}")
        return

    try:
        with open(oracle_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        trials = data.get("trials", {})
        updated = False

        for trial_id, trial_data in trials.items():
            if trial_data.get("status") == "RUNNING":
                logger.info(f"Marking stale trial {trial_id} as FAILED")
                trial_data["status"] = "FAILED"
                updated = True

        if updated:
            with open(oracle_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Stale trials have been marked as FAILED.")
        else:
            logger.info("No stale trials found to clean.")
    except Exception as e:
        logger.exception(f"Failed to clean stale trials: {e}")

# -- Main launcher --
def main():
    # -- Clean up stale trials from previous runs --
    if tune_params.get("reset_trials", True):
        oracle_file = Path(global_logdir) / "oracle.json"
        if oracle_file.exists():
            logger.info(f"🗑️ Deleting existing oracle.json at {oracle_file} for a fresh start.")
            try:
                os.remove(oracle_file)
            except Exception as e:
                logger.error(f"❌ Failed to delete oracle.json: {e}")
    else:
        clean_stale_trials_from_oracle(global_logdir)

    try:
        logger.info("🧠 Creating CustomOracle...")
        num_trials = tune_params.get("num_trials", 50)
       # oracle = CustomOracle(objective="val_loss", max_trials=num_trials, log=global_logdir, seed=42, reset_trials=True)

        # Example variables (adjust as per your actual context)
        tuner_id = f"tuner_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        oracle_dir = Path(global_logdir)
        logger.info(f"Oracle directory created at: {oracle_dir}")
        # Create the Oracle instance
        logger.info(f"Creating CustomOracle with max_trials={num_trials}, directory={oracle_dir}, project_name={tuner_id}")

        oracle = CustomOracle(
            objective="val_loss",
            max_trials=num_trials,
            directory=str(oracle_dir),
            project_name=tuner_id,
            reset_trials=True,
            seed=42,
        )

        # 👇 Prepopulate trials
        logger.info(f"🧪 Pre-populating {oracle.max_trials} trials in Oracle...")
        for _ in range(oracle.max_trials):
            oracle.create_trial("chief")
        oracle.save()
        logger.info(f"✅ Trial population complete. Oracle now has {len(oracle.trials)} trials.")

        logger.info(f"🚀 Starting OracleServer at http://{xerces_server}:{xerces_port}")
        server = OracleServer(oracle, tuner_id="chief",)

        thread = threading.Thread(target=run_uvicorn, args=(server.app, xerces_server, xerces_port))
        thread.start()

        time.sleep(1)
        if not thread.is_alive():
            logger.error("❌ Uvicorn server thread died immediately after starting.")
            raise RuntimeError("Uvicorn failed to start. Check configuration or port.")

        logger.info("[OK] OracleServer is now running.")
        while True:
            time.sleep(60)

    except Exception as e:
        logger.error(f"❌ Failed to start OracleServer: {e}")
        logger.info("⚙️ Cleaning up resources...")
        logger.info("[OK] Cleanup completed.")


if __name__ == "__main__":
    main()
