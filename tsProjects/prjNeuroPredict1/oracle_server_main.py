# ✅ oracle_server_main.py

from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle

from tsMqlOverrides import CMqlOverrides

import logging
from rich.logging import RichHandler
import time
import threading
import uvicorn
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


chiefsetgtuner= "pytorch"  # or "tensorflow"
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

# Set GTuner model based on app parameters or default to 'tensorflow'
mql_overrides.env.override_params({"app": {'gtuner_model': chiefsetgtuner}})
# Set backend for mql_overrides
mql_overrides.env.override_params({"mltune": {'backend': chiefsetgtuner}})

# -- start of logging setup --

gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(global_logfile)
# -- end of logging setup --

logger.info(f"Chief Using GTuner model: {gtuner_model}")
logger.info(f"Chief Using backend: {backend}")
print(f"Global logdir: {global_logdir}")
print(f"Global logfile: {global_logfile}")

def run_uvicorn(app, host, port):
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


def main():
    try:
        logger.info("🧠 Creating CustomOracle...")
        oracle = CustomOracle(objective="val_loss", max_trials=50,log=global_logdir, seed=42)

        logger.info("🚀 Starting OracleServer at http://192.168.1.103:9000")
        server = OracleServer(oracle, tuner_id="chief")

        thread = threading.Thread(target=run_uvicorn, args=(server.app, "192.168.1.103", 9000), daemon=True)
        thread.start()

        logger.info("✅ OracleServer is now running.")
        while True:
            time.sleep(60)

    except Exception as e:
        logger.error(f"❌ Failed to start OracleServer: {e}")
        logger.info("⚙️ Cleaning up resources...")
        logger.info("✅ Cleanup completed.")


if __name__ == "__main__":
    main()
