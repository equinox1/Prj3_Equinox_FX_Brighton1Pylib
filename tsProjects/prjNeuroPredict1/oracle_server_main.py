# ✅ oracle_server_main.py

from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer 
from tsMqlMLTuner.tsMqlMLCustomOracle import CustomOracle
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides

import logging
from rich.logging import RichHandler
import time
import threading
import uvicorn
import logging.config

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
mql_overrides.env.override_params({"app": {'gtuner_model': chiefsetgtuner}})
gtuner_model = app_params.get('gtuner_model', 'tensorflow')  # or "tensorflow"
print(f"Chief Using GTuner model: {gtuner_model}")

xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile

print(f"Using Xerces servername: {xerces_servername}")
print(f"Using Xerces server: {xerces_server} on port {xerces_port}")
print(f"Using Xerces logfile: {tunerlogfile}")
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)
logger = setup_config.setup_global_logger(logfilein=global_logfile)

print(f"Global logdir: {global_logdir}")
print(f"Global logfile: {global_logfile}")

# Optional: remove existing handlers first to avoid duplicates
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Define the format
log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
date_format = "[%Y-%m-%d %H:%M:%S]"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=date_format,
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False, markup=True),  # Console
        logging.FileHandler(global_logfile, mode='a', encoding='utf-8')   # File
    ]
)

logger = logging.getLogger("tsTuner")  # or just use logging.getLogger()
# ----- End Logging Setup -----


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
