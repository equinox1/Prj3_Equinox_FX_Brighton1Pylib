import threading
import time
import requests
import logging

# -- start of logging setup --
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
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})
from tsMqlSetup import CMqlSetup
gtuner_model = app_params.get('gtuner_model', 'pytorch')  # or "tensorflow"
backend = tune_params.get('backend', gtuner_model)  # or "tensorflow"
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')
tunerlogfile = xerces_logfile
global_logdir, global_logfile = setup_config.set_log_dir(logdir=None, logfile=tunerlogfile, servername=xerces_servername,ltuner=gtuner_model)

logger = setup_config.setup_global_logger(global_logfile, force_reset=True)
# -- end of logging setup ----

class OracleClient:
    def __init__(self, host="localhost", port=9000, heartbeat_interval=300):
        self.url = f"http://{host}:{port}"
        self.heartbeat_interval = heartbeat_interval
        self.start_heartbeat_loop()

    def start_heartbeat_loop(self):
        def heartbeat():
            while True:
                try:
                    response = requests.get(f"{self.url}/heartbeat", timeout=5)
                    logging.info(f"[Heartbeat] Oracle server says: {response.json()}")
                except Exception as e:
                    logging.warning(f"[Heartbeat] Oracle server unreachable: {e}")
                time.sleep(self.heartbeat_interval)
        threading.Thread(target=heartbeat, daemon=True).start()

    def get_trial(self):
        try:
            response = requests.get(f"{self.url}/get_trial", timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logging.error("Timed out while requesting trial from OracleServer")
        except Exception as e:
            logging.exception("Unexpected error in OracleClient.get_trial")

    def report_trial_result(self, trial_id, result):
        try:
            payload = {"trial_id": trial_id, "result": result}
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=30)
            response.raise_for_status()
        except requests.Timeout:
            logging.error("Timed out while reporting result to OracleServer")
        except Exception as e:
            logging.exception("Unexpected error in OracleClient.report_trial_result")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "status": status}
            response = requests.post(f"{self.url}/update_status", json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logging.exception("Failed to update trial status")
