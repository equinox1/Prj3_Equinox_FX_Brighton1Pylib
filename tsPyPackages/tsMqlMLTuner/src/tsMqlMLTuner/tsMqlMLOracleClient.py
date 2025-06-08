import threading
import time
import requests
import logging
import os # Import os to access environment variables
# -- Set up global logging --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()
clientlog_config.setup_logging()  # Ensure logging is configured before getting the logger
logger = logging.getLogger(__name__)
# -- end of logging setup ----

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')


class OracleClient:
    def __init__(self, host=None, port=None):
        # Use provided host/port or fall back to configured values
        self.host = host if host is not None else xerces_server
        self.port = port if port is not None else xerces_port
        self.url = f"http://{self.host}:{self.port}"
        logger.info(f"[OracleClient] Initialized with Oracle Server URL: {self.url}")

    def register_client(self, tuner_id):
        try:
            payload = {"tuner_id": tuner_id}
            response = requests.post(f"{self.url}/register", json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"[OracleClient] Registered as tuner_id: {tuner_id}. Response: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Failed to register: {e}")
            return None

    def get_trial(self):
        try:
            # Get tuner_id from environment variable, default to 'unknown_tuner'
            tuner_id = os.environ.get('TUNER_ID', 'unknown_tuner')
            payload = {"tuner_id": tuner_id} # Include tuner_id in the payload
            # Corrected: Call the /request_trial endpoint, not /get_trial
            response = requests.post(f"{self.url}/request_trial", json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Error requesting trial: {e}")
            return None
        except Exception as e:
            logger.exception("Unexpected error in OracleClient.get_trial")

    def get_best_trial(self):
        try:
            response = requests.get(f"{self.url}/list_trials", timeout=10)
            response.raise_for_status()
            trials = response.json().get("trials", [])
            trials = [t for t in trials if t.get("score") is not None]
            if not trials:
                return None
            trials.sort(key=lambda t: t["score"])  # Assuming lower score is better
            return trials[0]
        except Exception as e:
            logging.error(f"[OracleClient] Failed to get best trial: {e}")
            return None

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
        except requests.Timeout:
            logging.error("Timed out while updating trial status to OracleServer")
        except Exception as e:
            logging.exception("Unexpected error in OracleClient.update_trial_status")
