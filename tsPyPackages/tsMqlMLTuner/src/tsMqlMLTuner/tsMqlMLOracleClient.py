import threading
import time
import requests
import logging # Ensure logging is imported
import os # Import os to access environment variables

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for OracleClient. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
# -- end of logging setup ----

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

# Tuner and backend parameters from environment or defaults
tuner_model = tune_params.get('tuner_type', 'hyperband')
backend = tune_params.get('backend', 'tensorflow')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

class OracleClient:
    def __init__(self, oracle_host="127.0.0.1", oracle_port=9000):
        self.url = f"http://{oracle_host}:{oracle_port}"
        logger.info(f"[OracleClient] Initialized with Oracle Server URL: {self.url}")
        logger.debug(f"[OracleClient] Backend: {backend}, Tuner Model: {tuner_model}")

    def get_trial(self):
        """Requests a new trial from the Oracle Server."""
        logger.info("[OracleClient] Requesting new trial from Oracle Server.")
        try:
            response = requests.post(f"{self.url}/get_trial", timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors
            trial_data = response.json()
            if trial_data.get("trial_id"):
                logger.info(f"[OracleClient] Received trial_id: {trial_data['trial_id']}")
                logger.debug(f"[OracleClient] Trial hyperparameters: {trial_data['hyperparameters']}")
                return trial_data
            else:
                logger.warning(f"[OracleClient] Oracle Server returned no new trial: {trial_data.get('message', 'Unknown reason')}")
                return None
        except requests.exceptions.Timeout:
            logger.error("[OracleClient] Timed out while requesting trial from Oracle Server.")
            return None
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"[OracleClient] Connection error to Oracle Server: {ce}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Error requesting trial: {e}")
            return None
        except Exception as e:
            logger.exception("Unexpected error in OracleClient.get_trial")
            return None

    def get_best_trial(self):
        """Retrieves the best trial from the Oracle Server."""
        logger.info("[OracleClient] Requesting best trial from Oracle Server.")
        try:
            response = requests.get(f"{self.url}/list_trials", timeout=10)
            response.raise_for_status()
            trials = response.json().get("trials", [])
            
            # Filter for completed trials with a score
            trials = [t for t in trials if t.get("score") is not None and t.get("status") == "COMPLETED"]
            
            if not trials:
                logger.info("[OracleClient] No completed trials with scores found.")
                return None
            
            # Assuming lower score is better for objective (e.g., validation loss)
            trials.sort(key=lambda t: t["score"])
            best_trial = trials[0]
            logger.info(f"[OracleClient] Best trial found: {best_trial.get('trial_id')} with score: {best_trial.get('score')}")
            logger.debug(f"[OracleClient] Best trial details: {best_trial}")
            return best_trial
        except requests.exceptions.Timeout:
            logger.error("[OracleClient] Timed out while getting best trial from Oracle Server.")
            return None
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"[OracleClient] Connection error to Oracle Server when getting best trial: {ce}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Error getting best trial: {e}")
            return None
        except Exception as e:
            logger.exception("Unexpected error in OracleClient.get_best_trial")
            return None

    def report_trial_result(self, trial_id, result):
        """Reports the result of a trial to the Oracle Server."""
        logger.info(f"[OracleClient] Reporting result for trial_id: {trial_id}")
        try:
            payload = {"trial_id": trial_id, "result": result}
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"[OracleClient] Successfully reported result for trial_id: {trial_id}")
        except requests.exceptions.Timeout:
            logger.error(f"[OracleClient] Timed out while reporting result for trial {trial_id} to OracleServer")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"[OracleClient] Connection error reporting result for trial {trial_id}: {ce}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Error reporting result for trial {trial_id}: {e}")
        except Exception as e:
            logger.exception("Unexpected error in OracleClient.report_trial_result")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        """Updates the status of a trial on the Oracle Server."""
        logger.info(f"[OracleClient] Updating status for trial_id: {trial_id} to {status}")
        try:
            payload = {"trial_id": trial_id, "status": status}
            response = requests.post(f"{self.url}/update_status", json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"[OracleClient] Successfully updated status for trial_id: {trial_id} to {status}")
        except requests.exceptions.Timeout:
            logger.error(f"[OracleClient] Timed out while updating status for trial {trial_id} to OracleServer")
        except requests.exceptions.ConnectionError as ce:
            logger.error(f"[OracleClient] Connection error updating status for trial {trial_id}: {ce}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Error updating status for trial {trial_id}: {e}")
        except Exception as e:
            logger.exception("Unexpected error in OracleClient.update_trial_status")
