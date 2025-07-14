# filename: tsMqlMLOracleClient.py
import threading
import time
import requests
import logging
import os # Import os to access environment variables
# Setup logging for this module
logger = logging.getLogger(__name__)


from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})
tuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband

xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
# CHANGE THIS LINE: Align the client's connection target with the server's binding.
# If your server is binding to "127.0.0.1" for local access, the client should also connect to it.
xerces_server = app_params.get('xerces_server', '127.0.0.1') # Changed from '192.168.1.103' to '127.0.0.1'
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# Extract backend for logging path - crucial for correct log file path
# This will be passed to initialize_logging. It can also be obtained from env
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

class OracleClient:
    def __init__(self, oracle_url: str, request_timeout: int = 30):
        self.url = oracle_url
        self.timeout = request_timeout
        self.logger = logging.getLogger("OracleClient")
        self.logger.info(f"OracleClient initialized with URL: {self.url} and timeout: {self.timeout}s")

    def get_trial(self, tuner_id: str):
        try:
            payload = {"tuner_id": tuner_id}
            response = requests.post(f"{self.url}/get_trial", json=payload, timeout=self.timeout)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            trial_data = response.json()
            if trial_data:
                self.logger.info(f"Received trial {trial_data.get('trial_id')} from OracleServer.")
            else:
                self.logger.info("No new trial available from OracleServer.")
            return trial_data
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to OracleServer at {self.url}: {e}")
            return None
        except requests.exceptions.Timeout:
            self.logger.error("Timed out while getting trial from OracleServer")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error from OracleServer: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.get_trial")
            return None

    def report_trial_result(self, trial_id, score, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "score": score, "status": status}
            # Use the instance's timeout for result reporting
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=self.timeout)
            response.raise_for_status()
            self.logger.info(f"Reported trial {trial_id} with score {score} and status {status} to OracleServer.")
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to OracleServer at {self.url}: {e}")
        except requests.Timeout:
            self.logger.error("Timed out while reporting result to OracleServer")
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.report_trial_result")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "status": status}
            # Use the instance's timeout for status updates
            response = requests.post(f"{self.url}/update_status", json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout:
            self.logger.error("Timed out while updating trial status to OracleServer")
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.update_trial_status")

    def is_server_available(self):
        """Checks if the Oracle server is reachable."""
        try:
            response = requests.get(f"{self.url}/status", timeout=self.timeout) # Use the instance's timeout
            response.raise_for_status()
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            self.logger.debug(f"Oracle server at {self.url} is not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during Oracle server availability check: {e}", exc_info=True)
            return False

    def get_best_trial(self):
        try:
            response = requests.get(f"{self.url}/get_best_trial", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to OracleServer at {self.url}: {e}")
            return None
        except requests.exceptions.Timeout:
            self.logger.error("Timed out while getting best trial from OracleServer")
            return None
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"HTTP error from OracleServer when getting best trial: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.get_best_trial")
            return None
