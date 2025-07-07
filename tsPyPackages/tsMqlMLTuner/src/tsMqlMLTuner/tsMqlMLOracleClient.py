import threading
import time
import requests
import logging
import os # Import os to access environment variables
# Setup logging for this module
logger = logging.getLogger(__name__)

# Removed: from tsMqlOverrides import CMqlOverrides
# Removed: mql_overrides = CMqlOverrides()
# Removed: app_params = mql_overrides.env.all_params().get("app", {})
# Removed: tune_params = mql_overrides.env.all_params().get('mltune', {})
# Removed: gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband

# Removed: xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
# Removed: xerces_server = app_params.get('xerces_server', '192.168.1.103')
# Removed: xerces_port = app_params.get('xerces_port', 9000)
# Removed: xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# Removed: backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified


class OracleClient:
    def __init__(self, host=None, port=None, url=None, tuner_id="default_tuner"):
        self.host = host
        self.port = port
        self.url = url
        if not self.url:
            if not self.host or not self.port:
                raise ValueError("Either 'url' or both 'host' and 'port' must be provided for OracleClient.")
            self.url = f"http://{self.host}:{self.port}"
        self.tuner_id = tuner_id
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"[OracleClient] Initialized for tuner_id: {self.tuner_id}, connecting to Oracle at {self.url}")


    def request_trial(self, tuner_id, max_retries=5, retry_delay=5):
        payload = {"tuner_id": tuner_id}
        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"[OracleClient] {tuner_id} requesting trial (Attempt {attempt}/{max_retries})...")
                # Increased timeout to 60 seconds for trial requests
                response = requests.post(f"{self.url}/request_trial", json=payload, timeout=60)
                response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                
                trial_data = response.json()
                self.logger.info(f"[OracleClient] {tuner_id} received trial data: {trial_data}") # Log full response
                return trial_data
            except requests.exceptions.ConnectionError as e:
                self.logger.error(f"[OracleClient] Connection error requesting trial for {tuner_id}: {e}")
                if attempt < max_retries:
                    self.logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.critical(f"Max retries reached for {tuner_id}. Could not connect to OracleServer.")
                    return None
            except requests.exceptions.Timeout:
                self.logger.error(f"[OracleClient] Timed out requesting trial for {tuner_id}.")
                if attempt < max_retries:
                    self.logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.critical(f"Max retries reached for {tuner_id}. OracleServer timed out.")
                    return None
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"[OracleClient] HTTP error requesting trial for {tuner_id}: {e.response.status_code} - {e.response.text}")
                # If it's a 200 OK but with a detail, handle it as no more trials
                if e.response.status_code == 200 and "No more trials available" in e.response.text:
                    self.logger.info(f"[OracleClient] {tuner_id} received no new trial from server. Max trials reached or no idle trials.")
                    return None
                else:
                    # For other HTTP errors, retry or fail
                    if attempt < max_retries:
                        self.logger.warning(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        self.logger.critical(f"Max retries reached for {tuner_id}. OracleServer returned HTTP error.")
                        return None
            except Exception as e:
                self.logger.exception(f"[OracleClient] Unexpected error requesting trial for {tuner_id}")
                if attempt < max_retries:
                    self.logger.warning(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.critical(f"Max retries reached for {tuner_id}. Unexpected error.")
                    return None
        return None


    def report_result(self, trial_id, score, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "score": score, "status": status}
            # Increased timeout to 60 seconds
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=60)
            response.raise_for_status()
        except requests.Timeout:
            self.logger.error("Timed out while reporting result to OracleServer")
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.report_trial_result")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "status": status}
            # Increased timeout to 60 seconds
            response = requests.post(f"{self.url}/update_status", json=payload, timeout=60)
            response.raise_for_status()
        except requests.Timeout:
            self.logger.error("Timed out while updating trial status to OracleServer")
        except Exception as e:
            self.logger.exception("Unexpected error in OracleClient.update_trial_status")

    def is_server_available(self):
        """Checks if the Oracle server is reachable."""
        try:
            response = requests.get(f"{self.url}/status", timeout=5)
            response.raise_for_status()
            return True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            self.logger.debug(f"Oracle server at {self.url} is not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during Oracle server availability check: {e}", exc_info=True)
            return False
