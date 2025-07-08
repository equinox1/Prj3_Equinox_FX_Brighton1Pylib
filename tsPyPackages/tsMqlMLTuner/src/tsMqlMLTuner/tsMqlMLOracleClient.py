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
gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband

xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

# Extract backend for logging path - crucial for correct log file path
# This will be passed to initialize_logging. It can also be obtained from env if passed by launcher.
backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch')) # Default to pytorch if not specified



class OracleClient:
    def __init__(self, host=None, port=None, url=None, tuner_id="default_tuner", timeout=30.0):
        # If a full URL is provided, use it directly. Otherwise, construct from host/port.
        if url:
            self.url = url
        else:
            self.host = host if host else xerces_server
            self.port = port if port else xerces_port
            self.url = f"http://{self.host}:{self.port}"

        self.tuner_id = tuner_id
        self.timeout = timeout # Store the timeout value
        # Use the module-level logger that was already initialized
        self.logger = logger
        self.logger.info(f"[OracleClient] Initialized for tuner_id: {self.tuner_id}, connecting to Oracle at {self.url} with timeout {self.timeout}s")

    def get_trial(self, tuner_id):
        # CORRECTED: Increased max_retries and retry_delay for OracleClient
        max_retries = int(tune_params.get('oracle_client_max_retries', 60)) # Increased from 5 to 60
        retry_delay = int(tune_params.get('oracle_client_retry_delay', 2)) # Changed from 5 to 2

        for attempt in range(max_retries):
            try:
                self.logger.info(f"[OracleClient] {tuner_id} requesting trial (Attempt {attempt + 1}/{max_retries})...")
                # Use /request_trial endpoint as per OracleServer
                payload = {"tuner_id": tuner_id}
                # Use the instance's timeout for the request
                response = requests.post(f"{self.url}/request_trial", json=payload, timeout=self.timeout)
                response.raise_for_status()
                trial_data = response.json()
                
                # Check if the server explicitly returned None for trial_id or a specific status
                if trial_data.get('trial_id') is None:
                    if trial_data.get('status') == "NO_TRIALS_AVAILABLE":
                        self.logger.info(f"[OracleClient] {tuner_id} received 'NO_TRIALS_AVAILABLE' from server. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        continue # Continue to the next retry attempt
                    else:
                        self.logger.info(f"[OracleClient] {tuner_id} received no new trial from server (unknown reason). Max trials reached or no idle trials. Stopping retries.")
                        return None # Stop retrying if trial_id is None but not due to NO_TRIALS_AVAILABLE status

                self.logger.info(f"[OracleClient] {tuner_id} received trial: {trial_data.get('trial_id')}")
                return trial_data
            except requests.exceptions.ConnectionError as ce:
                self.logger.warning(f"[OracleClient] Connection error to OracleServer: {ce}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except requests.exceptions.Timeout:
                self.logger.warning(f"[OracleClient] Timeout requesting trial. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except requests.exceptions.HTTPError as he:
                # This block will now primarily catch non-200 HTTP errors from the server
                self.logger.error(f"[OracleClient] HTTP error requesting trial: {he.response.status_code} - {he.response.text}")
                return None # Don't retry on HTTP errors that indicate server-side issues
            except Exception as e:
                self.logger.error(f"[OracleClient] Unexpected error requesting trial: {e}", exc_info=True)
                return None # Don't retry on unexpected errors
        self.logger.error(f"[OracleClient] Max retries reached for {tuner_id}. Could not get trial.")
        return None

    def get_best_trial(self):
        try:
            self.logger.info("[OracleClient] Requesting best trial from OracleServer...")
            response = requests.get(f"{self.url}/get_best_trial", timeout=self.timeout) # Use the instance's timeout
            response.raise_for_status()
            best_trial_info = response.json()
            self.logger.info(f"[OracleClient] Received best trial: {best_trial_info.get('best_trial', {}).get('trial_id')}, score: {best_trial_info.get('best_trial', {}).get('score')}")
            return best_trial_info.get('best_trial') # Return the 'best_trial' dictionary
        except Exception as e:
            self.logger.error(f"[OracleClient] Failed to get best trial: {e}")
            return None

    def report_trial_result(self, trial_id, result):
        try:
            # The OracleServer expects a dictionary for 'result' in TrialResult model
            payload = {"trial_id": trial_id, "result": {"val_loss": result}} # Assuming 'val_loss' is the objective
            # Use the instance's timeout for result reporting
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=self.timeout)
            response.raise_for_status()
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
