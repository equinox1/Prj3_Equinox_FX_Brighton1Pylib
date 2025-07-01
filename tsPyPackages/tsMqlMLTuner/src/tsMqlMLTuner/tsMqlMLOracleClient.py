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
    def __init__(self, host=None, port=None, url=None, tuner_id="default_tuner"):
        self.host = host if host else xerces_server
        self.port = port if port else xerces_port
        self.url = url if url else f"http://{self.host}:{self.port}"
        self.tuner_id = tuner_id
        # Use the module-level logger that was already initialized
        self.logger = logger
        self.logger.info(f"[OracleClient] Initialized for tuner_id: {self.tuner_id}, connecting to Oracle at {self.url}")

    def get_trial(self, tuner_id):
        # CORRECTED: Ensure max_retries and retry_delay are integers
        max_retries = int(tune_params.get('oracle_client_max_retries', 5))
        retry_delay = int(tune_params.get('oracle_client_retry_delay', 5))
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"[OracleClient] {tuner_id} requesting trial (Attempt {attempt + 1}/{max_retries})...")
                payload = {"tuner_id": tuner_id}
                response = requests.post(f"{self.url}/get_trial", json=payload, timeout=10)
                response.raise_for_status()
                trial_data = response.json()
                self.logger.info(f"[OracleClient] {tuner_id} received trial: {trial_data.get('trial_id')}, status: {trial_data.get('status')}")
                return trial_data
            except requests.exceptions.ConnectionError as ce:
                self.logger.warning(f"[OracleClient] Connection error to OracleServer: {ce}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except requests.exceptions.Timeout:
                self.logger.warning(f"[OracleClient] Timeout requesting trial. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except requests.exceptions.HTTPError as he:
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
            response = requests.get(f"{self.url}/best_trial", timeout=30)
            response.raise_for_status()
            best_trial_info = response.json()
            self.logger.info(f"[OracleClient] Received best trial: {best_trial_info.get('trial_id')}, score: {best_trial_info.get('score')}")
            return best_trial_info
        except Exception as e:
            self.logger.error(f"[OracleClient] Failed to get best trial: {e}")
            return None

    def report_trial_result(self, trial_id, result):
        try:
            payload = {"trial_id": trial_id, "result": result}
            # Retaining 30 seconds for result reporting, can be increased if needed
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=30) 
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

