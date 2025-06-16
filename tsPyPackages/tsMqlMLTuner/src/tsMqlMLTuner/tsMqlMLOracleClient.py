import threading
import time
import requests
import logging
import os # Import os to access environment variables


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

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__,
    loglevel='INFO',
    # Explicitly set the logfile name to ensure consistency
    logfile='tsneuropredict_app.log',
    # Pass the determined backend so logging goes into the correct subdirectory
    backend=backend_for_log # Pass the backend to the logging setup
)





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
            # Increased timeout to 60 seconds
            response = requests.post(f"{self.url}/register", json=payload, timeout=60)
            response.raise_for_status()
            logger.info(f"[OracleClient] Registered as tuner_id: {tuner_id}. Response: {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"[OracleClient] Failed to register: {e}")
            return None

    def get_trial(self, max_retries=3, retry_delay=5):
        """
        Requests a new trial from the Oracle Server with retry logic.
        Args:
            max_retries (int): Maximum number of times to retry the request.
            retry_delay (int): Delay in seconds between retries.
        Returns:
            dict: Trial information if successful, None otherwise.
        """
        tuner_id = os.environ.get('TUNER_ID', 'unknown_tuner')
        payload = {"tuner_id": tuner_id}

        for attempt in range(max_retries):
            try:
                logger.info(f"[OracleClient] Attempt {attempt + 1}/{max_retries}: Requesting trial from {self.url}/request_trial for tuner_id: {tuner_id}")
                response = requests.post(f"{self.url}/request_trial", json=payload, timeout=60)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                return response.json()
            except requests.exceptions.Timeout as e:
                logger.warning(f"[OracleClient] Attempt {attempt + 1}: Timed out while requesting trial from Oracle Server: {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"[OracleClient] Attempt {attempt + 1}: Connection error while requesting trial from Oracle Server: {e}")
            except requests.exceptions.HTTPError as e:
                # For 4xx or 5xx errors, log and potentially break if it's not a transient error
                logger.error(f"[OracleClient] Attempt {attempt + 1}: HTTP error requesting trial ({e.response.status_code}): {e.response.text}")
                # If the server explicitly says no more trials (e.g., 204 or specific message), don't retry
                if e.response.status_code == 204: # No Content, might mean no trials
                    logger.info("[OracleClient] Server indicated no more trials via 204 No Content. Not retrying.")
                    return {"trial": None} # Return a structured response indicating no trials
                if e.response.status_code == 404 and "No more trials available" in e.response.text:
                     logger.info("[OracleClient] Server indicated no more trials. Not retrying.")
                     return {"trial": None}
            except Exception as e:
                logger.exception(f"[OracleClient] Attempt {attempt + 1}: Unexpected error in OracleClient.get_trial")

            if attempt < max_retries - 1:
                logger.info(f"[OracleClient] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error(f"[OracleClient] Failed to get trial after {max_retries} attempts.")
        return None


    def get_best_trial(self):
        try:
            # Increased timeout to 60 seconds
            response = requests.get(f"{self.url}/list_trials", timeout=60)
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
            # Retaining 30 seconds for result reporting, can be increased if needed
            response = requests.post(f"{self.url}/report_result", json=payload, timeout=30) 
            response.raise_for_status()
        except requests.Timeout:
            logging.error("Timed out while reporting result to OracleServer")
        except Exception as e:
            logging.exception("Unexpected error in OracleClient.report_trial_result")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        try:
            payload = {"trial_id": trial_id, "status": status}
            # Increased timeout to 60 seconds
            response = requests.post(f"{self.url}/update_status", json=payload, timeout=60)
            response.raise_for_status()
        except requests.Timeout:
            logging.error("Timed out while updating trial status to OracleServer")
        except Exception as e:
            logging.exception("Unexpected error in OracleClient.update_trial_status")
