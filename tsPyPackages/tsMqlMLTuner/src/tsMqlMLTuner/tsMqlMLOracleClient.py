import threading
import time
import requests
import logging

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
