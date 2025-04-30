import requests

class OracleClient:
    def __init__(self, host="localhost", port=9000):
        self.url = f"http://{host}:{port}"

    def get_trial(self):
        response = requests.get(f"{self.url}/get_trial")
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to get trial: {response.status_code} {response.text}")

    def report_trial_result(self, trial_id, result):
        payload = {"trial_id": trial_id, "result": result}
        response = requests.post(f"{self.url}/report_result", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to report result: {response.status_code} {response.text}")
    
    def update_trial_status(self, trial_id, status="COMPLETED"):
        payload = {"trial_id": trial_id, "status": status}
        response = requests.post(f"{self.url}/update_status", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to update status: {response.status_code} {response.text}")