#!/usr/bin/env python3
# File: oracle_client.py

import requests
import os

class OracleClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip('/')

    def get_trial(self):
        """Ask Oracle for the next hyperparameter trial."""
        response = requests.get(f"{self.server_url}/get_trial")
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to get trial: {response.text}")

    def report_result(self, trial_id, score):
        """Send back the trial result (validation score)."""
        payload = {"trial_id": trial_id, "score": score}
        response = requests.post(f"{self.server_url}/report_result", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to report result: {response.text}")

    def upload_model(self, trial_id, model_path):
        """Upload the trained model to Oracle."""
        with open(model_path, 'rb') as model_file:
            files = {'model': model_file}
            data = {'trial_id': trial_id}
            response = requests.post(f"{self.server_url}/upload_model", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to upload model: {response.text}")

    def list_trials(self):
        """Get the full list of all trials."""
        response = requests.get(f"{self.server_url}/trials")
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to get trials: {response.text}")

    def get_best_trial(self):
        """Get the best trial so far."""
        response = requests.get(f"{self.server_url}/best")
        if response.status_code == 200:
            return response.json()
        else:
            raise RuntimeError(f"Failed to get best trial: {response.text}")
