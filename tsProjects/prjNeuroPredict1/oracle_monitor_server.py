# File: oracle_monitor_server.py
from flask import Flask, jsonify
import requests
import os

app = Flask(__name__)

ORACLE_SERVER_IP = os.environ.get('ORACLE_SERVER_IP', '127.0.0.1')
ORACLE_SERVER_PORT = int(os.environ.get('ORACLE_SERVER_PORT', 9000))

@app.route('/')
def index():
    return "🎯 Oracle Monitor is running. Visit /trials to see live updates."

@app.route('/trials')
def list_trials():
    try:
        response = requests.get(f"http://{ORACLE_SERVER_IP}:{ORACLE_SERVER_PORT}/v1/trials")
        trials = response.json()
        return jsonify(trials)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/trials/status')
def trial_status_summary():
    try:
        response = requests.get(f"http://{ORACLE_SERVER_IP}:{ORACLE_SERVER_PORT}/v1/trials")
        trials = response.json()
        status_count = {}
        for trial in trials.get('trials', []):
            status = trial.get('status', 'UNKNOWN')
            status_count[status] = status_count.get(status, 0) + 1
        return jsonify(status_count)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get('MONITOR_PORT', 5001))
    print(f"🚀 Oracle Monitor running on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)
