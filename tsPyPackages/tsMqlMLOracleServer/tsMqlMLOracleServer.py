#!/usr/bin/env python3
# File: oracle_server.py
from flask import Flask, request, jsonify
import threading
import os
import json
import pickle
from datetime import datetime

app = Flask(__name__)

# ---- Global Oracle Data ----
lock = threading.Lock()
trials = {}
best_trial = None
best_score = float('inf')
save_model_dir = os.path.join(os.getcwd(), "saved_models")

os.makedirs(save_model_dir, exist_ok=True)

# ---- API Endpoints ----

@app.route('/')
def index():
    return "🎯 Oracle Server is running!"

@app.route('/get_trial', methods=['GET'])
def get_trial():
    with lock:
        # Simple trial suggestion (randomly pick hyperparams)
        # Later you can make this smarter!
        trial_id = str(len(trials))
        hyperparams = {
            "learning_rate": 10**(-4 * (len(trials) % 3)),  # cycle 1e-4, 1e-8, etc.
            "num_layers": (len(trials) % 3) + 1,
            "units": (len(trials) * 32) % 512 + 32
        }
        trials[trial_id] = {
            "hyperparams": hyperparams,
            "status": "RUNNING",
            "score": None,
            "created_at": str(datetime.now())
        }
        print(f"[Oracle] Assigned Trial {trial_id}: {hyperparams}")
        return jsonify({"trial_id": trial_id, "hyperparameters": hyperparams})

@app.route('/report_result', methods=['POST'])
def report_result():
    data = request.get_json()
    trial_id = data.get('trial_id')
    score = data.get('score')

    if not trial_id or score is None:
        return jsonify({"error": "Missing trial_id or score"}), 400

    with lock:
        if trial_id in trials:
            trials[trial_id]['status'] = "COMPLETED"
            trials[trial_id]['score'] = score
            print(f"[Oracle] Received Result for Trial {trial_id}: score={score}")

            # Update best trial
            global best_score, best_trial
            if score < best_score:
                best_score = score
                best_trial = trials[trial_id]
                print(f"[Oracle] 🎯 New Best Trial {trial_id} with score {score}")

            return jsonify({"status": "OK"})
        else:
            return jsonify({"error": "Invalid trial_id"}), 404

@app.route('/upload_model', methods=['POST'])
def upload_model():
    trial_id = request.form.get('trial_id')
    file = request.files.get('model')

    if not trial_id or not file:
        return jsonify({"error": "Missing trial_id or model file"}), 400

    model_path = os.path.join(save_model_dir, f"model_trial_{trial_id}.keras")
    file.save(model_path)

    print(f"[Oracle] 📦 Model saved for Trial {trial_id} at {model_path}")

    return jsonify({"status": "Model saved", "path": model_path})

@app.route('/trials', methods=['GET'])
def list_trials():
    with lock:
        return jsonify(trials)

@app.route('/best', methods=['GET'])
def best_result():
    with lock:
        return jsonify(best_trial or {})

# ---- Start Server ----

if __name__ == "__main__":
    port = int(os.environ.get("ORACLE_SERVER_PORT", 9000))
    print(f"🚀 Oracle Server starting at http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
