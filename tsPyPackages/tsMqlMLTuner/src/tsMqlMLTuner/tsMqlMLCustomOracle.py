# filename: tsMqlMLCustomOracle.py
import threading
import logging
import uuid
import json
import os
import random
from typing import Dict, Optional
from pathlib import Path
import datetime # Added import for datetime

logger = logging.getLogger("Oracle")

class CustomOracle:
    def __init__(self, max_trials=100, objective=None, directory=None, project_name="default_project", seed=None, overwrite=True):
        self.max_trials = max_trials
        self.trials_created = 0
        self.trials_completed = 0
        self.trial_lock = threading.Lock()
        self.trials = {}  # trial_id -> trial dict
        self.ongoing_trials = set()

        self.objective = objective
        self.project_name = project_name
        self.seed = seed or 42
        self.overwrite = overwrite
        self.directory = Path(directory or ".") / project_name
        self.directory.mkdir(parents=True, exist_ok=True)

        random.seed(self.seed)

        if not overwrite:
            self._load_existing_trials()

    def _get_trial_path(self, trial_id: str) -> Path:
        return self.directory / f"{trial_id}.json"

    def _save_trial(self, trial: Dict):
        path = self._get_trial_path(trial["trial_id"])
        with path.open("w", encoding="utf-8") as f:
            json.dump(trial, f, indent=2)
        logger.debug(f"[CustomOracle] Trial {trial['trial_id']} saved to {path}")

    def _load_trial(self, trial_id: str) -> Optional[Dict]:
        path = self._get_trial_path(trial_id)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                trial = json.load(f)
            return trial
        return None

    def _load_existing_trials(self):
        # This method is called if overwrite is False, to resume a previous session.
        logger.info(f"[CustomOracle] Attempting to load existing trials from {self.directory}")
        for trial_file in self.directory.glob("*.json"):
            try:
                with trial_file.open("r", encoding="utf-8") as f:
                    trial = json.load(f)
                if "trial_id" in trial:
                    self.trials[trial["trial_id"]] = trial
                    self.trials_created += 1
                    if trial.get("status") == "COMPLETED":
                        self.trials_completed += 1
                    elif trial.get("status") == "RUNNING":
                        self.ongoing_trials.add(trial["trial_id"])
                else:
                    logger.warning(f"Skipping malformed trial file: {trial_file}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON from {trial_file}: {e}")
            except Exception as e:
                logger.error(f"Error loading trial from {trial_file}: {e}")
        logger.info(f"[CustomOracle] Loaded {len(self.trials)} existing trials.")

    def _sample_hyperparameters(self) -> Dict:
        """
        Placeholder for real sampling logic.
        This should be replaced with actual hyperparameter sampling based on the tuner type (Hyperband, RandomSearch, Bayesian).
        For now, it returns dummy hyperparameters.
        """
        # Example: a simple random search for dummy parameters
        return {
            "learning_rate": random.choice([0.01, 0.001, 0.0001]),
            "num_layers": random.randint(1, 3),
            "units": random.choice([32, 64, 128, 256]),
            "activation": random.choice(["relu", "tanh"]),
            "optimizer": random.choice(["adam", "rmsprop"]),
            "dropout_rate": random.uniform(0.1, 0.5)
        }

    def get_trial(self, tuner_id: str) -> Optional[Dict]:
        with self.trial_lock:
            if self.trials_created >= self.max_trials:
                logger.info("[CustomOracle] Max trials reached. No new trials will be created.")
                return None

            # Try to find a previously created but not yet completed trial (e.g., FAILED or IDLE)
            for trial_id, trial in self.trials.items():
                if trial.get("status") in ["FAILED", "IDLE"] and trial_id not in self.ongoing_trials:
                    trial["status"] = "RUNNING"
                    trial["tuner_id"] = tuner_id
                    self.ongoing_trials.add(trial_id)
                    self._save_trial(trial)
                    logger.info(f"[CustomOracle] Resuming trial {trial_id} for tuner {tuner_id}.")
                    return trial

            # If no existing trials to resume, create a new one
            trial_id = str(uuid.uuid4())
            hyperparameters = self._sample_hyperparameters() # Call the sampling logic
            trial = {
                "trial_id": trial_id,
                "tuner_id": tuner_id,
                "hyperparameters": hyperparameters,
                "score": None,
                "status": "RUNNING",
                "start_time": datetime.datetime.now().isoformat()
            }
            self.trials[trial_id] = trial
            self.ongoing_trials.add(trial_id)
            self.trials_created += 1
            self._save_trial(trial)
            logger.info(f"[CustomOracle] Created new trial {trial_id} for tuner {tuner_id}. Total trials created: {self.trials_created}/{self.max_trials}")
            return trial

    def report_trial_result(self, trial_id: str, score: float, status: str = "COMPLETED"):
        with self.trial_lock:
            if trial_id not in self.trials:
                logger.warning(f"[CustomOracle] Unknown trial_id: {trial_id}")
                return
            self.trials[trial_id]["score"] = score
            self.trials[trial_id]["status"] = status
            self.ongoing_trials.discard(trial_id)
            if status == "COMPLETED":
                self.trials_completed += 1
            self._save_trial(self.trials[trial_id])
            logger.info(f"[CustomOracle] Trial {trial_id} reported as {status} with score {score}")

    def update_trial_status(self, trial_id: str, status: str):
        with self.trial_lock:
            if trial_id not in self.trials:
                logger.warning(f"[CustomOracle] Unknown trial_id: {trial_id}")
                return
            self.trials[trial_id]["status"] = status
            if status != "RUNNING":
                self.ongoing_trials.discard(trial_id)
            self._save_trial(self.trials[trial_id])
            logger.info(f"[CustomOracle] Trial {trial_id} status updated to {status}")

    def get_best_trial(self) -> Optional[Dict]:
        with self.trial_lock:
            completed_trials = [t for t in self.trials.values() if t.get("status") == "COMPLETED" and t.get("score") is not None]
            if not completed_trials:
                return None

            # Assuming objective is always to minimize val_loss (lower is better)
            # You might need to make this configurable (e.g., maximize accuracy)
            best_trial = min(completed_trials, key=lambda t: t["score"])
            return best_trial
