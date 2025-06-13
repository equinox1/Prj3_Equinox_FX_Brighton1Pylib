# --- File: tsMqlMLCustomOracle.py ---
from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib

import logging
import os
from pathlib import Path
import threading
import random # Import random for sampling hyperparameters
import uuid # Import uuid for generating unique IDs
from typing import Optional # Import Optional for type hinting

logger = logging.getLogger(__name__)

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})

class CustomOracle(Oracle):
    def __init__(
        self,
        objective="val_loss",
        max_trials=50,
        directory="oracle_dir", # Passed to CustomOracle
        project_name="default_project", # Passed to CustomOracle
        seed=42,
        reset_trials=True,
        **kwargs # Accept additional kwargs
    ):
        # Extract 'overwrite' and other specific kwargs before passing to super()
        self_overwrite = kwargs.pop('overwrite', False)
        tune_new_entries = kwargs.pop('tune_new_entries', True)
        allow_new_entries = kwargs.pop('allow_new_entries', True)

        # Call the base Oracle's __init__ with arguments it expects.
        # DO NOT pass 'directory' or 'project_name' to super().__init__()
        # if the TypeError occurred previously.
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            **kwargs # Pass any remaining kwargs
        )
        self.logger = logging.getLogger(__name__)
        self.objective = objective
        self.max_trials = max_trials

        # Explicitly set the internal KerasTuner expected attributes
        # `_directory` typically refers to the base directory, and `_project_name` to the project name itself.
        # These are used by the base Oracle's `_project_dir` property and saving/loading methods.
        self._directory = directory # Set the base directory as _directory
        self._project_name = project_name # Set the project name as _project_name
        
        # Ensure the root directory for the project (e.g., `directory/project_name/`) exists
        self._project_root_dir = Path(self._directory) / self._project_name
        self._project_root_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the path to the oracle.json save file
        self._save_file = self._project_root_dir / "oracle.json"

        self._seed = seed
        random.seed(self._seed)

        self.lock = threading.Lock()
        # Initialize ongoing_trials as a dictionary, as expected by KerasTuner's Oracle.
        self.ongoing_trials = {} 


        self.logger.info(f"[CustomOracle] Initialized with directory: {self._directory}, project: {self._project_name}")
        self.logger.debug(f"[CustomOracle] Objective: {self.objective}, Max trials: {self.max_trials}")

        if not self_overwrite and self._save_file.exists():
            try:
                # Reload calls super().reload() which uses KerasTuner's internal saving/loading.
                # It now should find self._directory and self._project_name.
                self.reload()
                self.logger.info("[CustomOracle] Successfully reloaded existing trials.")
            except Exception as e:
                self.logger.warning(f"[CustomOracle] Could not reload existing trials: {e}. Starting fresh.")
                self.trials = {} # Fallback to empty trials
                self.ongoing_trials = {} # Also reset ongoing_trials if trials are reset.
        else:
            self.trials = {} # Initialize trials if starting fresh or overwriting
            self.ongoing_trials = {} # Initialize ongoing_trials as a dictionary


        self.tune_params = tune_params
        self.hp_ranges = {
            'trans_dim': (self.tune_params.get('trans_dim_min', 32), self.tune_params.get('trans_dim_max', 256), self.tune_params.get('trans_dim_step', 32)),
            'lstm_units': (self.tune_params.get('lstm_units_min', 32), self.tune_params.get('lstm_units_max', 128), self.tune_params.get('lstm_units_step', 32)),
            'gru_units': (self.tune_params.get('gru_units_min', 32), self.tune_params.get('gru_units_max', 128), self.tune_params.get('gru_units_step', 32)),
            'cnn_units': (self.tune_params.get('cnn_units_min', 32), self.tune_params.get('cnn_units_max', 128), self.tune_params.get('cnn_units_step', 32)),
            'trans_heads': (self.tune_params.get('trans_heads_min', 2), self.tune_params.get('trans_heads_max', 8), self.tune_params.get('trans_heads_step', 2)),
            'trans_ff': (self.tune_params.get('trans_ff_min', 64), self.tune_params.get('trans_ff_max', 512), self.tune_params.get('trans_ff_step', 64)),
            'dense_units': (self.tune_params.get('dense_units_min', 32), self.tune_params.get('dense_units_max', 128), self.tune_params.get('dense_units_step', 32)),
        }

    def populate_space(self, trial_id):
        self.logger.info(f"[CustomOracle] Populating space for trial_id: {trial_id}")
        if len(self.trials) >= self.max_trials:
            self.logger.info(f"[CustomOracle] Reached max_trials ({self.max_trials}). No more trials to generate.")
            return {"status": trial_lib.TrialStatus.STOPPED, "hyperparameters": {}}

        sampled_hps = {}
        for param_name, (min_val, max_val, step) in self.hp_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int) and isinstance(step, int):
                choices = list(range(min_val, max_val + step, step))
                if not choices:
                    choices = [min_val]
                sampled_value = random.choice(choices)
            elif isinstance(min_val, float) and isinstance(max_val, float) and isinstance(step, float):
                sampled_value = random.uniform(min_val, max_val)
            else:
                self.logger.warning(f"Unsupported hyperparameter range type for {param_name}. Skipping.")
                continue
            sampled_hps[param_name] = sampled_value
            self.logger.debug(f"[CustomOracle] Sampled {param_name}: {sampled_value}")

        hp = HyperParameters()
        for param_name, (min_val, max_val, step) in self.hp_ranges.items():
            if isinstance(min_val, int):
                hp.Int(param_name, min_value=min_val, max_value=max_val, step=step)
            elif isinstance(min_val, float):
                hp.Float(param_name, min_value=min_val, max_value=max_val, step=step)
        
        hp.values = sampled_hps

        new_trial = self.new_trial(trial_id, hp.get_config())
        new_trial.status = trial_lib.TrialStatus.IDLE

        self.trials[trial_id] = new_trial
        self.save() # Call save after adding the trial

        self.logger.info(f"[CustomOracle] Generated hyperparameters for trial {trial_id}: {sampled_hps}")
        return {
            "status": new_trial.status,
            "hyperparameters": new_trial.hyperparameters.values,
            "trial_id": trial_id,
        }

    def new_trial(self, trial_id=None, hyperparameters_config=None):
        if trial_id is None:
            trial_id = str(uuid.uuid4())

        hp = HyperParameters()
        if hyperparameters_config:
            if 'values' in hyperparameters_config:
                hp.values = hyperparameters_config['values']
                if 'space' in hyperparameters_config:
                    for param_config in hyperparameters_config['space']:
                        class_name = param_config['class_name']
                        config = param_config['config']
                        if class_name == 'Int':
                            hp.Int(config['name'], min_value=config['min_value'], max_value=config['max_value'], step=config.get('step', 1))
                        elif class_name == 'Float':
                            hp.Float(config['name'], min_value=config['min_value'], max_value=config['max_value'], step=config.get('step', 0.01))
                        elif class_name == 'Choice':
                             hp.Choice(config['name'], values=config['values'])
            else:
                hp.values = hyperparameters_config

        trial = trial_lib.Trial(
            hyperparameters=hp,
            trial_id=trial_id,
            status=trial_lib.TrialStatus.IDLE,
        )
        return trial

    def create_trial(self, tuner_id):
        with self.lock:
            self.logger.info(f"[CustomOracle] Request to create trial from tuner_id: {tuner_id}")
            if len(self.trials) >= self.max_trials:
                self.logger.info(f"[CustomOracle] Max trials ({self.max_trials}) reached. Not creating new trial.")
                return None

            trial_id = self.new_trial_id()
            self.logger.debug(f"[CustomOracle] Generated new trial ID: {trial_id}")

            try:
                response = self.populate_space(trial_id)
                self.logger.debug(f"[CustomOracle] Populated space response: {response}")

                if response["status"] == trial_lib.TrialStatus.STOPPED:
                    return None

                trial = self.trials[trial_id]
                # Store the trial object directly, mapping tuner_id to the single active trial
                self.ongoing_trials[tuner_id] = trial 

                self.save()
                self.logger.info(f"[CustomOracle] Created trial {trial_id} with status: {trial.status} and HP: {trial.hyperparameters.values}")
                return trial

            except Exception as e:
                self.logger.error(f"[CustomOracle] Error creating trial: {e}", exc_info=True)
                if trial_id in self.trials:
                    self.trials[trial_id].status = trial_lib.TrialStatus.FAILED
                    # If an error occurs, remove the trial from ongoing_trials for this tuner
                    if tuner_id in self.ongoing_trials and self.ongoing_trials[tuner_id].trial_id == trial_id:
                        del self.ongoing_trials[tuner_id]
                    self.save()
                return None

    def get_trial(self, tuner_id=None):
        with self.lock:
            self.logger.info(f"[CustomOracle] Request for trial from tuner_id: {tuner_id}")
            for trial_id, trial in self.trials.items():
                if trial.status == trial_lib.TrialStatus.IDLE:
                    trial.status = trial_lib.TrialStatus.RUNNING
                    # Assign this trial to the requesting tuner as its ongoing trial
                    self.ongoing_trials[tuner_id] = trial
                    self.save()
                    self.logger.info(f"[CustomOracle] Assigned existing IDLE trial {trial_id} to tuner {tuner_id}.")
                    return {
                        "trial_id": trial_id,
                        "hyperparameters": trial.hyperparameters.values,
                        "status": trial.status,
                        "score": trial.score
                    }

            if len(self.trials) < self.max_trials:
                self.logger.info(f"[CustomOracle] No IDLE trials found. Attempting to create a new trial.")
                new_trial_obj = self.create_trial(tuner_id) # This call will add it to ongoing_trials
                if new_trial_obj:
                    # The create_trial method already sets the status to IDLE and adds to ongoing_trials.
                    # Here we just need to ensure the returned trial is marked as RUNNING if it's being taken.
                    if new_trial_obj.status == trial_lib.TrialStatus.IDLE:
                        new_trial_obj.status = trial_lib.TrialStatus.RUNNING
                        # Ensure it's correctly set as the ongoing trial for this tuner
                        self.ongoing_trials[tuner_id] = new_trial_obj
                        self.save() # Save after status change
                        self.logger.info(f"[CustomOracle] Created and assigned new trial {new_trial_obj.trial_id} to tuner {tuner_id}.")
                        return {
                            "trial_id": new_trial_obj.trial_id,
                            "hyperparameters": new_trial_obj.hyperparameters.values,
                            "status": new_trial_obj.status,
                            "score": new_trial_obj.score
                        }
                    else: # If create_trial already returned it as RUNNING (e.g. if reloading from previous state), return as is
                        self.logger.info(f"[CustomOracle] Assigned already running trial {new_trial_obj.trial_id} to tuner {tuner_id}.")
                        return {
                            "trial_id": new_trial_obj.trial_id,
                            "hyperparameters": new_trial_obj.hyperparameters.values,
                            "status": new_trial_obj.status,
                            "score": new_trial_obj.score
                        }
                else:
                    self.logger.info(f"[CustomOracle] Failed to create a new trial. Max trials might be reached or an error occurred during creation.")
                    return None
            else:
                self.logger.info(f"[CustomOracle] Max trials ({self.max_trials}) reached. No new trials can be generated.")
                return None

    def update_trial(self, trial_id, metrics=None, step=0, status: Optional[str] = None):
        with self.lock:
            self.logger.info(f"[CustomOracle] Updating trial {trial_id} with metrics: {metrics}, step: {step}, status: {status}")
            trial = self.trials.get(trial_id)
            if not trial:
                self.logger.warning(f"[CustomOracle] Trial {trial_id} not found for update.")
                return

            if metrics:
                score = metrics.get(self.objective)
                if score is not None:
                    trial.score = score
                if not hasattr(trial, 'metrics_history'):
                    trial.metrics_history = {}
                for metric_name, metric_value in metrics.items():
                    if metric_name not in trial.metrics_history:
                        trial.metrics_history[metric_name] = []
                    trial.metrics_history[metric_name].append((step, metric_value))

            # If status is explicitly provided, update it. This handles the case
            # where OracleServer might incorrectly call this with a status.
            if status:
                trial.status = status
            else:
                # Otherwise, maintain RUNNING status for intermediate updates
                # unless it was already set to COMPLETED/FAILED externally.
                if trial.status not in [trial_lib.TrialStatus.COMPLETED, trial_lib.TrialStatus.FAILED]:
                    trial.status = trial_lib.TrialStatus.RUNNING 
            self.save()
            self.logger.info(f"[CustomOracle] Trial {trial_id} updated. Current score: {trial.score}, Status: {trial.status}")

    def report_trial_result(self, trial_id, score, status="COMPLETED"):
        with self.lock:
            self.logger.info(f"[CustomOracle] Reporting result for trial {trial_id}: score={score}, status={status}")
            trial = self.trials.get(trial_id)
            if not trial:
                self.logger.warning(f"[CustomOracle] Trial {trial_id} not found for result reporting.")
                return

            trial.score = score
            trial.status = status
            
            # Remove the trial from ongoing_trials using its tuner_id if found
            for tuner_id, ongoing_trial_obj in list(self.ongoing_trials.items()):
                if ongoing_trial_obj.trial_id == trial_id:
                    del self.ongoing_trials[tuner_id]
                    self.logger.debug(f"[CustomOracle] Removed trial {trial_id} from ongoing_trials for tuner {tuner_id}.")
                    break
            else:
                self.logger.warning(f"[CustomOracle] Trial {trial_id} not found in ongoing_trials when reporting result.")

            self.save()
            self.logger.info(f"[CustomOracle] Trial {trial_id} result reported. Final score: {trial.score}, Final Status: {trial.status}")

    def update_trial_status(self, trial_id, status="COMPLETED"):
        with self.lock:
            self.logger.info(f"[CustomOracle] Updating trial {trial_id} status to: {status}")
            trial = self.trials.get(trial_id)
            if not trial:
                self.logger.warning(f"[CustomOracle] Trial {trial_id} not found for status update.")
                return

            trial.status = status
            if status == trial_lib.TrialStatus.COMPLETED or status == trial_lib.TrialStatus.FAILED:
                # Remove the trial from ongoing_trials using its tuner_id if found
                for tuner_id, ongoing_trial_obj in list(self.ongoing_trials.items()):
                    if ongoing_trial_obj.trial_id == trial_id:
                        del self.ongoing_trials[tuner_id]
                        self.logger.debug(f"[CustomOracle] Removed trial {trial_id} from ongoing_trials for tuner {tuner_id}.")
                        break
                else:
                    self.logger.warning(f"[CustomOracle] Trial {trial_id} not found in ongoing_trials when updating status to {status}.")
            self.save()
            self.logger.info(f"[CustomOracle] Trial {trial_id} status updated to: {trial.status}")


    def new_trial_id(self):
        """Generates a new unique trial ID using UUID."""
        return str(uuid.uuid4())

    def save(self):
        """Saves the current state of the Oracle to disk."""
        # Ensure the internal KerasTuner attributes are set before calling super().save()
        if not hasattr(self, '_directory') or not hasattr(self, '_project_name'):
            self.logger.warning("Internal KerasTuner attributes _directory or _project_name missing. Attempting to set them from CustomOracle's init args.")
            self._directory = self.directory
            self._project_name = self.project_name

        if not hasattr(self, '_directory') or not Path(self._directory).exists():
            self.logger.error("Attempted to save Oracle state, but _directory is not set or does not exist.")
            return

        actual_save_dir = Path(self._directory) / self._project_name
        actual_save_dir.mkdir(parents=True, exist_ok=True)

        super().save()

        self._save_file = actual_save_dir / "oracle.json"
        self.logger.info(f"[CustomOracle] Oracle state saved to {self._save_file}.")


    def reload(self):
        """Reloads the Oracle state from disk."""
        if not hasattr(self, '_directory') or not hasattr(self, '_project_name'):
            self.logger.warning("Internal KerasTuner attributes _directory or _project_name missing during reload. Attempting to set them from CustomOracle's init args.")
            self._directory = self.directory
            self._project_name = self.project_name

        if not hasattr(self, '_directory') or not Path(self._directory).exists():
            self.logger.error("Attempted to reload Oracle state, but _directory is not set or does not exist.")
            return
        
        actual_load_dir = Path(self._directory) / self._project_name
        if not actual_load_dir.exists():
            self.logger.warning(f"Attempted to reload Oracle state, but directory {actual_load_dir} does not exist. Cannot reload.")
            return

        super().reload()
        self.logger.info(f"[CustomOracle] Oracle state reloaded from {actual_load_dir / 'oracle.json'}.")
