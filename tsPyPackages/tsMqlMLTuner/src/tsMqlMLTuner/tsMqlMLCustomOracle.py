# --- File: tsMqlMLCustomOracle.py ---
from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib
from keras_tuner.engine.trial import TrialStatus # Import TrialStatus for consistency

import logging # Ensure logging is imported
import os # Import os to access environment variables

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for CustomOracle. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
# -- end of logging setup ----

from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile') # This global_logfile is superseded by env variable

class CustomOracle(Oracle):
    def __init__(
        self,
        objective="val_loss",
        max_trials=50,
        directory="oracle_dir",
        project_name="default_project",
        seed=42,
        **kwargs
    ):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            seed=seed,
            **kwargs
        )
        self.logger = logging.getLogger(self.__class__.__name__) # Logger for the Oracle instance
        self.logger.info(f"[CustomOracle] Initialized with max_trials={max_trials}, project='{project_name}'")
        self.logger.debug(f"Oracle directory: {directory}")

        # Internal dictionary to store trials (can be improved with persistent storage if needed)
        self._trials = {}
        # Load existing trials if they exist in the directory (keras_tuner handles this via super().__init__)
        # However, for custom serialization, you might need custom load/save logic.
        self.load() # Attempt to load trials from disk

    def populate_space(self, trial_id):
        # This method is called by KerasTuner to get hyperparameters for a new trial.
        # We define the search space here.
        hp = HyperParameters()
        self._define_hyperparameters(hp)
        return hp

    def _define_hyperparameters(self, hp):
        # Define your hyperparameter search space here
        hp.Int("num_layers", 1, 3, default=2)
        hp.Int("units", 32, 128, step=32, default=64)
        hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-3)
        hp.Choice("optimizer", ["Adam", "RMSprop", "SGD"], default="Adam")
        hp.Int("epochs", 5, 20, step=5, default=10)
        hp.Int("n_units1", 64, 256, step=64, default=128)
        hp.Int("n_units2", 64, 256, step=64, default=128)
        hp.Int("lstm_units", 32, 128, step=32, default=64)
        hp.Int("cnn_filters", 16, 128, step=16, default=32)
        hp.Int("cnn_kernel_size", 2, 5, step=1, default=3)
        hp.Int("trans_heads", 2, 8, step=2, default=4)
        hp.Int("trans_ff_dim", 64, 512, step=64, default=128)
        hp.Choice("loss", ["mse", "mae", "binary_crossentropy"], default="mse")
        hp.Choice("metric", ["mse", "mae", "accuracy"], default="mse")
        return hp

    def create_trial(self, tuner_id):
        # This method is called by the client to request a new trial.
        # It creates a new Trial object with proposed hyperparameters.
        if len(self._trials) >= self.max_trials:
            self.logger.info("[CustomOracle] Max trials reached. No new trial created.")
            return None

        # Delegate to KerasTuner's internal trial generation mechanism
        # This will call populate_space internally to get HPs
        trial = super().create_trial(tuner_id)
        if trial:
            self.logger.info(f"[CustomOracle] Created new trial_id: {trial.trial_id} for tuner: {tuner_id}")
            self._trials[trial.trial_id] = trial # Store internally
            self.save() # Save the updated state
        else:
            self.logger.warning("[CustomOracle] Failed to create a trial via KerasTuner super method.")
        return trial

    def update_trial(self, trial_id, score, hyperparameters):
        # This method is called by the client to report results for a trial.
        # It updates the internal state of the oracle.
        self.logger.info(f"[CustomOracle] Updating trial {trial_id} with score {score}.")
        if trial_id in self._trials:
            trial = self._trials[trial_id]
            trial.score = score
            trial.status = TrialStatus.COMPLETED # Mark as completed
            # This is where KerasTuner's Oracle.update_trial is usually called
            # However, our server directly modifies `_trials` and then saves.
            # If `super().update_trial` is used, it handles the `self.trials` dictionary for us.
            # For simplicity with direct _trials manipulation in OracleServer, we'll keep this custom update.
            self.save() # Save changes
            self.logger.debug(f"Trial {trial_id} updated and saved.")
        else:
            self.logger.warning(f"[CustomOracle] Attempted to update non-existent trial: {trial_id}")

    def get_best_trials(self, num_trials=1):
        # Returns the best trials based on the objective.
        self.logger.info(f"[CustomOracle] Retrieving top {num_trials} best trials.")
        # Filter out trials without a score or not completed
        scored_trials = [t for t in self._trials.values() if hasattr(t, 'score') and t.score is not None and t.status == TrialStatus.COMPLETED]
        if not scored_trials:
            self.logger.info("[CustomOracle] No completed trials with scores available.")
            return []

        # Sort trials by score (assuming lower score is better for val_loss objective)
        sorted_trials = sorted(scored_trials, key=lambda t: t.score)
        best_n_trials = sorted_trials[:num_trials]
        self.logger.debug(f"[CustomOracle] Found {len(best_n_trials)} best trials.")
        return best_n_trials

    def save(self):
        # Implement custom saving if needed beyond KerasTuner's default.
        # KerasTuner's Oracle.save() typically handles saving `self.trials`.
        super().save()
        self.logger.info("[CustomOracle] Oracle state saved.")

    def load(self):
        # Implement custom loading if needed beyond KerasTuner's default.
        # KerasTuner's Oracle.reload() or internal loading handles loading `self.trials`.
        # For simplicity, we assume super().load() does most of the work.
        try:
            super().reload() # This reloads trials into self.trials
            # After reloading, populate our internal _trials dictionary from super's trials
            self._trials = {trial.trial_id: trial for trial in self.trials}
            self.logger.info(f"[CustomOracle] Oracle state loaded. Loaded {len(self._trials)} trials.")
        except Exception as e:
            self.logger.warning(f"[CustomOracle] Could not load previous oracle state: {e}. Starting fresh.")
            self._trials = {} # Initialize empty if loading fails
