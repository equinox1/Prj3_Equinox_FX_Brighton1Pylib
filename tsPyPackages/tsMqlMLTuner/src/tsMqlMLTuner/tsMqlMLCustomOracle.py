# --- File: tsMqlMLCustomOracle.py ---
from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib

import logging
import os # Import os to access environment variables
from pathlib import Path # Import Path for directory manipulation

# -- Set up global logging --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging()  # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for CustomOracle. Using default logging.")

logger = logging.getLogger(__name__)
# -- end of logging setup ----


from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides()
app_params = mql_overrides.env.all_params().get("app", {})

# Use GLOBAL_LOGDIR_PATH for directory setting for consistency
global_logdir_from_env = os.environ.get('GLOBAL_LOGDIR_PATH')
# Fallback logic if environment variable is not set, though multiworker_launcher should set it.
if global_logdir_from_env:
    global_logdir = Path(global_logdir_from_env)
else:
    global_logdir = Path(app_params.get('LOGDIR', 'Logdir')) # Fallback to app_params if env var not set

global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

class CustomOracle(Oracle):
    def __init__(
        self,
        objective="val_loss",
        max_trials=50,
        directory="oracle_dir", # This will now be the path passed from oracle_server_main
        project_name="default_project",
        seed=42,
        reset_trials=True,
        **kwargs # Accept additional kwargs for potential future use
    ):
        # The base KerasTuner Oracle.__init__ does NOT take 'directory' or 'project_name'
        # directly as named arguments. Instead, these are often set as internal
        # attributes (prefixed with an underscore) by the calling Tuner or directly
        # by a custom Oracle subclass when it's instantiated independently.
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            **kwargs # Pass other arbitrary kwargs to super
        )
        self.logger = logger # Use the global logger
        self.reset_trials = reset_trials # Store reset_trials setting
        
        # Explicitly set the internal _directory and _project_name attributes
        # that the base KerasTuner Oracle expects for managing its files.
        self._directory = str(Path(directory).resolve()) # Resolve to absolute path
        self._project_name = project_name

        logger.info(f"CustomOracle initialized with objective: {objective}, max_trials: {max_trials}, directory: {self._directory}, project_name: {self._project_name}, reset_trials: {reset_trials}")

    def populate_space(self, trial_id):
        """
        Populates the hyperparameter space for a given trial.
        This method is called by the base Oracle's `create_trial` method.
        It should return a dictionary with 'status' and 'values' (hyperparameter values).
        """
        hp = HyperParameters()
        self.logger.info(f"[CustomOracle] Populating hyperparameters for trial_id: {trial_id}")

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
        
        # Return a dictionary as expected by the base Oracle's populate_space
        return {
            "status": trial_lib.TrialStatus.RUNNING, # Or "STOPPED" if max trials reached
            "values": hp.values
        }


    def create_trial(self, tuner_id):
        """Creates a new trial based on the populated space.
        This method now defers to the base Oracle's create_trial, which will
        internally call this CustomOracle's `populate_space` method.
        """
        try:
            # Let the base Oracle handle the trial creation and population logic.
            # It will internally call self.populate_space.
            new_trial = super().create_trial(tuner_id=tuner_id)
            if new_trial:
                self.logger.info(f"[CustomOracle] Created trial {new_trial.trial_id} with HPs: {new_trial.hyperparameters.values}")
                self.save() # Save the Oracle state after the base class creates a new trial
            return new_trial
        except Exception as e:
            self.logger.error(f"[CustomOracle] Error creating trial via super().create_trial: {e}", exc_info=True)
            return None


    def update_trial(self, trial_id, status, score=None, hyperparameters=None):
        """Updates the internal state of a trial, typically with its results."""
        # This method is called by KerasTuner internally (via `super().update_trial`).
        # We pass the update directly to the base Oracle.
        try:
            super().update_trial(trial_id, status, score, hyperparameters)
            self.logger.info(f"[CustomOracle] Updated trial {trial_id} with status: {status}, score: {score}")
            self.save() # Ensure state is saved after update
        except Exception as e:
            self.logger.error(f"[CustomOracle] Error updating trial via super().update_trial for trial {trial_id}: {e}", exc_info=True)


    def new_trial_id(self):
        """Generates a new unique trial ID. Overridden to use KerasTuner's default `_generate_id`."""
        return super()._generate_id()

    # REMOVED the @property decorator for 'trials' to prevent conflict with base Oracle
    # The base class already manages the 'trials' dictionary.
    # Access it directly via 'self.trials' if needed within CustomOracle.
    # For clarity, commenting out the property but leaving the methods that might use it.
    # @property
    # def trials(self):
    #     """Returns the dictionary of trials, directly from the base Oracle."""
    #     return super().trials

    def save(self):
        """Saves the current state of the Oracle to disk."""
        # KerasTuner's Oracle base class handles saving its state (e.g., `oracle.json`).
        super().save()
        self.logger.info(f"[CustomOracle] Oracle state saved to {self._directory}.") # Use _directory here

    def reload(self):
        """Reloads the Oracle state from disk."""
        super().reload()
        self.logger.info(f"[CustomOracle] Oracle state reloaded from {self._directory}.") # Use _directory here
