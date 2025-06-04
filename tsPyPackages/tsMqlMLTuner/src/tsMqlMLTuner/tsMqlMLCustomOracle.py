# --- File: tsMqlMLCustomOracle.py ---
from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine import trial as trial_lib

import logging
import os

import logging
from keras_tuner.engine.oracle import Oracle


class CustomOracle(Oracle):
    def __init__(
        self,
        objective="val_loss",
        max_trials=50,
        directory="oracle_dir",
        project_name="default_project",
        seed=42,
        reset_trials=True,
    ):
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            # Removed 'directory' and 'project_name' from super().__init__()
            # as the base Oracle class does not accept them.
        )

        self.logger = logging.getLogger("CustomOracle")
        # You can keep these for your custom logging if needed
        self.directory = directory  # Store them as instance variables if you need them later
        self.project_name = project_name
        self.logger.info(f"[CustomOracle] Initialized with directory={directory}, project={project_name}")

        if reset_trials:
            self.logger.info("[CustomOracle] Resetting internal trial state.")
            self._trials = {}

    def save(self):
        self.logger.info("[CustomOracle] save() called — no-op.")


    def populate_space(self, trial_id):
        hp = HyperParameters()
        hp.Float("lr", 1e-4, 1e-2, sampling="log", default=1e-3)
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
        if len(self._trials) >= self.max_trials:
            self.logger.info("[CustomOracle] Max trials reached.")
            return None

        trial_id = f"{len(self._trials):02d}"
        hp = self.populate_space(trial_id)
        trial = trial_lib.Trial(
            hyperparameters=hp,
            trial_id=trial_id,
            status=trial_lib.TrialStatus.RUNNING,
        )
        self._trials[trial_id] = trial
        self.logger.info(f"[CustomOracle] Created trial {trial_id}: {hp.values}")
        return trial

    def score_trial(self, trial_id, result):
        if trial_id in self._trials:
            self._trials[trial_id].score = result
            self._trials[trial_id].status = trial_lib.TrialStatus.COMPLETED