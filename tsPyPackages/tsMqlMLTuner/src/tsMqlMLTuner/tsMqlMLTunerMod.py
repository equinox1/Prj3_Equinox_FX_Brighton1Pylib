# filename: tsMqlMLTunerMod.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTuner.py
File: tsPyPackages/tsMqlMLTuner/tsMqlMLTuner.py
Description: The Tuner method for the machine learning model with performance optimizations.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.4.0
License: MIT License
"""

import os
import logging
import tensorflow as tf
from keras_tuner import Hyperband, RandomSearch, BayesianOptimization, Objective, HyperParameters
from tensorflow.keras import mixed_precision
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from pathlib import Path # Import Path
from typing import Optional, Dict, Any

# Import Keras Callbacks
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TerminateOnNaN, EarlyStopping, ReduceLROnPlateau

# Import OracleClient for the custom callback
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Set environment variables
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get("mltune", {})
base_params = all_params.get("base", {})

# Setup mixed precision
mixed_precision_policy = tune_params.get('mixed_precision_policy', 'mixed_float16')
mixed_precision.set_global_policy(mixed_precision_policy)
logger.info(f"TensorFlow Mixed Precision Policy set to: {mixed_precision_policy}")

# Ensure `LOGDIR` is defined and accessible. Assuming it's based on `mp_glob_base_log_path`
LOGDIR = Path(base_params.get('mp_glob_base_log_path', './Logdir'))

# Function to get callbacks
def get_callbacks(trial_id: str, tuner_type: str, log_dir: Path, app_params: Dict[str, Any]):
    model_checkpoint_path = log_dir / "checkpoints" / tuner_type / trial_id
    model_checkpoint_path.mkdir(parents=True, exist_ok=True)

    csv_log_path = log_dir / "csv_logs" / tuner_type
    csv_log_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_checkpoint_path / 'best_model.weights.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            verbose=0
        ),
        CSVLogger(
            filename=str(csv_log_path / f'training_log_{trial_id}.csv'),
            append=True
        ),
        TerminateOnNaN(),
        EarlyStopping(
            monitor='val_loss',
            patience=app_params.get('early_stopping_patience', 10),
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=app_params.get('reduce_lr_factor', 0.2),
            patience=app_params.get('reduce_lr_patience', 5),
            min_lr=app_params.get('min_lr', 1e-7),
            verbose=1
        )
    ]
    return callbacks

class CMdtuner:
    def __init__(self, tuner_id: str, oracle_client: OracleClient, is_chief: bool,
                 train_data, val_data, input_shape,
                 model_save_dir: Path,
                 hypermodel_params: Optional[Dict] = None, **kwargs):
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.is_chief = is_chief
        self.train_data = train_data
        self.val_data = val_data
        self.input_shape = input_shape
        self.model_save_dir = model_save_dir
        self.hypermodel_params = hypermodel_params if hypermodel_params is not None else {}

        self.max_epochs = tune_params.get('max_epochs', 50)
        self.objective_metric = tune_params.get('objective', 'val_loss')
        self.max_trials = tune_params.get('num_trials', 50) # This is the total number of trials the Oracle will generate
        self.executions_per_trial = tune_params.get('executions_per_trial', 1)
        self.directory = Path(tune_params.get('tuner_dir', LOGDIR / "keras_tuner_data")) / tuner_id
        self.project_name = tune_params.get('project_name', 'my_hp_tuning_project')
        self.overwrite = tune_params.get('overwrite', False)

        self.model = None # To store the best model after tuning

        logger.info(f"[CMdtuner] Tuner initialized: ID={self.tuner_id}, IsChief={self.is_chief}, LogDir={self.directory}")


    def _build_model(self, hp):
        """
        Hypermodel definition. This function is passed to KerasTuner.
        It defines the model architecture and hyperparameter search space.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=self.input_shape))

        # Input layer normalization
        if hp.Boolean("normalize_input"):
            model.add(tf.keras.layers.BatchNormalization())

        # Tune the number of hidden layers
        for i in range(hp.Int("num_layers", 1, 3)):
            model.add(tf.keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice(f"activation_{i}", ["relu", "tanh", "sigmoid"])
            ))
            if hp.Boolean(f"dropout_{i}"):
                model.add(tf.keras.layers.Dropout(hp.Float(f"dropout_rate_{i}", min_value=0.0, max_value=0.5, step=0.1)))

        model.add(tf.keras.layers.Dense(1, activation='linear')) # Output layer for regression

        # Tune the optimizer, learning rate
        optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'adagrad'])
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        if optimizer_choice == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else: # adagrad
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    def run(self):
        logger.info(f"[CMdtuner] Worker {self.tuner_id} starting run method. IsChief: {self.is_chief}")

        while True:
            # 1. Request a trial from the OracleServer
            trial_info = self.oracle_client.get_trial(self.tuner_id)

            if trial_info:
                trial_id = trial_info['trial_id']
                hyperparameters = trial_info['hyperparameters']
                logger.info(f"[CMdtuner] Worker {self.tuner_id} received trial {trial_id} with HP: {hyperparameters}")

                # Construct a HyperParameters object from the received dict
                hp = HyperParameters()
                for key, value in hyperparameters.items():
                    hp.values[key] = value

                try:
                    # Build and train the model with the provided HPs
                    model = self._build_model(hp)
                    callbacks = get_callbacks(trial_id, tune_params.get('tuner_type', 'hyperband'), LOGDIR, app_params)

                    # Append a custom callback to synchronize with our external OracleServer
                    oracle_sync_callback = OracleSyncCallback(
                        self.oracle_client,
                        trial_id,
                        self.objective_metric,
                        self.model_save_dir,
                        save_best_model=self.is_chief # Only chief saves the actual model
                    )
                    callbacks.append(oracle_sync_callback)

                    # Prepare data (assuming train_data and val_data are already tf.data.Dataset or numpy arrays)
                    x_train, y_train = self.train_data
                    x_val, y_val = self.val_data

                    logger.info(f"Fitting model for trial {trial_id} on worker {self.tuner_id}...")
                    history = model.fit(
                        x_train, y_train,
                        epochs=self.max_epochs,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks,
                        verbose=1 # Set verbose to 1 to see training progress
                    )

                    logger.info(f"[CMdtuner] Worker {self.tuner_id} completed trial {trial_id}.")

                except Exception as e:
                    logger.error(f"[CMdtuner] Worker {self.tuner_id} encountered error during trial {trial_id}: {e}", exc_info=True)
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
            else:
                logger.info(f"[CMdtuner] Worker {self.tuner_id} received no new trial from Oracle. Assuming all trials are processed or no more available.")
                break # Exit loop if no new trials are available

        logger.info("✅ Worker finished trial execution for TensorFlow.")

    def get_best_model(self) -> Optional[tf.keras.Model]:
        logger.info("[CMdtuner] Chief worker attempting to retrieve best model.")
        best_trial_info = self.oracle_client.get_best_trial()
        if not best_trial_info:
            logger.warning("No best trial found from Oracle. Cannot retrieve best model.")
            return None

        best_trial_id = best_trial_info.get('trial_id')
        best_hyperparameters = best_trial_info.get('hyperparameters', {})
        logger.debug(f"Best trial hyperparameters from Oracle for loading model: {best_hyperparameters}")

        # Re-build the hypermodel using the best hyperparameters
        hp_for_best_model = HyperParameters()
        for key, value in best_hyperparameters.items():
            hp_for_best_model.values[key] = value

        model = self._build_model(hp_for_best_model)

        # Define the path where the best model's weights would have been saved by the chief worker
        best_model_weights_path = self.model_save_dir / f"best_model_{best_trial_id}.weights.h5"

        if not best_model_weights_path.exists():
            logger.error(f"❌ Best model weights not found at {best_model_weights_path}. This might indicate an issue with saving or a worker not completing.")
            return None

        try:
            model.load_weights(str(best_model_weights_path))
            logger.info(f"✅ Best model for trial {best_trial_id} loaded successfully from {best_model_weights_path}.")
            self.model = model
            return self.model
        except Exception as e:
            logger.error(f"❌ Error loading best model weights from {best_model_weights_path}: {e}", exc_info=True)
            return None

    def get_model_dir(self):
        """Returns the directory where models are saved."""
        return self.model_save_dir

class OracleSyncCallback(tf.keras.callbacks.Callback):
    """
    A Keras Callback to synchronize trial results with the external OracleServer.
    """
    def __init__(self, oracle_client: OracleClient, trial_id: str, objective_metric: str, model_save_dir: Path, save_best_model: bool = False):
        super().__init__()
        self.oracle_client = oracle_client
        self.trial_id = trial_id
        self.objective_metric = objective_metric
        self.model_save_dir = model_save_dir
        self.save_best_model = save_best_model
        self.best_score = float('inf') if 'loss' in objective_metric else float('-inf') # Initialize based on objective
        self.best_epoch = 0
        logger.info(f"[OracleSyncCallback] Initialized for trial {trial_id}. Objective: {objective_metric}, Save Best Model: {save_best_model}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.objective_metric)
        if current_score is not None:
            # For 'loss' metrics, lower is better. For 'accuracy', higher is better.
            is_improvement = False
            if 'loss' in self.objective_metric:
                if current_score < self.best_score:
                    self.best_score = current_score
                    is_improvement = True
            else: # Assuming higher is better for other metrics like accuracy
                if current_score > self.best_score:
                    self.best_score = current_score
                    is_improvement = True

            if is_improvement and self.save_best_model:
                # Save the model weights using a naming convention that includes trial_id
                save_path = self.model_save_dir / f"best_model_{self.trial_id}.weights.h5"
                self.model.save_weights(save_path)
                logger.info(f"[OracleSyncCallback] Saved best model weights for trial {self.trial_id} at epoch {epoch} to {save_path} with {self.objective_metric}: {current_score:.4f}")

            # Optionally report intermediate results to the OracleServer
            # self.oracle_client.report_trial_result(self.trial_id, current_score, status="RUNNING")
        else:
            logger.warning(f"[OracleSyncCallback] Objective metric '{self.objective_metric}' not found in logs for trial {self.trial_id} at epoch {epoch}.")

    def on_train_end(self, logs=None):
        logs = logs or {}
        final_score = logs.get(self.objective_metric, self.best_score) # Use best score found if available
        status = "COMPLETED"
        if final_score is None:
            logger.warning(f"[OracleSyncCallback] Final objective metric '{self.objective_metric}' not found for trial {self.trial_id}. Reporting as FAILED or INCOMPLETE.")
            status = "FAILED" # Or "INCOMPLETE"
            final_score = -1.0 # Indicate an invalid score

        self.oracle_client.report_trial_result(self.trial_id, final_score, status=status)
        logger.info(f"[OracleSyncCallback] Reported final result for trial {self.trial_id}: Score={final_score}, Status={status}")
