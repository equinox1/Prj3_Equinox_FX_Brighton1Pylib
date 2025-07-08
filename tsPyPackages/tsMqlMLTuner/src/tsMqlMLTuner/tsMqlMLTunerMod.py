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
from keras_tuner import Hyperband, RandomSearch, BayesianOptimization, Objective
from tensorflow.keras import mixed_precision
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides
from pathlib import Path # Import Path

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
base_params = all_params.get("base", {}) # Get base_params

_logical_cores = os.cpu_count() or 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1
setup_config = CMqlSetup(
    loglevel=app_params.get('LOGLEVEL', 'INFO'),
    warn='ignore',
    precision=app_params.get('TF_PRECISION', 'mixed_float16'),
    tfdebug=app_params.get('TFDEBUG', False),
    num_cores=_estimated_physical_cores,
    num_threads=1
)

# Configure global mixed precision policy
tf_policy = mixed_precision.Policy(setup_config.precision)
mixed_precision.set_global_policy(tf_policy)
logger.info(f"✨ Global mixed precision policy set to: {mixed_precision.global_policy().compute_dtype}")


class OracleSyncCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras Callback to synchronize trial status and results with the Oracle Server.
    This callback is used by worker processes to report their progress and final results.
    """
    def __init__(self, trial_id, oracle_client, objective_name, direction, **kwargs):
        super().__init__(**kwargs)
        self.trial_id = trial_id
        self.oracle_client = oracle_client
        self.objective_name = objective_name
        self.direction = direction
        self.best_val_score = float('inf') if direction == 'min' else float('-inf')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"OracleSyncCallback initialized for trial {self.trial_id}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self.objective_name)
        if current_score is not None:
            # Report intermediate status to Oracle.
            # OracleClient's update_trial_status does not take metrics.
            self.oracle_client.update_trial_status(
                trial_id=self.trial_id,
                status="RUNNING"
            )
            self.logger.debug(f"Trial {self.trial_id} epoch {epoch+1}: Reported status to Oracle.")
            
            # Update best score internally for the callback's tracking
            if self.direction == 'min':
                if current_score < self.best_val_score:
                    self.best_val_score = current_score
            else: # max
                if current_score > self.best_val_score:
                    self.best_val_score = current_score
        else:
            self.logger.warning(f"Objective '{self.objective_name}' not found in logs for trial {self.trial_id} epoch {epoch+1}.")

    def on_train_end(self, logs=None):
        logs = logs or {}
        final_score = logs.get(self.objective_name)
        if final_score is None:
            # If objective not found in final logs, try to use the best score recorded
            final_score = self.best_val_score if self.best_val_score != float('inf') and self.best_val_score != float('-inf') else None
            if final_score is None:
                self.logger.error(f"Trial {self.trial_id}: Final objective '{self.objective_name}' not found in logs and no best score recorded. Reporting FAILED.")
                self.oracle_client.update_trial_status(self.trial_id, status="FAILED")
                return

        # Report final result and mark as COMPLETED
        # Use report_trial_result which takes score and status
        self.oracle_client.report_trial_result(self.trial_id, float(final_score), status="COMPLETED")
        self.logger.info(f"Trial {self.trial_id} finished. Final score: {final_score:.4f}. Status: COMPLETED.")


def get_callbacks(hp, model_dir, trial_id, oracle_client, objective_name, direction):
    """
    Returns a list of Keras callbacks for a given trial.
    Args:
        hp: KerasTuner HyperParameters object for the current trial.
        model_dir (str): Directory where model checkpoints and logs will be saved.
        trial_id (str): Unique ID of the current trial.
        oracle_client (OracleClient): Instance of OracleClient to communicate with the Oracle Server.
        objective_name (str): The name of the objective metric (e.g., 'val_loss').
        direction (str): 'min' or 'max' for the objective.
    Returns:
        list: A list of Keras Callback instances.
    """
    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'trial_{trial_id}_checkpoint_epoch_{{epoch:02d}}.h5'), # Include trial_id in filename
            monitor=objective_name,
            save_best_only=False, # Save all checkpoints for debugging
            mode=direction,
            verbose=0 # Make verbose=0 for workers to reduce log spam
        ),
        CSVLogger(os.path.join(model_dir, f'trial_{trial_id}_training.log')), # Include trial_id in filename
        # TensorBoard(log_dir=os.path.join(model_dir, 'tensorboard_logs'), update_freq='epoch') # Optional
    ]

    # Add OracleSyncCallback for workers to report progress
    if oracle_client:
        callbacks.append(OracleSyncCallback(
            trial_id=trial_id,
            oracle_client=oracle_client,
            objective_name=objective_name,
            direction=direction
        ))
        logger.info(f"Added OracleSyncCallback for trial {trial_id}.")

    # Optional: EarlyStopping and ReduceLROnPlateau can be added based on hp choices
    # if hp.Boolean('use_early_stopping'):
    #     callbacks.append(EarlyStopping(
    #         monitor=objective_name,
    #         patience=hp.Int('es_patience', min_value=5, max_value=20, step=5, default=10),
    #         mode=direction,
    #         restore_best_weights=True,
    #         verbose=1
    #     ))
    # if hp.Boolean('use_reduce_lr_on_plateau'):
    #     callbacks.append(ReduceLROnPlateau(
    #         monitor=objective_name,
    #         factor=hp.Float('lr_factor', min_value=0.2, max_value=0.5, step=0.1, default=0.2),
    #         patience=hp.Int('lr_patience', min_value=3, max_value=10, step=2, default=5),
    #         mode=direction,
    #         min_lr=hp.Float('min_lr', min_value=1e-7, max_value=1e-5, sampling='LOG', default=1e-6),
    #         verbose=1
    #     ))

    return callbacks


class CMdtuner(Hyperband):
    def __init__(self, input_shape, num_classes, hypermodel_params, tuner_type='hyperband', tuner_id=None, oracle_client=None, is_chief=True, **kwargs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hypermodel_params = hypermodel_params
        self.tune_params = hypermodel_params.get('mltune', {})
        self.app_params = hypermodel_params.get('app', {})
        
        self.tuner_id = tuner_id
        self.oracle_client = oracle_client
        self.is_chief = is_chief

        # Store data loaders for use in run()
        self.train_data = kwargs.pop('train_dataset', None)
        self.val_data = kwargs.pop('val_dataset', None)

        # Get model_save_dir from kwargs, default to a path within base_path if not provided
        base_path = base_params.get('mp_glob_base_log_path') # Get from base_params
        model_id = kwargs.get("model_id", "tsneuromodel_1")
        self.model_dir = kwargs.get("model_save_dir", Path(base_path) / model_id / "keras_models")
        
        # Ensure the directory exists
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

        # Select tuner
        tuner_map = {
            'hyperband': Hyperband,
            'randomsearch': RandomSearch,
            'bayesian_optimization': BayesianOptimization
        }
        tuner_class = tuner_map.get(tuner_type.lower(), Hyperband)

        allowed_keys = {
            'objective', 'max_epochs', 'factor', 'hyperband_iterations',
            'directory', 'project_name', 'seed', 'overwrite',
            'executions_per_trial', 'distribution_strategy', 'tune_new_entries',
            'allow_new_entries'
        }
        # Filter kwargs to only include those relevant for the base KerasTuner.__init__
        kt_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        dist_strat = kt_kwargs.get('distribution_strategy', None)
        if isinstance(dist_strat, str):
            if dist_strat.lower() == 'mirrored':
                kt_kwargs['distribution_strategy'] = tf.distribute.MirroredStrategy()
                logger.info("✅ Using MirroredStrategy for distribution.")
            else:
                logger.warning(f"⚠️ Invalid distribution_strategy '{dist_strat}' removed.")
                kt_kwargs.pop('distribution_strategy', None)

        raw_obj = self.tune_params.get('objective', 'val_loss')
        if isinstance(raw_obj, str):
            objective = Objective(name=raw_obj, direction='min')
        else:
            objective = raw_obj

        # Use the correct directory for KerasTuner's internal files
        kt_directory = Path(base_path) / "keras_tuner_projects"
        kt_directory.mkdir(parents=True, exist_ok=True) # Ensure it exists

        super().__init__(
            objective=objective,
            hypermodel=self.build_model,
            max_epochs=self.tune_params.get('max_epochs', 50),
            factor=self.tune_params.get('factor', 3),
            hyperband_iterations=self.tune_params.get('hyperband_iterations', 1),
            directory=str(kt_directory), # Use the unified base path
            project_name=kt_kwargs.pop('project_name', 'default_keras_tuner_project'),
            seed=self.tune_params.get('seed', 42),
            overwrite=kt_kwargs.pop('overwrite', True),
            **kt_kwargs # Pass filtered kwargs to super()
        )

        logger.info(f"🛠 Initialized {tuner_type} tuner with objective: {objective.name if hasattr(objective, 'name') else objective}")

    def build_model(self, hp):
        logger.info(f"🔧 Building model with shape={self.input_shape}, classes={self.num_classes}")
        from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.models import Model
        from tensorflow.keras.optimizers import Adam, Nadam
        from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy, BinaryCrossentropy
        from tensorflow.keras.metrics import MeanAbsoluteError as MAE, RootMeanSquaredError as RMSE

        inputs = Input(shape=self.input_shape)
        units = hp.Int("units", 32, 128, step=32)
        dropout = hp.Float("dropout", 0.0, 0.5, step=0.1)
        x = Bidirectional(LSTM(units=units, return_sequences=False))(inputs)
        x = Dropout(dropout)(x)

        activation = 'linear'
        if self.app_params.get("mp_app_task_type") == 'classification':
            activation = 'sigmoid' if self.num_classes == 1 else 'softmax'

        outputs = Dense(self.num_classes, activation=activation)(x)
        model = Model(inputs, outputs)

        optimizer_choice = hp.Choice("optimizer", ["adam", "nadam"], default="adam")
        lr = hp.Float("learning_rate", 1e-5, 1e-2, sampling='LOG')
        optimizer = Adam(lr) if optimizer_choice == 'adam' else Nadam(lr)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        loss_name = self.app_params.get("mp_app_loss_function", "mean_squared_error")
        loss = {
            'mean_absolute_error': MeanAbsoluteError(),
            'categorical_crossentropy': CategoricalCrossentropy(),
            'binary_crossentropy': BinaryCrossentropy()
        }.get(loss_name, MeanSquaredError())

        metrics = [MAE(), RMSE()] if self.app_params.get("mp_app_task_type") != 'classification' else []
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def finalize_best_trial(self):
        try:
            self.best_model = self.get_best_models(1)[0]
            return self.best_model
        except Exception as e:
            logger.error(f"❌ Error finalizing best trial: {e}", exc_info=True)
            return None

    def get_best_weights_path(self):
        import glob
        trials = self.oracle.get_best_trials(1)
        if trials:
            trial_id = trials[0].trial_id
            # Use the correct base directory for weights
            weights_dir = Path(self.directory) / self.project_name / trial_id
            checkpoints = glob.glob(os.path.join(weights_dir, "checkpoint_epoch_*.h5"))
            return max(checkpoints, key=os.path.getctime) if checkpoints else None
        return None

    def run(self):
        """
        Main method for CMdtuner (TensorFlow/Keras) to run the tuning process.
        Handles both chief and worker logic.
        """
        if self.is_chief:
            self._run_chief()
        else:
            self._run_worker()

    def _run_chief(self):
        logger.info("🚀 Chief starting distributed tuning for TensorFlow...")
        # Chief's responsibility: orchestrate trials, manage Oracle
        for i in range(self.num_trials): # self.num_trials comes from tune_params via super()
            logger.info(f"[CMdtuner] Chief requesting trial {i+1}...")
            trial_data = self.oracle.get_trial(self.tuner_id) # Use self.oracle.get_trial for KerasTuner's internal Oracle

            if trial_data and trial_data.get('trial_id'):
                trial_id = trial_data['trial_id']
                hyperparameters = trial_data['hyperparameters']
                logger.info(f"[CMdtuner] Chief received trial {trial_id} with hyperparameters: {hyperparameters}")
                # Chief does not train, it just manages the Oracle.
                # The actual training is done by workers.
                pass
            else:
                logger.info("[CMdtuner] Chief received no new trial. All trials might be completed or no idle trials.")
                break
        logger.info("✅ Chief finished tuning for TensorFlow.")


    def _run_worker(self):
        logger.info("👷 Worker starting trial execution loop for TensorFlow...")
        while True:
            # Worker requests a trial from the Oracle
            # Use the oracle_client to get a trial from the remote Oracle Server
            trial_data = self.oracle_client.get_trial(self.tuner_id)

            if trial_data and trial_data.get('trial_id'):
                trial_id = trial_data['trial_id']
                hyperparameters = trial_data['hyperparameters']
                logger.info(f"[CMdtuner] Worker {self.tuner_id} received trial {trial_id} with hyperparameters: {hyperparameters}")

                try:
                    # Prepare callbacks for the current trial
                    model_dir_for_callbacks = Path(self.directory) / self.project_name / trial_id
                    model_dir_for_callbacks.mkdir(parents=True, exist_ok=True)
                    
                    objective_name = self.objective.name if hasattr(self.objective, 'name') else str(self.objective)
                    objective_direction = self.objective.direction if hasattr(self.objective, 'direction') else 'min'

                    callbacks = get_callbacks(
                        hp=None, # hp is not directly used by get_callbacks for trial-specific HPs
                        model_dir=str(model_dir_for_callbacks),
                        trial_id=trial_id,
                        oracle_client=self.oracle_client,
                        objective_name=objective_name,
                        direction=objective_direction
                    )

                    # Train the model using tuner.search()
                    # The `search` method of KerasTuner's Hyperband class implicitly manages the trial
                    # and reports results to its internal Oracle. The OracleSyncCallback then
                    # synchronizes this with the external OracleServer.
                    self.search(
                        x=self.train_data,
                        epochs=self.max_epochs,
                        validation_data=self.val_data,
                        callbacks=callbacks,
                        # The KerasTuner `search` method will use the current trial context.
                        # No need to explicitly pass `trial_id` here, as the callbacks handle it.
                    )
                    
                    # The OracleSyncCallback.on_train_end should handle reporting the final result
                    # and status to the external OracleServer.
                    logger.info(f"[CMdtuner] Worker {self.tuner_id} completed trial {trial_id}.")

                except Exception as e:
                    logger.error(f"[CMdtuner] Worker {self.tuner_id} encountered error during trial {trial_id}: {e}", exc_info=True)
                    self.oracle_client.update_trial_status(trial_id, status="FAILED")
            else:
                logger.info(f"[CMdtuner] Worker {self.tuner_id} received no new trial from Oracle. Assuming all trials are processed or no more available.")
                break # Exit loop if no new trials are available

        logger.info("✅ Worker finished trial execution for TensorFlow.")
