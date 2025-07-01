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
            # Report intermediate results to Oracle
            metrics = {k: float(v) for k, v in logs.items()} # Ensure metrics are serializable
            self.oracle_client.update_trial(
                trial_id=self.trial_id,
                metrics=metrics,
                step=epoch,
                status="RUNNING"
            )
            self.logger.debug(f"Trial {self.trial_id} epoch {epoch+1}: Reported metrics to Oracle.")
            
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
            filepath=os.path.join(model_dir, 'checkpoint_epoch_{epoch:02d}.h5'),
            monitor=objective_name,
            save_best_only=False, # Save all checkpoints for debugging
            mode=direction,
            verbose=0 # Make verbose=0 for workers to reduce log spam
        ),
        CSVLogger(os.path.join(model_dir, 'training.log')),
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
    def __init__(self, input_shape, num_classes, hypermodel_params, tuner_type='hyperband', log_dir=None, **kwargs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hypermodel_params = hypermodel_params
        self.tune_params = hypermodel_params.get('mltune', {})
        self.app_params = hypermodel_params.get('app', {})
        self.log_dir = log_dir

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
        kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        dist_strat = kwargs.get('distribution_strategy', None)
        if isinstance(dist_strat, str):
            if dist_strat.lower() == 'mirrored':
                kwargs['distribution_strategy'] = tf.distribute.MirroredStrategy()
                logger.info("✅ Using MirroredStrategy for distribution.")
            else:
                logger.warning(f"⚠️ Invalid distribution_strategy '{dist_strat}' removed.")
                kwargs.pop('distribution_strategy', None)

        raw_obj = self.tune_params.get('objective', 'val_loss')
        if isinstance(raw_obj, str):
            objective = Objective(name=raw_obj, direction='min')
        else:
            objective = raw_obj

        super().__init__(
            objective=objective,
            hypermodel=self.build_model,
            max_epochs=self.tune_params.get('max_epochs', 50),
            factor=self.tune_params.get('factor', 3),
            hyperband_iterations=self.tune_params.get('hyperband_iterations', 1),
            directory=kwargs.pop('directory', 'kt_tuner_dir'),
            project_name=kwargs.pop('project_name', 'default_keras_tuner_project'),
            seed=self.tune_params.get('seed', 42),
            overwrite=kwargs.pop('overwrite', True),
            **kwargs
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
            weights_dir = os.path.join(self.directory, self.project_name, trial_id)
            checkpoints = glob.glob(os.path.join(weights_dir, "checkpoint_epoch_*.h5"))
            return max(checkpoints, key=os.path.getctime) if checkpoints else None
        return None

