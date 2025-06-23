#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTuner.py
File: tsPyPackages/tsMqlMLTuner/tsMqlMLTuner.py
Description: The Tuner method for the machine learning model with performance optimizations.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.3.1 (Optimized for reduced memory usage and increased execution performance)
License: MIT License
"""

import logging # Ensure logging is imported
import os # Import os to access environment variables

# --- Logging setup ---
logger = logging.getLogger(__name__)
# -- end of logging setup ----


import os
import pathlib
import uuid  # Ensure uuid is imported for use in get_callbacks
# Machine Learning packages
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ✅ Determine backend from environment
backend = os.environ.get('MLTUNE_BACKEND', 'tensorflow').lower()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"🔧 Detected tuning backend from environment: {backend}")


import tensorflow as tf
from datetime import date
import numpy as np
from keras_tuner.engine.hyperparameters import HyperParameters
# CORRECTED: Changed
from keras_tuner import Hyperband, RandomSearch, BayesianOptimization # Import tuner types
from keras_tuner.src.engine.tuner import Tuner # Import base Tuner
from keras_tuner.engine.oracle import Oracle # For CustomOracle dependency if needed
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Attention, MultiHeadAttention, LayerNormalization, Embedding, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint, CSVLogger, TensorBoard
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.initializers import GlorotUniform

# Import from tsMqlSetup for configuration access
from tsMqlSetup import CMqlSetup
from tsMqlOverrides import CMqlOverrides

# Ensure setup_config is initialized for this module if needed for shared configs
# This needs to be done carefully to avoid multiple initializations if it's already done by launcher
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})

# Initialize CMqlSetup (ensure it's not re-initializing logging if already done by launcher)
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel=app_params.get('LOGLEVEL', 'INFO'),
    warn='ignore',
    precision=app_params.get('TF_PRECISION', 'mixed_float16'),
    tfdebug=app_params.get('TFDEBUG', False),
    num_cores=_estimated_physical_cores,
    num_threads=1 # Can adjust if needed
)

# Set global mixed precision policy
policy = mixed_precision.Policy(setup_config.precision)
mixed_precision.set_global_policy(policy)
logger.info(f"✨ Global mixed precision policy in TunerMod set to: {mixed_precision.global_policy().compute_dtype}")

# Custom Positional Encoding Layer (as provided in original snippet)
class AddPositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(self.dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.dim, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, axis=0)  # (1, seq_len, dim)
        return x + tf.cast(pos_encoding, x.dtype)


# Define CMdtuner class
class CMdtuner(Hyperband): # Inherit from Hyperband or RandomSearch based on tuner_type
    def __init__(self, input_shape, num_classes, hypermodel_params, tuner_type='hyperband', **kwargs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hypermodel_params = hypermodel_params # Store all_params for hypermodel config
        self.tune_params = hypermodel_params.get('mltune', {})
        self.app_params = hypermodel_params.get('app', {})

        # Determine which base tuner to use
        tuner_class = Hyperband if tuner_type == 'hyperband' else RandomSearch
        # If BayesianOptimization is desired and configured
        if tuner_type == 'bayesian_optimization':
            tuner_class = BayesianOptimization

        super().__init__(
            hypermodel=self.build_model,
            objective=self.tune_params.get('objective', 'val_loss'), # Default objective
            max_epochs=self.tune_params.get('max_epochs', 50),
            factor=self.tune_params.get('factor', 3), # For Hyperband
            hyperband_iterations=self.tune_params.get('hyperband_iterations', 1), # For Hyperband
            directory=kwargs.pop('directory', 'kt_tuner_dir'), # Directory where models and logs are stored
            project_name=kwargs.pop('project_name', 'default_keras_tuner_project'),
            seed=self.tune_params.get('seed', 42),
            overwrite=kwargs.pop('overwrite', True), # Pass overwrite from selector
            **kwargs # Pass remaining kwargs to the base tuner
        )
        
        # Defensive check for objective initialization
        objective_name_for_log = "Unknown Objective"
        if hasattr(self, 'objective') and self.objective is not None:
            if hasattr(self.objective, 'name'):
                objective_name_for_log = self.objective.name
            else:
                # If objective is a string (e.g., 'val_loss'), it won't have a .name attribute
                objective_name_for_log = str(self.objective)
        else:
            logger.warning("Keras Tuner objective not fully initialized yet in CMdtuner.__init__. Falling back to default.")
            objective_name_for_log = self.tune_params.get('objective', 'val_loss') # Fallback to the one passed to super()

        logger.info(f"Initialized Keras Tuner: {tuner_type} with objective: {objective_name_for_log}")


    def build_model(self, hp):
        # Dynamically determine the input shape and number of classes from tuner init
        input_shape = self.input_shape
        num_classes = self.num_classes
        logger.info(f"Building model with input_shape: {input_shape} and num_classes: {num_classes}")
        
        # Ensure input_shape is a tuple (timesteps, features)
        if not isinstance(input_shape, tuple) or len(input_shape) < 1:
            logger.error(f"Invalid input_shape provided: {input_shape}. Expected (timesteps, features) or similar.")
            # Default to a safe input shape or raise an error
            input_shape = (5, 1) # Example default if critical
            # raise ValueError("Invalid input_shape provided to build_model.")

        # Determine the number of features based on input_shape
        if len(input_shape) == 2:
            timesteps, num_features = input_shape
        elif len(input_shape) == 1:
            timesteps = input_shape[0]
            num_features = 1 # Assuming single feature if only one dimension
        else:
            logger.warning(f"Unexpected input_shape format: {input_shape}. Assuming last dimension is features.")
            timesteps = input_shape[0] if input_shape else None
            num_features = input_shape[-1] if input_shape else None
            if timesteps is None or num_features is None:
                raise ValueError(f"Could not infer timesteps and features from input_shape: {input_shape}")

        logger.info(f"Derived timesteps: {timesteps}, num_features: {num_features}")

        model_type = self.app_params.get('mp_app_model_type', 'LSTM').lower()

        inputs = Input(shape=(timesteps, num_features), dtype=tf.float32)
        x = inputs

        # Apply Positional Encoding if enabled and model type is suitable
        if self.tune_params.get('enable_positional_encoding', False):
            if model_type in ['transformer', 'lstm', 'gru']:
                x = AddPositionalEncoding(dim=num_features)(x)
                logger.info("Added Positional Encoding layer.")
            else:
                logger.warning("Positional Encoding requested but model type is not suitable (only for transformer, LSTM, GRU). Skipping.")

        # LSTM/GRU Layers
        num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=self.tune_params.get('max_lstm_layers', 3), default=1)
        
        for i in range(num_lstm_layers):
            # Corrected: Fetch min/max/step values from tune_params, not hp.get()
            lstm_units = hp.Int(f'lstm_units_{i}', 
                                min_value=self.tune_params.get('min_lstm_units', 32), 
                                max_value=self.tune_params.get('max_lstm_units', 256), 
                                step=self.tune_params.get('lstm_unit_step', 32), 
                                default=64)
            dropout_rate = hp.Float(f'dropout_rate_{i}', 
                                    min_value=self.tune_params.get('min_dropout_rate', 0.0), 
                                    max_value=self.tune_params.get('max_dropout_rate', 0.5), 
                                    step=self.tune_params.get('dropout_step', 0.1), 
                                    default=0.2)
            recurrent_dropout_rate = hp.Float(f'recurrent_dropout_rate_{i}', 
                                            min_value=self.tune_params.get('min_recurrent_dropout_rate', 0.0), 
                                            max_value=self.tune_params.get('max_recurrent_dropout_rate', 0.2), 
                                            step=self.tune_params.get('recurrent_dropout_step', 0.05), 
                                            default=0.0)
            
            return_sequences = (i < num_lstm_layers - 1) or (model_type == 'transformer') # Only return sequences if not last layer or if it's a transformer

            if model_type == 'lstm':
                x = Bidirectional(LSTM(
                    units=lstm_units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_initializer=GlorotUniform(),
                    recurrent_initializer=GlorotUniform(),
                    kernel_regularizer=l1_l2(l1=hp.Float(f'l1_reg_{i}', min_value=1e-7, max_value=1e-3, sampling='LOG', default=1e-5),
                                            l2=hp.Float(f'l2_reg_{i}', min_value=1e-7, max_value=1e-3, sampling='LOG', default=1e-5)),
                    kernel_constraint=MaxNorm(hp.Float(f'max_norm_kernel_{i}', min_value=1.0, max_value=5.0, default=3.0)),
                    recurrent_constraint=MaxNorm(hp.Float(f'max_norm_recurrent_{i}', min_value=1.0, max_value=5.0, default=3.0))
                ))(x)
                logger.info(f"Added Bidirectional LSTM layer {i+1} with {lstm_units} units.")
            elif model_type == 'gru':
                x = Bidirectional(tf.keras.layers.GRU(
                    units=lstm_units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_initializer=GlorotUniform(),
                    recurrent_initializer=GlorotUniform(),
                    kernel_regularizer=l1_l2(l1=hp.Float(f'l1_reg_{i}', min_value=1e-7, max_value=1e-3, sampling='LOG', default=1e-5),
                                            l2=hp.Float(f'l2_reg_{i}', min_value=1e-7, max_value=1e-3, sampling='LOG', default=1e-5)),
                    kernel_constraint=MaxNorm(hp.Float(f'max_norm_kernel_{i}', min_value=1.0, max_value=5.0, default=3.0)),
                    recurrent_constraint=MaxNorm(hp.Float(f'max_norm_recurrent_{i}', min_value=1.0, max_value=5.0, default=3.0))
                ))(x)
                logger.info(f"Added Bidirectional GRU layer {i+1} with {lstm_units} units.")
            elif model_type == 'transformer':
                # Transformer Encoder Block (Simplified for brevity)
                # MultiHeadAttention
                num_heads = hp.Int(f'num_heads_{i}', min_value=1, max_value=4, default=2) # Ensure num_heads divides lstm_units
                key_dim = lstm_units // num_heads if lstm_units % num_heads == 0 else lstm_units
                x = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)(x, x)
                x = Dropout(dropout_rate)(x)
                x = LayerNormalization(epsilon=1e-6)(x)
                
                # Point-wise Feed-forward Network
                ffn_units = hp.Int(f'ffn_units_{i}', min_value=lstm_units, max_value=lstm_units * 2, step=32, default=lstm_units)
                x = Dense(ffn_units, activation="relu")(x)
                x = Dense(lstm_units)(x) # Output dim matches input dim for residual connection
                x = Dropout(dropout_rate)(x)
                x = LayerNormalization(epsilon=1e-6)(x)
                logger.info(f"Added Transformer Encoder Block {i+1} with {lstm_units} units.")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            x = Dropout(dropout_rate)(x) # Additional dropout after each layer

        # Global pooling for sequence models if not already handled by return_sequences=False
        if return_sequences: # If the last layer returned sequences, we need to pool
             x = GlobalAveragePooling1D()(x) # Or GlobalMaxPooling1D()

        # Dense Layers
        num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=self.tune_params.get('max_dense_layers', 2), default=1)
        for i in range(num_dense_layers):
            # Corrected: Fetch min/max/step values from tune_params, not hp.get()
            dense_units = hp.Int(f'dense_units_{i}', 
                                 min_value=self.tune_params.get('min_dense_units', 32), 
                                 max_value=self.tune_params.get('max_dense_units', 128), 
                                 step=self.tune_params.get('dense_unit_step', 32), 
                                 default=64)
            x = Dense(dense_units, activation=hp.Choice(f'dense_activation_{i}', values=['relu', 'tanh', 'leaky_relu'], default='relu'))(x)
            x = Dropout(hp.Float(f'dense_dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1, default=0.2))(x)

        # Output Layer
        # The activation function depends on the prediction task (regression/classification)
        # Assuming regression (e.g., price prediction), use linear activation.
        # If it's a classification task, num_classes would define the output shape and activation.
        output_activation = 'linear' # Default for regression
        if self.app_params.get('mp_app_task_type', 'regression') == 'classification':
            if num_classes > 2:
                output_activation = 'softmax' # Multi-class classification
            else:
                output_activation = 'sigmoid' # Binary classification
        
        outputs = Dense(num_classes, activation=output_activation, name='output_layer')(x)

        model = Model(inputs=inputs, outputs=outputs, name=self.app_params.get('ml_model_name', 'tsneuromodel'))

        # Optimizer
        optimizer_type = hp.Choice('optimizer', values=['adam', 'nadam'], default='adam')
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)

        if optimizer_type == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_type == 'nadam':
            optimizer = Nadam(learning_rate=learning_rate)
        
        # Wrap optimizer with mixed precision policy if applicable
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        # Loss Function
        loss_function = self.app_params.get('mp_app_loss_function', 'mean_squared_error')
        if loss_function == 'mean_squared_error':
            loss = tf.keras.losses.MeanSquaredError()
        elif loss_function == 'mean_absolute_error':
            loss = tf.keras.losses.MeanAbsoluteError()
        elif loss_function == 'categorical_crossentropy' and self.app_params.get('mp_app_task_type') == 'classification':
            loss = tf.keras.losses.CategoricalCrossentropy()
        elif loss_function == 'binary_crossentropy' and self.app_params.get('mp_app_task_type') == 'classification' and num_classes == 1:
            loss = tf.keras.losses.BinaryCrossentropy()
        else:
            logger.warning(f"Unsupported loss function: {loss_function}. Defaulting to Mean Squared Error.")
            loss = tf.keras.losses.MeanSquaredError()

        # Metrics
        metrics = []
        task_type = self.app_params.get('mp_app_task_type', 'regression')
        if task_type == 'regression':
            metrics.append(MeanAbsoluteError(name='mae'))
            metrics.append(RootMeanSquaredError(name='rmse'))
        elif task_type == 'classification':
            metrics.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
            metrics.append(tf.keras.metrics.Precision(name='precision'))
            metrics.append(tf.keras.metrics.Recall(name='recall'))
        
        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info(f"Model compiled with {optimizer_type} optimizer, {loss_function} loss, and metrics: {[m.name for m in metrics]}")
        model.summary(print_fn=logger.info)
        return model

    # No explicit `fit` method override here.
    # Keras Tuner's base `Tuner` or `Hyperband` classes will call `hypermodel.fit`,
    # which in turn calls `model.fit`. Since build_model correctly returns a Keras model,
    # the default Keras Tuner fitting process should work with tf.data.Dataset.

    def finalize_best_trial(self):
        try:
            best_trials = self.oracle.get_best_trials(num_trials=1) # Get the single best trial
            if best_trials:
                best_trial = best_trials[0]
                best_hp_values = best_trial.hyperparameters.values # This is already a dict
                
                # Create a new HyperParameters object and set its values
                hp_for_build = HyperParameters()
                for key, value in best_hp_values.items():
                    # This approach assumes best_hp_values contains only actual hyperparameter values
                    # and not the full HP definition. Keras Tuner's `hp.Int`, `hp.Float`, etc.,
                    # would define the hyperparameter and its range.
                    # For simply using the values, we pass a dictionary to the hypermodel.
                    # However, self.build_model expects an 'hp' object that functions like HyperParameters.
                    # The most robust way is to rebuild the model by getting a *structured* HP object
                    # from the trial's full HP config.
                    pass # We will create a fresh HP object below based on the actual build_model signature.

                # Re-build the model with the best hyperparameters from the trial
                # The build_model method expects a HyperParameters object.
                # We need to reconstruct one that acts like the 'hp' object passed during tuning.
                # The direct way to do this is to ensure best_trial.hyperparameters
                # itself is passed, as it IS the HyperParameters object.
                self.best_model = self.build_model(best_trial.hyperparameters) # Pass the actual HyperParameters object
                self.best_model.load_weights(self.get_best_weights_path()) # Assume get_best_weights_path exists
                logger.info("TensorFlow best model finalized and weights loaded.")
            else:
                logger.warning("No best trial found to finalize for TensorFlow model.")
                self.best_model = None
            return self.best_model
        except Exception as e:
            logger.error(f"Error finalizing TensorFlow model: {e}", exc_info=True)
            self.best_model = None
            return None

    def get_best_weights_path(self):
        # This method needs to be implemented to retrieve the path to the best trial's weights
        # Keras Tuner saves weights in trial directories.
        # This is a simplified example, the actual path needs to be resolved from trial metadata.
        best_trial = self.oracle.get_best_trials(num_trials=1)
        if best_trial:
            trial_id = best_trial[0].trial_id
            # This path logic depends on how ModelCheckpoint is configured in callbacks
            # and where Keras Tuner stores checkpoints.
            # Usually it's `directory/project_name/trial_id/checkpoint_file.h5`
            return os.path.join(self.directory, self.project_name, trial_id, 'checkpoint.h5')
        return None # Placeholder, implement robust path retrieval


# Custom Callback to sync results with OracleClient if running in a distributed setup
class OracleSyncCallback(tf.keras.callbacks.Callback):
    def __init__(self, oracle_client, trial_id, objective_name='val_loss', direction='min', **kwargs):
        super().__init__(**kwargs)
        self.oracle_client = oracle_client
        self.trial_id = trial_id
        # These will be set dynamically in on_train_begin, or passed directly from tuner if available
        self._objective_name = objective_name # Store the passed objective name
        self._direction = direction # Store the passed direction
        self.best_score = float('inf') if self._direction == 'min' else float('-inf')
        self.best_epoch = 0
        logger.info(f"Initialized OracleSyncCallback for trial {self.trial_id} with objective '{self._objective_name}' ({self._direction}).")

    def on_train_begin(self, logs=None):
        # On train begin, if objective name or direction need to be more dynamically determined
        # from the model's tuner, this is where you'd do it.
        # For now, we rely on them being passed during initialization.
        pass

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_score = logs.get(self._objective_name) # Use self._objective_name
        if current_score is None:
            logger.warning(f"Objective metric '{self._objective_name}' not found in logs for epoch {epoch}. Cannot report score.")
            return

        # Check for improvement
        improved = False
        if self._direction == 'min': # Use self._direction
            if current_score < self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                improved = True
        else: # direction == 'max'
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_epoch = epoch
                improved = True

        if improved:
            logger.info(f"Trial {self.trial_id} - Epoch {epoch+1}: New best {self._objective_name} = {current_score:.4f}")
            # Report interim results to OracleServer
            # Structure result as Keras Tuner expects: {'metric_name': value, 'step': epoch_num}
            result_to_report = {self._objective_name: float(current_score), 'step': int(epoch)}
            self.oracle_client.report_trial_result(self.trial_id, result_to_report)
        else:
            logger.debug(f"Trial {self.trial_id} - Epoch {epoch+1}: {self._objective_name} = {current_score:.4f} (No improvement)")
            # Still report, but maybe don't log as much detail if not improved to reduce noise
            result_to_report = {self._objective_name: float(current_score), 'step': int(epoch)}
            self.oracle_client.report_trial_result(self.trial_id, result_to_report)


    def on_train_end(self, logs=None):
        # Final update of trial status
        logger.info(f"Trial {self.trial_id} finished. Finalizing status as COMPLETED.")
        self.oracle_client.update_trial_status(self.trial_id, status="COMPLETED")

# Add a utility function to get common callbacks
def get_callbacks(hp, model_dir, trial_id, oracle_client=None, objective_name='val_loss', direction='min'):
    callbacks = [
        # EarlyStopping(
        #     monitor=objective_name,
        #     patience=hp.Int('es_patience', min_value=5, max_value=20, step=5, default=10),
        #     mode=direction,
        #     restore_best_weights=True,
        #     verbose=1
        # ),
        # ReduceLROnPlateau(
        #     monitor=objective_name,
        #     factor=hp.Float('lr_factor', min_value=0.2, max_value=0.5, step=0.1, default=0.2),
        #     patience=hp.Int('lr_patience', min_value=3, max_value=10, step=2, default=5),
        #     mode=direction,
        #     min_lr=hp.Float('min_lr', min_value=1e-7, max_value=1e-5, sampling='LOG', default=1e-6),
        #     verbose=1
        # ),
        TerminateOnNaN(),
        # Use ModelCheckpoint with a dynamic path for each trial
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'checkpoint.h5'),
            monitor=objective_name,
            save_best_only=True,
            mode=direction,
            verbose=0 # Make verbose=0 to avoid too much output
        ),
        CSVLogger(os.path.join(model_dir, 'training_log.csv')),
        TensorBoard(log_dir=os.path.join(model_dir, 'tensorboard_logs'), update_freq='epoch')
    ]

    if oracle_client:
        # Add the custom callback for syncing results with the OracleServer
        callbacks.append(OracleSyncCallback(oracle_client, trial_id, objective_name, direction))
    
    return callbacks
