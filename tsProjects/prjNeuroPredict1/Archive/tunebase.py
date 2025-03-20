#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTuner.py
File: tsPyPackages/tsMqlMLTuner/tsMqlMLTuner.py
Description: The Tuner method for the machine learning model.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
License: MIT License
"""

import logging
import os
logger = logging.getLogger(__name__)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision

from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam, Adadelta, Adagrad, Adamax, Ftrl
from tensorflow.keras.activations import relu, tanh, selu, elu, linear, sigmoid, softmax, softplus
from tensorflow.keras.regularizers import l2

from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.losses import MeanSquaredLogarithmicError, Poisson, KLDivergence, CosineSimilarity
from tensorflow.keras.metrics import MSE, MAE, MAPE, MSLE, Poisson, KLDivergence, CosineSimilarity, Accuracy

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_tuner as kt
import pathlib
import numpy as np
from datetime import date


class CMdtuner:
    def __init__(self, **kwargs):
        # Initialize hypermodel parameters from kwargs
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        logger.info(f"Hypermodel parameters: {self.hypermodel_params}")

        # Set additional defaults from hypermodel_params
        self.verbose = self.hypermodel_params.get('verbose', 1)
        self.restore_best_weights = self.hypermodel_params.get('restore_best_weights', True)
        self.save_best_only = self.hypermodel_params.get('save_best_only', True)
        self.metric = self.hypermodel_params.get('metric', 'mse')
        self.loss = self.hypermodel_params.get('loss', 'mse')
        self.learning_rate = self.hypermodel_params.get('learning_rate', 1e-3)
        self.modelsummary = self.hypermodel_params.get('modelsummary', False)

        # Set dropout (defaulting to 0.2 if not provided)
        self.dropout = self.hypermodel_params.get('dropout', 0.2)

        # Datasets and cast mode
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')
        self.castmode = kwargs.get('castmode', 'float64')

        if self.castmode == 'float64':
            self.castval = self.cast_to_float64
        elif self.castmode == 'float32':
            self.castval = self.cast_to_float32
        elif self.castmode == 'float16':
            self.castval = self.cast_to_float16 

        if self.traindataset is not None:
            self.traindataset = self.traindataset.map(self.castval)
        if self.valdataset is not None:
            self.valdataset = self.valdataset.map(self.castval)
        if self.testdataset is not None:
            self.testdataset = self.testdataset.map(self.castval)

        # Environment source parameters from hypermodel_params
        self.mp_pl_platform_base = self.hypermodel_params.get('base', {}).get('mp_glob_base_platform_dir', None)
        self.checkpoint_filepath = self.hypermodel_params.get('base', {}).get('mp_glob_base_ml_checkpoint_filepath', None)
       
        self.base_path = self.hypermodel_params.get('base', {}).get('mp_glob_base_path', None)
        self.project_dir = self.hypermodel_params.get('base', {}).get('mp_glob_base_ml_project_dir', None)
        self.baseuniq = self.hypermodel_params.get('base', {}).get('mp_glob_sub_ml_baseuniq', None)
        self.modeldatapath = self.hypermodel_params.get('base', {}).get('mp_glob_base_ml_modeldir', None)
        
        self.modelname = self.hypermodel_params.get('base', {}).get('mp_glob_sub_ml_model_name', None)
        self.project_name = self.hypermodel_params.get('base', {}).get('mp_glob_sub_ml_model_name', None)
        logging.info(f"TuneParams: self.modeldatapath: {self.modeldatapath}")

        logging.info(f"TuneParams: self.mp_pl_platform_base: {self.mp_pl_platform_base}")
        logging.info(f"TuneParams: self.base_path: {self.base_path}")
        logging.info(f"TuneParams: self.project_dir: {self.project_dir}")
        logging.info(f"TuneParams: self.baseuniq: {self.baseuniq}")
        logging.info(f"TuneParams: self.modelname: {self.modelname}")
       
        logging.info(f"TuneParams: self.checkpoint_filepath: {self.checkpoint_filepath}")
        
        # Validate critical path parameters
        if not self.base_path:
            raise ValueError("The 'mp_glob_base_path' parameter must be provided in hypermodel_params.")
        if not self.project_name:
            raise ValueError("The 'mp_glob_base_ml_project_dir' parameter must be provided in hypermodel_params.")

        # Hypermodel parameters
        self.cnn_model = self.hypermodel_params.get('mltune',{}).get('cnn_model', True)
        self.lstm_model = self.hypermodel_params.get('mltune',{}).get('lstm_model', True)
        self.gru_model = self.hypermodel_params.get('mltune',{}).get('gru_model', True)
        self.transformer_model = self.hypermodel_params.get('mltune',{}).get('transformer_model', True)
        self.multiactivate = self.hypermodel_params.get('mltune',{}).get('multiactivate', True)
        self.multi_branches = self.hypermodel_params.get('mltune',{}).get('multi_branches', True)
        self.input_shape = self.hypermodel_params.get('mltune',{}).get('input_shape', None)
        self.data_input_shape = self.hypermodel_params.get('mltune',{}).get('data_input_shape', None)
        self.multi_inputs = self.hypermodel_params.get('mltune',{}).get('multi_inputs', False)
        self.batch_size = self.hypermodel_params.get('mltune',{}).get('batch_size', 32)
        self.multi_outputs = self.hypermodel_params.get('mltune',{}).get('multi_outputs', False)
        self.label_columns = self.hypermodel_params.get('mltune',{}).get('label_columns', None)
        self.input_width = self.hypermodel_params.get('mltune',{}).get('input_width', 24)
        self.shift = self.hypermodel_params.get('mltune',{}).get('shift', 24)
        self.total_window_size = self.input_width + self.shift
        self.batch_size = self.hypermodel_params.get('mltune',{}).get('batch_size', 8)
        self.keras_tuner = self.hypermodel_params.get('mltune',{}).get('keras_tuner', 'Hyperband')
        self.hyperband_iterations = self.hypermodel_params.get('mltune',{}).get('hyperband_iterations', 1)
        self.max_epochs = self.hypermodel_params.get('mltune',{}).get('max_epochs', 100)
        self.min_epochs = self.hypermodel_params.get('mltune',{}).get('min_epochs', 10)
        self.tf_param_epochs = self.hypermodel_params.get('mltune',{}).get('tf_param_epochs', 10)
        self.epochs = self.hypermodel_params.get('mltune',{}).get('epochs', 2)
        self.tune_new_entries = self.hypermodel_params.get('mltune',{}).get('tune_new_entries', True)
        self.allow_new_entries = self.hypermodel_params.get('mltune',{}).get('allow_new_entries', True)
        self.num_trials = self.hypermodel_params.get('mltune',{}).get('num_trials', 3)
        self.max_retries_per_trial = self.hypermodel_params.get('mltune',{}).get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = self.hypermodel_params.get('mltune',{}).get('max_consecutive_failed_trials', 3)
        self.steps_per_execution = self.hypermodel_params.get('mltune',{}).get('steps_per_execution', 50)
        self.executions_per_trial = self.hypermodel_params.get('mltune',{}).get('executions_per_trial', 1)
        self.overwrite = self.hypermodel_params.get('mltune',{}).get('overwrite', True)
        self.factor = self.hypermodel_params.get('mltune',{}).get('factor', 10)
        self.objective = self.hypermodel_params.get('mltune',{}).get('objective', 'val_loss')

        # Additional defaults for callbacks and tuning
        self.chk_verbosity = self.hypermodel_params.get('mltune',{}).get('chk_verbosity', 1)
        self.chk_mode = self.hypermodel_params.get('mltune',{}).get('chk_mode', 'min')
        self.chk_monitor = self.hypermodel_params.get('mltune',{}).get('chk_monitor', 'val_loss')
        self.chk_sav_freq = self.hypermodel_params.get('mltune',{}).get('chk_sav_freq', 'epoch')
        self.chk_patience = self.hypermodel_params.get('mltune',{}).get('chk_patience', 3)

        self.unitmin = self.hypermodel_params.get('mltune',{}).get('unitmin', 32)
        self.unitmax = self.hypermodel_params.get('mltune',{}).get('unitmax', 512)
        self.unitstep = self.hypermodel_params.get('mltune',{}).get('unitstep', 32)
        self.defaultunits = self.hypermodel_params.get('mltune',{}).get('defaultunits', 128)
        self.all_modelscale = self.hypermodel_params.get('mltune',{}).get('all_modelscale', 1.0)
        self.cnn_modelscale = self.hypermodel_params.get('mltune',{}).get('cnn_modelscale', 1.0)
        self.lstm_modelscale = self.hypermodel_params.get('mltune',{}).get('lstm_modelscale', 1.0)
        self.gru_modelscale = self.hypermodel_params.get('mltune',{}).get('gru_modelscale', 1.0)
        self.trans_modelscale = self.hypermodel_params.get('mltune',{}).get('trans_modelscale', 1.0)
        self.transh_modelscale = self.hypermodel_params.get('mltune',{}).get('transh_modelscale', 1.0)
        self.transff_modelscale = self.hypermodel_params.get('mltune',{}).get('transff_modelscale', 1.0)
        self.dense_modelscale = self.hypermodel_params.get('mltune',{}).get('dense_modelscale', 1.0)

        # Model configuration defaults for transformer, LSTM, GRU, CNN and Dense layers
        self.trans_dim_min = self.hypermodel_params.get('mltune',{}).get('trans_dim_min', 32 // self.trans_modelscale)
        self.trans_dim_max = self.hypermodel_params.get('mltune',{}).get('trans_dim_max', 256 // self.trans_modelscale)
        self.trans_dim_step = self.hypermodel_params.get('mltune', {}).get('trans_dim_step', 32 // self.trans_modelscale)
        self.trans_dim_default = self.hypermodel_params.get('mltune',{}).get('trans_dim_default', 64 // self.trans_modelscale)
        self.lstm_units_min = self.hypermodel_params.get('mltune',{}).get('lstm_units_min', 32 // self.lstm_modelscale)
        self.lstm_units_max = self.hypermodel_params.get('mltune',{}).get('lstm_units_max', 128 // self.lstm_modelscale)
        self.lstm_units_step = self.hypermodel_params.get('mltune',{}).get('lstm_units_step', 32 // self.lstm_modelscale)
        self.lstm_units_default = self.hypermodel_params.get('mltune',{}).get('lstm_units_default', 64 // self.lstm_modelscale)
        self.gru_units_min = self.hypermodel_params.get('mltune',{}).get('gru_units_min', 32 // self.gru_modelscale)
        self.gru_units_max = self.hypermodel_params.get('mltune',{}).get('gru_units_max', 128 // self.gru_modelscale)
        self.gru_units_step = self.hypermodel_params.get('mltune',{}).get('gru_units_step', 32 // self.gru_modelscale)
        self.gru_units_default = self.hypermodel_params.get('mltune',{}).get('gru_units_default', 64 // self.gru_modelscale)
        self.cnn_units_min = self.hypermodel_params.get('mltune',{}).get('cnn_units_min', 32 // self.cnn_modelscale)
        self.cnn_units_max = self.hypermodel_params.get('mltune',{}).get('cnn_units_max', 128 // self.cnn_modelscale)
        self.cnn_units_step = self.hypermodel_params.get('mltune',{}).get('cnn_units_step', 32 // self.cnn_modelscale)
        self.cnn_units_default = self.hypermodel_params.get('mltune',{}).get('cnn_units_default', 64 // self.cnn_modelscale)
        self.trans_heads_min = self.hypermodel_params.get('mltune',{}).get('trans_heads_min', 2 // self.transh_modelscale)
        self.trans_heads_max = self.hypermodel_params.get('mltune',{}).get('trans_heads_max', 8 // self.transh_modelscale)
        self.trans_heads_step = self.hypermodel_params.get('mltune',{}).get('trans_heads_step', 2 // self.transh_modelscale)
        self.trans_ff_min =  self.hypermodel_params.get('mltune' ,{}).get('trans_ff_min', int(64 // self.transff_modelscale))
        self.trans_ff_max =  self.hypermodel_params.get('mltune' ,{}).get('trans_ff_max', int(512 // self.transff_modelscale))
        self.trans_ff_step = self.hypermodel_params.get('mltune' ,{}).get('trans_ff_step', int(64 // self.transff_modelscale))
        self.dense_units_min = self.hypermodel_params.get('mltune',{}).get('dense_units_min', int(32 // self.dense_modelscale))
        self.dense_units_max = self.hypermodel_params.get('mltune',{}).get('dense_units_max', int(128 // self.dense_modelscale))
        self.dense_units_step = self.hypermodel_params.get('mltune',{}).get('dense_units_step', int(32 // self.dense_modelscale))

        # Loss and metric configurations
        self.loss_functions = {
            'mse': MeanSquaredError,
            'binary_crossentropy': BinaryCrossentropy,
            'mae': MeanAbsoluteError,
            'mape': MeanAbsolutePercentageError,
            'msle': MeanSquaredLogarithmicError,
            'poisson': Poisson,
            'kld': KLDivergence,
            'cosine_similarity': CosineSimilarity,
            'mean_squared_error': MeanSquaredError,    
        }

        self.metrics_dict = {
            'accuracy': Accuracy,
            'mae': MAE,
            'mse': MSE,
            'mape': MAPE,
            'msle': MSLE,
            'poisson': Poisson,
            'cosine_similarity': CosineSimilarity
        }

        # TensorFlow debugging flags
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)

        # Output dimensions
        self.output_dim = kwargs.get('output_dim', 1)

        # Ensure the base path exists
        logger.info(f"Base path: {self.base_path}")
        os.makedirs(self.base_path, exist_ok=True)

        # Enable XLA for performance boost
        tf.config.optimizer.set_jit(True)

        # Set global policy for mixed precision
        mixed_precision.set_global_policy('mixed_float16')

        # Distributed Training Strategy
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

        # TensorFlow debugging (optional)
        self.enable_debugging(kwargs)

        # Prepare input shapes (fix input shape if needed)
        self.prepare_shapes()

        # Validate configurations
        self.validate_config()

        # Initialize the tuner
        self.initialize_tuner()

    @staticmethod
    def cast_to_float32(x, y):
        return tf.cast(x, tf.float32), y

    @staticmethod
    def cast_to_float64(x, y):
        return tf.cast(x, tf.float64), y

    @staticmethod
    def cast_to_float16(x, y):
        return tf.cast(x, tf.float16), y

    def enable_debugging(self, kwargs):
        if kwargs.get('tf1', False):
            tf.debugging.set_log_device_placement(True)
        if kwargs.get('tf2', False):
            tf.debugging.enable_check_numerics()

    def prepare_shapes(self):
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")
        # If data_input_shape has an extra dimension (e.g. 4D), remove the first extra dimension.
        if len(self.data_input_shape) == 4:
            # Assume the first dimension is redundant (e.g. (None, 1440, 1) is desired)
            self.data_input_shape = self.data_input_shape[1:]
            logger.info(f"Adjusted data input shape to: {self.data_input_shape}")
        self.main_input_shape = self.get_shape(self.data_input_shape)

    @staticmethod
    def get_shape(data_shape):
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D or 3D.")
        return tuple(data_shape)

    def validate_config(self):
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

    def initialize_tuner(self):
        hp = kt.HyperParameters()

        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam', 'adadelta', 'adagrad', 'adamax', 'ftrl'])
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        hp.Choice('loss', ['binary_crossentropy', 'mse', 'mae', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity'])
        hp.Choice('activation', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
        hp.Choice('dense_1_activation', ['relu', 'tanh'])
        hp.Choice('metric', ['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'cosine_similarity'])

        hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log', default=1e-4)

        self.tunemodeepochs = self.hypermodel_params.get('tunemodeepochs', False)
        self.tunemode = self.hypermodel_params.get('tunemode', False)

        if self.tunemodeepochs:
            hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=1)
        else:
            hp.Fixed('epochs', self.min_epochs)

        if self.tunemode:
            hp.Int('cnn_filters', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
            hp.Int('cnn_kernel_size', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
        else:
            hp.Int('cnn_filters', min_value=2, max_value=5, step=1, default=3)
            hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1, default=3)

        hp.Int('lstm_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
        hp.Int('gru_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
        hp.Int('cnn_units', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
        hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step)

        hp.Int('num_transformer_blocks', min_value=1, max_value=3, step=1)
        hp.Int('trans_dim', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step, default=self.trans_dim_default)
        
        hp.Int('num_cnn_layers', min_value=1, max_value=3)
        hp.Int('num_lstm_layers', min_value=1, max_value=2)
        hp.Int('num_gru_layers', min_value=1, max_value=2)
        
        logger.info(f"Tuning Max epochs between {self.min_epochs} and {self.max_epochs}")
        logger.info(f"Tuner mode: {self.tunemode}, Tuner mode epochs: {self.tunemodeepochs}")

        tuner_classes = {
            'random': kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian': kt.BayesianOptimization
        }
        logger.info(f"Tuner Service Checker: {self.keras_tuner}")
        if self.keras_tuner in tuner_classes:
            logger.info(f"Tuner Service is: {self.keras_tuner}")
        try:
            if self.keras_tuner in tuner_classes:
                self.tuner = tuner_classes[self.keras_tuner](
                    hypermodel=self.build_model,
                    hyperparameters=hp,  # Pass the HyperParameters object
                    hyperband_iterations=self.hyperband_iterations,
                    objective=self.objective,
                    max_epochs=self.max_epochs,
                    factor=self.factor,
                    directory=self.base_path,
                    project_name=self.modelname,
                    overwrite=self.overwrite,
                    tune_new_entries=self.tune_new_entries,
                    allow_new_entries=self.allow_new_entries,
                    max_retries_per_trial=self.max_retries_per_trial,
                    max_consecutive_failed_trials=self.max_consecutive_failed_trials,
                    executions_per_trial=self.executions_per_trial,
                )
                self.tuner.search_space_summary()  # Only call if tuner initialized
            else:
                raise ValueError(f"Unsupported keras_tuner type: {self.keras_tuner}")
        except Exception as e:
            logging.error(f"Error initializing tuner: {e}")
            self.tuner = None  # Set to None to avoid future errors

    def build_model(self, hp):
        # Shared Input Logic
        shared_input = Input(shape=self.main_input_shape, name='shared_input')
        # If the input shape is still 4D, add a Reshape layer to convert it to 3D for Conv1D layers.
        if len(self.main_input_shape) == 3 and None in self.main_input_shape:
            # This is a safeguard; normally prepare_shapes should have fixed the shape.
            x = Reshape(self.main_input_shape[1:])(shared_input)
            shared_input = x
        inputs = [] if self.multi_inputs else [shared_input]
        branches = []
        logger.info(f"Shared input shape: {shared_input.shape} self.multi_inputs {self.multi_inputs}")
       
        # CNN Branch
        if self.cnn_model:
            cnn_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='cnn_input')
            if self.multi_inputs:
                inputs.append(cnn_input)
            logger.info(f"Input shape cnn: {cnn_input.shape}")
                
            if self.tunemode:
                cnn_branch = cnn_input
                num_cnn_layers = hp.get('num_cnn_layers')
                for i in range(num_cnn_layers):  
                    filters = hp.get(f'cnn_filters_{i}')
                    kernel_size = hp.get(f'cnn_kernel_size_{i}')
                    activation = hp.get(f'activation_{i}')
                    cnn_branch = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
                    cnn_branch = LayerNormalization()(cnn_branch)
                    cnn_branch = Dropout(self.dropout)(cnn_branch)
            else:
                cnn_branch = cnn_input
                num_cnn_layers = hp.Int('num_cnn_layers', min_value=1, max_value=3)
                for i in range(num_cnn_layers):
                    filters = hp.Int(f'cnn_filters_{i}', min_value=32, max_value=128, step=32)
                    kernel_size = hp.Int(f'cnn_kernel_size_{i}', min_value=2, max_value=5, step=1)
                    activation = hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
                    cnn_branch = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
                    cnn_branch = LayerNormalization()(cnn_branch)
                    cnn_branch = Dropout(self.dropout)(cnn_branch)
            cnn_branch = Flatten()(cnn_branch)
            branches.append(cnn_branch)
       
        # LSTM Branch
        if self.lstm_model:
            lstm_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='lstm_input')
            logger.info(f"Input shape lstm: {lstm_input.shape}")
            if self.multi_inputs:
                inputs.append(lstm_input)
          
            if self.tunemode:
                lstm_branch = lstm_input
                num_lstm_layers = hp.get('num_lstm_layers')
                for i in range(num_lstm_layers):
                    units = hp.get(f'lstm_units_{i}')
                    activation = hp.get(f'activation_{i}')
                    lstm_branch = LSTM(units=units, activation=activation, return_sequences=(i < num_lstm_layers - 1))(lstm_branch)
                    lstm_branch = LayerNormalization()(lstm_branch)
                    lstm_branch = Dropout(self.dropout)(lstm_branch)
            else:
                lstm_branch = lstm_input
                num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=2)
                for i in range(num_lstm_layers):
                    units = hp.Int(f'lstm_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep)
                    activation = hp.Choice(f'lstm_activation_{i}', ['tanh', 'relu'])
                    lstm_branch = LSTM(units=units, activation=activation, return_sequences=(i < num_lstm_layers - 1))(lstm_branch)
                    lstm_branch = LayerNormalization()(lstm_branch)
                    lstm_branch = Dropout(self.dropout)(lstm_branch)
            branches.append(lstm_branch)
       
        # GRU Branch
        if self.gru_model:
            gru_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='gru_input')
            if self.multi_inputs:
                inputs.append(gru_input)
            logger.info(f"Input shape gru : {gru_input.shape}")
            if self.tunemode:
                gru_branch = gru_input
                num_gru_layers = hp.get('num_gru_layers')
                for i in range(num_gru_layers):
                    units = hp.get(f'gru_units_{i}')
                    activation = hp.get(f'activation_{i}')
                    gru_branch = GRU(units=units, activation=activation, return_sequences=(i < num_gru_layers - 1))(gru_branch)
                    gru_branch = LayerNormalization()(gru_branch)
                    gru_branch = Dropout(self.dropout)(gru_branch)
            else:
                gru_branch = gru_input
                num_gru_layers = hp.Int('num_gru_layers', min_value=1, max_value=2)
                for i in range(num_gru_layers):
                    units = hp.Int(f'gru_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep)
                    activation = hp.Choice(f'gru_activation_{i}', ['tanh', 'relu'])
                    gru_branch = GRU(units=units, activation=activation, return_sequences=(i < num_gru_layers - 1))(gru_branch)
                    gru_branch = LayerNormalization()(gru_branch)
                    gru_branch = Dropout(self.dropout)(gru_branch)
            branches.append(gru_branch)

        # Transformer Branch
        if self.transformer_model:
            transformer_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='transformer_input')
            if self.multi_inputs:
                inputs.append(transformer_input)
            logger.info(f"Input shape trans: {transformer_input.shape}")
            transformer_branch = transformer_input

            num_transformer_blocks = hp.get('num_transformer_blocks')
            trans_dim = hp.get('trans_dim')

            for i in range(num_transformer_blocks):
                transformer_branch = self.transformer_block(transformer_branch, hp, i, trans_dim)
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)
            transformer_branch = Dense(64, activation=hp.get('activation'))(transformer_branch)
            transformer_branch = Dropout(self.dropout)(transformer_branch)
            branches.append(transformer_branch)

        # Concatenate branches if multiple
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        dense_1 = Dense(
            units=hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step),
            activation=hp.Choice('dense_1_activation', ['relu', 'tanh']),
            kernel_regularizer=tf.keras.regularizers.l2(hp.get('l2_reg'))
        )(concatenated)
        dense_dropout = Dropout(self.dropout)(dense_1)
        output = Dense(self.output_dim, activation="linear" if self.output_dim > 1 else "sigmoid")(dense_dropout)

        logger.info(f"Including {len(inputs)} input(s) and {len(branches)} branch(es).")
       
        if len(inputs) == 1:
            logger.info(f"Input shape1: {inputs[0].shape}")
            model = Model(inputs=inputs, outputs=output)
        elif len(inputs) in [2, 3, 4]:
            logger.info("Input shapes: " + ", ".join(str(inp.shape) for inp in inputs))
            model = Model(inputs=inputs, outputs=output)          
        else:
            model = Model(inputs=shared_input, outputs=output)
           
        if len(branches) == 1:
            logger.info(f"Branch shape1: {branches[0].shape}")
        else:
            logger.info("Branch shapes: " + ", ".join(str(branch.shape) for branch in branches))

        logger.info(f"Output shape: {output.shape}")
        logger.info("Model built and compiled.")

        if self.tunemode:
            loss_fn = self.loss_functions[hp.get('loss')]
            metric_fn = self.metrics_dict[hp.get('metric')]
        else:
            loss_fn = self.loss_functions[self.loss]
            metric_fn = self.metrics_dict[self.metric]

        if self.tunemode:
            optimizer = self.get_optimizer(hp.get('optimizer'), hp.get('learning_rate'))
        else:
            optimizer = self.get_optimizer(self.hypermodel_params.get('optimizer', 'adam'), self.learning_rate)

        model.compile(
            optimizer=optimizer,
            loss=hp.get('loss') if self.tunemode else self.loss,
            steps_per_execution=self.steps_per_execution,
            metrics=[hp.get('metric')] if self.tunemode else [self.metric]
        )

        if self.modelsummary:
            model.summary()
         
        return model

    def get_optimizer(self, optimizer_name, learning_rate):
        optimizers = {
            'adam': Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'sgd': tf.keras.optimizers.SGD,
            'nadam': tf.keras.optimizers.Nadam,
            'adadelta': tf.keras.optimizers.Adadelta,
            'adagrad': tf.keras.optimizers.Adagrad,
            'adamax': tf.keras.optimizers.Adamax,
            'ftrl': tf.keras.optimizers.Ftrl
        }
        return optimizers[optimizer_name](learning_rate=learning_rate)

    def get_callbacks(self):
        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=self.verbose, restore_best_weights=self.restore_best_weights),
            ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=self.save_best_only, verbose=self.verbose),
            TensorBoard(log_dir=os.path.join(self.base_path, 'logs')),
            ReduceLROnPlateau(monitor=self.objective, factor=0.1, patience=self.chk_patience, min_lr=1e-6, verbose=self.verbose)
        ]

    def transformer_block(self, inputs, hp, block_num, dim):
        key_dim = hp.Int(f'key_dim_{block_num}', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step)
        num_heads = hp.Int(f'num_heads_{block_num}', min_value=self.trans_heads_min, max_value=self.trans_heads_max, step=self.trans_heads_step)
        ff_dim = hp.Int(f'ff_dim_{block_num}', min_value=self.trans_ff_min, max_value=self.trans_ff_max, step=self.trans_ff_step)
        activation = hp.Choice(f'activation_{block_num}', ['relu', 'tanh', 'gelu'])

        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attn_out = LayerNormalization()(inputs + attn_out)

        ffn_out = Dense(ff_dim, activation=activation)(attn_out)
        ffn_out = Dropout(self.dropout)(ffn_out)
        ffn_out = Dense(inputs.shape[-1])(ffn_out)
        ffn_out = LayerNormalization()(attn_out + ffn_out)
        ffn_out = Dropout(self.dropout)(ffn_out)

        return ffn_out

    def run_search(self):
        logger.info("Running tuner search...")
        try:
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                epochs=self.max_epochs,
                verbose=self.verbose,
                callbacks=self.get_callbacks()
            )

            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps:
                raise ValueError("No hyperparameters found. Ensure tuning has been run successfully.")
            
            logger.info(f"Best hyperparameters: {best_hps[0].values}")

        except Exception as e:
            logger.error(f"Error during tuning: {e}")

    def export_best_model(self, ftype='tf'):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.base_path, self.project_name, 'best_model')
            logger.info(f"Exporting best model to {export_path}")
            if ftype == 'h5':
                best_model.save(export_path + '.h5')
                logger.info(f"Model saved to {export_path}.h5")
            else:
                best_model.save(export_path + '.keras')
                logger.info(f"Model saved to {export_path}.keras")
        except IndexError:
            logger.info("No models found to export.")
        except Exception as e:
            logger.info(f"Error saving the model: {e}")

    def check_and_load_model(self, lpbase_path, ftype='tf'):
        """
        Check if the model file exists and load it.
        """
        logger.info(f"Checking for model file at base_path {lpbase_path}")
        logger.info(f"Project name: {self.project_name}")

        if ftype == 'h5':
            modelext = 'h5'
            model_path = os.path.join(lpbase_path, self.project_name)
            logger.info(f"Model path: {model_path}")
        else:
            modelext = 'keras'
            model_path = os.path.join(lpbase_path, self.project_name)
            logger.info(f"Model path: {model_path}")

        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
                model.summary()
                return model
            else:
                logger.info(f"Model file does not exist at {model_path}")
                return None
        except Exception as e:
            logger.info(f"Error loading model: {e}")
            return None

    def get_best_hyperparameters(self):
        try:
            return self.tuner.get_best_hyperparameters(num_trials=1)[0]
        except Exception as e:
            logger.info(f"Error retrieving best hyperparameters: {e}")
            return None

    def run_prediction(self, test_data, batch_size=None):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            if isinstance(test_data, tf.data.Dataset):
                test_data = test_data.map(self.cast_to_float32)
            predictions = best_model.predict(test_data, batch_size=batch_size or self.batch_size)
            return predictions
        except IndexError:
            logger.info("No models found. Ensure tuning has been run successfully.")
        except Exception as e:
            logger.info(f"Error during prediction: {e}")
            return None

# End of file