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

import logging
import os
import pathlib
import tensorflow as tf
from datetime import date

# Get a logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure the logger level is set appropriately

# Platform imports
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk         = run_platform.RunPlatform()
os_platform  = platform_checker.get_platform()
loadmql      = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# TensorFlow/Keras imports
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Nadam, Adadelta, Adagrad, Adamax
from tensorflow.keras.activations import relu, tanh, selu, elu, linear, sigmoid, softmax, softplus
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError,
                                     MeanAbsolutePercentageError, MeanSquaredLogarithmicError,
                                     Poisson, KLDivergence, CosineSimilarity)
from tensorflow.keras.metrics import (MSE, MAE, MAPE, MSLE, Poisson, KLDivergence,
                                      CosineSimilarity, Accuracy)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_tuner as kt
import numpy as np

# Enable mixed precision and XLA JIT compilation for performance
tf.config.optimizer.set_jit(True)
mixed_precision.set_global_policy('mixed_float16')


class CMdtuner:
    def __init__(self, **kwargs):
        # Extract hypermodel parameters
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        logger.info(f"Hypermodel parameters: {self.hypermodel_params}")

        base = self.hypermodel_params.get('base', {})
        self.mp_pl_platform_base      = base.get('mp_glob_base_platform_dir', None)
        self.checkpoint_filepath       = base.get('mp_glob_base_ml_checkpoint_filepath', None)
        self.modeldatapath            = base.get('mp_glob_sub_ml_src_modeldata', None)
        # Model path
        self.base_path                 = base.get('mp_glob_base_path', None)
        self.project_dir               = base.get('mp_glob_base_ml_project_dir', None)
        self.baseuniq                  = base.get('mp_glob_sub_ml_baseuniq', None)
        self.modelname                 = base.get('mp_glob_sub_ml_model_name', None)

        logger.info(f"TuneParams: mp_pl_platform_base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: checkpoint_filepath: {self.checkpoint_filepath}")
        logger.info(f"TuneParams: base_path          : {self.base_path}")
        logger.info(f"TuneParams: project_dir        : {self.project_dir}")
        logger.info(f"TuneParams: baseuniq           : {self.baseuniq}")
        logger.info(f"TuneParams: modeldatapath      : {self.modeldatapath}")
        logger.info(f"TuneParams: modelname          : {self.modelname}")
       
        if not self.base_path:
            raise ValueError("The 'mp_glob_base_path' parameter must be provided in hypermodel_params.")
        if not self.project_dir:
            raise ValueError("The 'mp_glob_base_ml_project_dir' parameter must be provided in hypermodel_params.")

        # Data parameters
        data = self.hypermodel_params.get('data', {})
        self.mp_data_load       = data.get('mp_data_load', True)
        self.mp_data_save       = data.get('mp_data_save', False)
        self.mp_data_rows       = data.get('mp_data_rows', 1000)
        self.mp_data_rowcount   = data.get('mp_data_rowcount', 10000)
        self.df1_mp_data_filter_int = data.get('df1_mp_data_filter_int', False)

        logger.info(f"Data parameters: mp_data_load: {self.mp_data_load}")
        logger.info(f"Data parameters: mp_data_save: {self.mp_data_save}")
        logger.info(f"Data parameters: mp_data_rows: {self.mp_data_rows}")
        logger.info(f"Data parameters: mp_data_rowcount: {self.mp_data_rowcount}")
        logger.info(f"Data parameters: df1_mp_data_filter_int: {self.df1_mp_data_filter_int}")

        # Machine Learning parameters
        ml = self.hypermodel_params.get('ml', {})
        self.input_keyfeat         = ml.get('mp_ml_input_keyfeat', 'Close')
        self.feature1              = ml.get('feature1', 'Open')
        self.feature2              = ml.get('feature2', 'High')
        self.feature3              = ml.get('feature3', 'Low')
        self.feature4              = ml.get('feature4', 'Close')
        self.feature5              = ml.get('feature5', 'Volume')

        logger.info(f"ML parameters: input_keyfeat: {self.input_keyfeat}")
        logger.info(f"ML parameters: feature1: {self.feature1}")
        logger.info(f"ML parameters: feature2: {self.feature2}")
        logger.info(f"ML parameters: feature3: {self.feature3}")
        logger.info(f"ML parameters: feature4: {self.feature4}")
        logger.info(f"ML parameters: feature5: {self.feature5}")

        # Tuning parameters
        mltune = self.hypermodel_params.get('mltune', {})
        self.today                   = mltune.get('today', '2025-03-16 17:27:46')
        self.seed                    = mltune.get('seed', 42)
        self.tunemode                = mltune.get('tunemode', 'Hyperband')
        self.tunemodeepochs          = mltune.get('tunemodeepochs', True)
        self.batch_size              = mltune.get('batch_size', 16)  # Reduced batch size
        self.epochs                  = mltune.get('epochs', 2)
        self.num_trials              = mltune.get('num_trials', 3)
        self.max_epochs              = mltune.get('max_epochs', 120)
        self.min_epochs              = mltune.get('min_epochs', 10)
        self.hyperband_iterations    = mltune.get('hyperband_iterations', 1)
        self.factor                  = mltune.get('factor', 10)
        self.objective               = mltune.get('objective', 'val_loss')
        self.input_shape             = mltune.get('input_shape', None)
        self.data_input_shape        = mltune.get('data_input_shape', None)
        self.multi_inputs            = mltune.get('multi_inputs', False)
        self.multi_branches          = mltune.get('multi_branches', True)
        self.multi_outputs           = mltune.get('multi_outputs', False)
        self.label_columns           = mltune.get('label_columns', None)
        self.shift                   = mltune.get('shift', 24)
        self.input_width             = mltune.get('input_width', 1440)
        self.total_window_size       = self.input_width + self.shift
        self.tune_new_entries       = mltune.get('tune_new_entries', True)
        self.allow_new_entries       = mltune.get('allow_new_entries', True)
        self.max_retries_per_trial   = mltune.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = mltune.get('max_consecutive_failed_trials', 3)
        self.executions_per_trial    = mltune.get('executions_per_trial', 1)
        self.overwrite               = mltune.get('overwrite', False)

        logger.info(f"Tuning parameters: today            : {self.today}")
        logger.info(f"Tuning parameters: seed             : {self.seed}")
        logger.info(f"Tuning parameters: input_shape      : {self.input_shape}")
        logger.info(f"Tuning parameters: data_input_shape : {self.data_input_shape}")
        logger.info(f"Tuning parameters: multi_inputs     : {self.multi_inputs}")
        logger.info(f"Tuning parameters: multi_branches   : {self.multi_branches}")
        logger.info(f"Tuning parameters: multi_outputs    : {self.multi_outputs}")
        logger.info(f"Tuning parameters: label_columns    : {self.label_columns}")
        logger.info(f"Tuning parameters: shift            : {self.shift}")
        logger.info(f"Tuning parameters: input_width      : {self.input_width}")
        logger.info(f"Tuning parameters: total_window_size: {self.total_window_size}")
        logger.info(f"Tuning parameters: tune_new_entries : {self.tune_new_entries}")
        logger.info(f"Tuning parameters: allow_new_entries: {self.allow_new_entries}")
        logger.info(f"Tuning parameters: max_retries_per_trial: {self.max_retries_per_trial}")
        logger.info(f"Tuning parameters: max_consecutive_failed_trials: {self.max_consecutive_failed_trials}")
        logger.info(f"Tuning parameters: executions_per_trial: {self.executions_per_trial}")

        # New tuning parameters 
        self.unitmin         = mltune.get('unitmin', 32)
        self.unitmax         = mltune.get('unitmax', 512)
        self.unitstep        = mltune.get('unitstep', 32)
        self.defaultunits    = mltune.get('defaultunits', 128)
        self.all_modelscale  = mltune.get('all_modelscale', 8.0)
        self.cnn_modelscale  = mltune.get('cnn_modelscale', 8.0)
        self.lstm_modelscale = mltune.get('lstm_modelscale', 8.0)
        self.gru_modelscale  = mltune.get('gru_modelscale', 8.0)
        self.trans_modelscale = mltune.get('trans_modelscale', 8.0)
        self.transh_modelscale = mltune.get('transh_modelscale', 8.0)
        self.transff_modelscale = mltune.get('transff_modelscale', 8.0)
        self.dense_modelscale = mltune.get('dense_modelscale', 8.0)
        self.trans_dim_min      = mltune.get('trans_dim_min', 32 // self.trans_modelscale)
        self.trans_dim_max      = mltune.get('trans_dim_max', 256 // self.trans_modelscale)
        self.trans_dim_step     = mltune.get('trans_dim_step', 32 // self.trans_modelscale)
        self.trans_dim_default  = mltune.get('trans_dim_default', 64 // self.trans_modelscale)
        self.trans_heads_min    = mltune.get('trans_heads_min', 2)
        self.trans_heads_max    = mltune.get('trans_heads_max', 8)
        self.trans_heads_step   = mltune.get('trans_heads_step', 2)
        self.trans_ff_min       = mltune.get('trans_ff_min', int(64 // self.transff_modelscale))
        self.trans_ff_max       = mltune.get('trans_ff_max', int(512 // self.transff_modelscale))
        self.trans_ff_step      = mltune.get('trans_ff_step', int(64 // self.transff_modelscale))
        self.dense_units_min    = mltune.get('dense_units_min', int(32 // self.dense_modelscale))
        self.dense_units_max    = mltune.get('dense_units_max', int(128 // self.dense_modelscale))
        self.dense_units_step   = mltune.get('dense_units_step', int(32 // self.dense_modelscale))

        logger.info(f"Tuning parameters: unitmin          : {self.unitmin}")
        logger.info(f"Tuning parameters: unitmax          : {self.unitmax}")
        logger.info(f"Tuning parameters: unitstep         : {self.unitstep}")
        logger.info(f"Tuning parameters: defaultunits     : {self.defaultunits}")
        logger.info(f"Tuning parameters: all_modelscale   : {self.all_modelscale}")
        logger.info(f"Tuning parameters: cnn_modelscale   : {self.cnn_modelscale}")
        logger.info(f"Tuning parameters: lstm_modelscale  : {self.lstm_modelscale}")
        logger.info(f"Tuning parameters: gru_modelscale   : {self.gru_modelscale}")
        logger.info(f"Tuning parameters: 'trans_modelscale': {self.trans_modelscale}")
        logger.info(f"Tuning parameters: 'transh_modelscale': {self.transh_modelscale}")
        logger.info(f"Tuning parameters: 'transff_modelscale': {self.transff_modelscale}")
        logger.info(f"Tuning parameters: 'dense_modelscale': {self.dense_modelscale}")
        logger.info(f"Tuning parameters: 'trans_dim_min': {self.trans_dim_min}")
        logger.info(f"Tuning parameters: 'trans_dim_max': {self.trans_dim_max}")
        logger.info(f"Tuning parameters: 'trans_dim_step': {self.trans_dim_step}")
        logger.info(f"Tuning parameters: 'trans_dim_default': {self.trans_dim_default}")
        logger.info(f"Tuning parameters: 'trans_heads_min': {self.trans_heads_min}")
        logger.info(f"Tuning parameters: 'trans_heads_max': {self.trans_heads_max}")
        logger.info(f"Tuning parameters: 'trans_heads_step': {self.trans_heads_step}")
        logger.info(f"Tuning parameters: 'trans_ff_min': {self.trans_ff_min}")
        logger.info(f"Tuning parameters: 'trans_ff_max': {self.trans_ff_max}")
        logger.info(f"Tuning parameters: 'trans_ff_step': {self.trans_ff_step}")    
        logger.info(f"Tuning parameters: 'dense_units_min': {self.dense_units_min}")
        logger.info(f"Tuning parameters: 'dense_units_max': {self.dense_units_max}")
        logger.info(f"Tuning parameters: 'dense_units_step': {self.dense_units_step}")

        logger.info(f"Tuning parameters: tunemode          : {self.tunemode}")
        logger.info(f"Tuning parameters: tunemodeepochs    : {self.tunemodeepochs}")
        logger.info(f"Tuning parameters: batch_size        : {self.batch_size}")
        logger.info(f"Tuning parameters: epochs            : {self.epochs}")
        logger.info(f"Tuning parameters: num_trials        : {self.num_trials}")
        logger.info(f"Tuning parameters: max_epochs        : {self.max_epochs}")
        logger.info(f"Tuning parameters: min_epochs        : {self.min_epochs}")
        logger.info(f"Tuning parameters: hyperband_iterations: {self.hyperband_iterations}")
        logger.info(f"Tuning parameters: factor            : {self.factor}")
        logger.info(f"Tuning parameters: objective         : {self.objective}")

        # Checkpoint parameters  
        self.checkpoint_dir = self.checkpoint_filepath
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.modelname)
        self.overwrite = mltune.get('overwrite', False)
        self.chk_fullmodel = mltune.get('chk_fullmodel', True)
        self.chk_verbosity = mltune.get('chk_verbosity', 1)
        self.chk_mode = mltune.get('chk_mode', 'min')
        self.chk_monitor = mltune.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = mltune.get('chk_sav_freq', 'epoch')
        self.chk_patience = mltune.get('chk_patience', 10)
        self.save_best_only = mltune.get('save_best_only', True)

        logger.info(f"Checkpoint parameters: checkpoint_dir: {self.checkpoint_dir}")
        logger.info(f"Checkpoint parameters: checkpoint_path: {self.checkpoint_path}")
        logger.info(f"Checkpoint parameters: overwrite: {self.overwrite}")
        logger.info(f"Checkpoint parameters: chk_fullmodel: {self.chk_fullmodel}")
        logger.info(f"Checkpoint parameters: chk_verbosity: {self.chk_verbosity}")
        logger.info(f"Checkpoint parameters: chk_mode: {self.chk_mode}")
        logger.info(f"Checkpoint parameters: chk_monitor: {self.chk_monitor}")
        logger.info(f"Checkpoint parameters: chk_sav_freq: {self.chk_sav_freq}")
        logger.info(f"Checkpoint parameters: chk_patience: {self.chk_patience}")
        logger.info(f"Checkpoint parameters: save_best_only: {self.save_best_only}")

        # Datasets and cast mode setup
        self.traindataset = kwargs.get('traindataset')
        self.valdataset   = kwargs.get('valdataset')
        self.testdataset  = kwargs.get('testdataset')
        self.castmode     = kwargs.get('castmode', 'float64')
        if self.castmode == 'float64':
            self.castval = self.cast_to_float64
        elif self.castmode == 'float32':
            self.castval = self.cast_to_float32
        elif self.castmode == 'float16':
            self.castval = self.cast_to_float16

        # Optimize dataset pipelines: parallel mapping, caching and prefetching.
        AUTOTUNE = tf.data.AUTOTUNE
        if self.traindataset is not None:
            self.traindataset = self.traindataset.map(self.castval, num_parallel_calls=AUTOTUNE)\
                                                 .cache()\
                                                 .prefetch(buffer_size=AUTOTUNE)
        if self.valdataset is not None:
            self.valdataset = self.valdataset.map(self.castval, num_parallel_calls=AUTOTUNE)\
                                             .cache()\
                                             .prefetch(buffer_size=AUTOTUNE)
        if self.testdataset is not None:
            self.testdataset = self.testdataset.map(self.castval, num_parallel_calls=AUTOTUNE)\
                                               .cache()\
                                               .prefetch(buffer_size=AUTOTUNE)

        # Set up distributed training strategy
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)
        self.enable_debugging(kwargs)

        self.prepare_shapes()

        self.cnn_model         = mltune.get('cnn_model', True)
        self.lstm_model        = mltune.get('lstm_model', True)
        self.gru_model         = mltune.get('gru_model', True)
        self.transformer_model = mltune.get('transformer_model', True)
        self.multiactivate     = mltune.get('multiactivate', True)
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

        self.modelsummary = self.hypermodel_params.get('modelsummary', False)
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
        if len(self.data_input_shape) == 4:
            self.data_input_shape = self.data_input_shape[1:]
            logger.info(f"Adjusted data input shape to: {self.data_input_shape}")
        self.main_input_shape = self.get_shape(self.data_input_shape)

    @staticmethod
    def get_shape(data_shape):
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D or 3D.")
        return tuple(data_shape)

    def initialize_tuner(self):
        # Initialize hyperparameters (only once)
        hp = kt.HyperParameters()
        logger.info(f"Initializing tuner with hyperparameters... {hp}")

        # Define common hyperparameters
        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam', 'adadelta', 'adagrad', 'adamax'])
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        hp.Choice('loss', ['binary_crossentropy', 'mse', 'mae', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity'])
        hp.Choice('dense_1_activation', ['relu', 'tanh'])
        hp.Choice('metric', ['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'cosine_similarity'])
        hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log', default=1e-4)

        if self.tunemodeepochs:
            hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=1)
        else:
            hp.Fixed('epochs', self.min_epochs)

        if self.tunemode:
            # Tuning for CNN branch
            hp.Int('num_cnn_layers', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'cnn_filters_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Int(f'cnn_kernel_size_{i}', min_value=2, max_value=5, step=1, default=3)
                hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
            # Tuning for LSTM branch
            hp.Int('num_lstm_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'lstm_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Choice(f'lstm_activation_{i}', ['tanh', 'relu'])
            # Tuning for GRU branch
            hp.Int('num_gru_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'gru_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Choice(f'gru_activation_{i}', ['tanh', 'relu'])
            # Tuning for Transformer branch
            hp.Int('num_transformer_blocks', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'key_dim_{i}', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step, default=self.trans_dim_default)
                hp.Int(f'num_heads_{i}', min_value=self.trans_heads_min, max_value=self.trans_heads_max, step=self.trans_heads_step, default=self.trans_heads_min)
                hp.Int(f'ff_dim_{i}', min_value=self.trans_ff_min, max_value=self.trans_ff_max, step=self.trans_ff_step, default=self.trans_ff_min)
                hp.Choice(f'transformer_activation_{i}', ['relu', 'tanh', 'gelu'])
        else:
            hp.Fixed('cnn_filters', 3)
            hp.Fixed('cnn_kernel_size', 3)

        # Dense layer units
        hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step)

        logger.info(f"Tuning Max epochs between {self.min_epochs} and {self.max_epochs}")
        logger.info(f"Tuner mode: {self.tunemode}, Tuner mode epochs: {self.tunemodeepochs}")

        tuner_classes = {
            'random':    kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian':  kt.BayesianOptimization
        }
        logger.info(f"Tuner Service Checker: {self.tunemode}")
        if self.tunemode in tuner_classes:
            logger.info(f"Tuner Service is: {self.tunemode}")
            logger.info(f" Tuner directory is {self.base_path}")
            logger.info(f" Tuner Project name is {self.modelname}")
        try:
            if self.tunemode in tuner_classes:
                self.tuner = tuner_classes[self.tunemode](
                    hypermodel=self.build_model,
                    hyperparameters=hp,
                    hyperband_iterations=self.hyperband_iterations,
                    objective=self.objective,
                    max_epochs=self.max_epochs,
                    factor=self.factor,
                    directory=self.project_dir,
                    project_name=self.modelname,
                    overwrite=self.overwrite,
                    tune_new_entries=self.tune_new_entries,
                    allow_new_entries=self.allow_new_entries,
                    max_retries_per_trial=self.max_retries_per_trial,
                    max_consecutive_failed_trials=self.max_consecutive_failed_trials,
                    executions_per_trial=self.executions_per_trial,
                )
                self.tuner.search_space_summary()
            else:
                raise ValueError(f"Unsupported keras_tuner type: {self.tunemode}")
        except Exception as e:
            logger.error(f"Error initializing tuner: {e}")
            self.tuner = None

    def build_model(self, hp):
        # Shared input setup
        shared_input = Input(shape=self.main_input_shape, name='shared_input')
        if len(self.main_input_shape) == 3 and None in self.main_input_shape:
            shared_input = Reshape(self.main_input_shape[1:])(shared_input)
        inputs = [] if self.multi_inputs else [shared_input]
        branches = []
        logger.info(f"Shared input shape: {shared_input.shape}, multi_inputs: {self.multi_inputs}")

        # CNN Branch
        if self.cnn_model:
            cnn_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='cnn_input')
            if self.multi_inputs:
                inputs.append(cnn_input)
            logger.info(f"Input shape cnn: {cnn_input.shape}")
            cnn_branch = cnn_input
            if self.tunemode:
                num_cnn_layers = hp.get('num_cnn_layers')
                for i in range(num_cnn_layers):
                    filters = hp.get(f'cnn_filters_{i}')
                    kernel_size = hp.get(f'cnn_kernel_size_{i}')
                    activation = hp.get(f'cnn_activation_{i}')
                    cnn_branch = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
                    cnn_branch = LayerNormalization()(cnn_branch)
                    cnn_branch = Dropout(0.2)(cnn_branch)
            else:
                filters = hp.get('cnn_filters')
                kernel_size = hp.get('cnn_kernel_size')
                activation = 'relu'
                cnn_branch = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
                cnn_branch = LayerNormalization()(cnn_branch)
                cnn_branch = Dropout(0.2)(cnn_branch)
            cnn_branch = Flatten()(cnn_branch)
            branches.append(cnn_branch)

        # LSTM Branch
        if self.lstm_model:
            lstm_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='lstm_input')
            if self.multi_inputs:
                inputs.append(lstm_input)
            logger.info(f"Input shape lstm: {lstm_input.shape}")
            lstm_branch = lstm_input
            if self.tunemode:
                num_lstm_layers = hp.get('num_lstm_layers')
                for i in range(num_lstm_layers):
                    units = hp.get(f'lstm_units_{i}')
                    activation = hp.get(f'lstm_activation_{i}')
                    lstm_branch = LSTM(units=units, activation=activation, return_sequences=(i < num_lstm_layers - 1))(lstm_branch)
                    lstm_branch = LayerNormalization()(lstm_branch)
                    lstm_branch = Dropout(0.2)(lstm_branch)
            else:
                lstm_branch = LSTM(units=128, activation='tanh', return_sequences=False)(lstm_branch)
                lstm_branch = LayerNormalization()(lstm_branch)
                lstm_branch = Dropout(0.2)(lstm_branch)
            branches.append(lstm_branch)

        # GRU Branch
        if self.gru_model:
            gru_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='gru_input')
            if self.multi_inputs:
                inputs.append(gru_input)
            logger.info(f"Input shape gru: {gru_input.shape}")
            gru_branch = gru_input
            if self.tunemode:
                num_gru_layers = hp.get('num_gru_layers')
                for i in range(num_gru_layers):
                    units = hp.get(f'gru_units_{i}')
                    activation = hp.get(f'gru_activation_{i}')
                    gru_branch = GRU(units=units, activation=activation, return_sequences=(i < num_gru_layers - 1))(gru_branch)
                    gru_branch = LayerNormalization()(gru_branch)
                    gru_branch = Dropout(0.2)(gru_branch)
            else:
                gru_branch = GRU(units=128, activation='tanh', return_sequences=False)(gru_branch)
                gru_branch = LayerNormalization()(gru_branch)
                gru_branch = Dropout(0.2)(gru_branch)
            branches.append(gru_branch)

        # Transformer Branch
        if self.transformer_model:
            transformer_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='transformer_input')
            if self.multi_inputs:
                inputs.append(transformer_input)
            logger.info(f"Input shape transformer: {transformer_input.shape}")
            transformer_branch = transformer_input
            # Downsample sequence length to reduce memory usage.
            transformer_branch = MaxPooling1D(pool_size=16, padding='same')(transformer_branch)
            num_transformer_blocks = hp.get('num_transformer_blocks') if self.tunemode else 1
            for i in range(num_transformer_blocks):
                transformer_branch = self.transformer_block(transformer_branch, hp, i, hp.get(f'key_dim_{i}'))
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)
            transformer_branch = Dense(64, activation=hp.get('dense_1_activation') if self.tunemode else 'relu')(transformer_branch)
            transformer_branch = Dropout(0.2)(transformer_branch)
            branches.append(transformer_branch)

        # Concatenate branches if multiple are enabled
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        dense_1 = Dense(
            units=hp.get('dense_1_units'),
            activation=hp.get('dense_1_activation') if self.tunemode else 'relu',
            kernel_regularizer=tf.keras.regularizers.l2(hp.get('l2_reg'))
        )(concatenated)
        dense_dropout = Dropout(0.2)(dense_1)
        output = Dense(1, activation="sigmoid")(dense_dropout)

        logger.info(f"Including {len(inputs)} input(s) and {len(branches)} branch(es).")
        if len(inputs) == 1:
            logger.info(f"Input shape: {inputs[0].shape}")
            model = Model(inputs=inputs, outputs=output)
        else:
            logger.info("Input shapes: " + ", ".join(str(inp.shape) for inp in inputs))
            model = Model(inputs=inputs, outputs=output)

        if len(branches) == 1:
            logger.info(f"Branch shape: {branches[0].shape}")
        else:
            logger.info("Branch shapes: " + ", ".join(str(branch.shape) for branch in branches))
        logger.info(f"Output shape: {output.shape}")
        logger.info("Model built and compiled.")

        if self.tunemode:
            loss_fn   = hp.get('loss')
            metric_fn = hp.get('metric')
            optimizer = self.get_optimizer(hp.get('optimizer'), hp.get('learning_rate'))
        else:
            loss_fn   = 'mse'
            metric_fn = 'mse'
            optimizer = self.get_optimizer('adam', 1e-3)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            steps_per_execution=50,
            metrics=[metric_fn]
        )

        if self.modelsummary:
            model.summary()
        return model

    def get_optimizer(self, optimizer_name, learning_rate):
        optimizers = {
            'adam':    Adam,
            'rmsprop': tf.keras.optimizers.RMSprop,
            'sgd':     tf.keras.optimizers.SGD,
            'nadam':   tf.keras.optimizers.Nadam,
            'adadelta':tf.keras.optimizers.Adadelta,
            'adagrad': tf.keras.optimizers.Adagrad,
            'adamax':  tf.keras.optimizers.Adamax
        }
        optimizer_class = optimizers.get(optimizer_name, Adam)
        return optimizer_class(learning_rate=learning_rate)

    def get_callbacks(self):
        checkpoint_filepath = self.checkpoint_filepath
        if checkpoint_filepath and isinstance(checkpoint_filepath, pathlib.Path):
            checkpoint_filepath = str(checkpoint_filepath)
        if checkpoint_filepath and not checkpoint_filepath.endswith('.keras'):
            checkpoint_filepath = self.modelname
        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=self.chk_verbosity, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=self.save_best_only, verbose=self.chk_verbosity) if checkpoint_filepath else None,
            TensorBoard(log_dir=os.path.join(self.modeldatapath, 'tboard_logs')),
            ReduceLROnPlateau(monitor=self.objective, factor=0.1, patience=self.chk_patience, min_lr=1e-6, verbose=self.chk_verbosity)
        ]

    def transformer_block(self, inputs, hp, block_num, dim):
        key_dim = hp.get(f'key_dim_{block_num}')
        num_heads = hp.get(f'num_heads_{block_num}')
        ff_dim = hp.get(f'ff_dim_{block_num}')
        activation = hp.get(f'transformer_activation_{block_num}')
        attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attn_out = LayerNormalization()(inputs + attn_out)
        ffn_out = Dense(ff_dim, activation=activation)(attn_out)
        ffn_out = Dropout(0.2)(ffn_out)
        ffn_out = Dense(inputs.shape[-1])(ffn_out)
        ffn_out = LayerNormalization()(attn_out + ffn_out)
        ffn_out = Dropout(0.2)(ffn_out)
        return ffn_out

    def run_search(self):
        if self.tuner is None:
            logger.error("Tuner not initialized. Aborting search.")
            return
        logger.info("Running tuner search...")
        try:
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                epochs=self.max_epochs,
                verbose=self.chk_verbosity,
                callbacks=self.get_callbacks()
            )
            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps:
                raise ValueError("No hyperparameters found. Ensure tuning has been run successfully.")
            logger.info(f"Best hyperparameters: {best_hps[0].values}")
            
            # If tunemodeepochs is enabled, retrieve the best epoch value and update the mltune overrides
            if self.tunemodeepochs:
                best_epochs = best_hps[0].values.get('epochs', self.min_epochs)
                logger.info(f"Best epochs from tuning: {best_epochs}")
                if 'mltune' in self.hypermodel_params:
                    self.hypermodel_params['mltune']['epochs'] = best_epochs
                    logger.info("Updated mltune overrides with best epochs value.")
        except Exception as e:
            logger.error(f"Error during tuning: {e}")

    @tf.function
    def _predict_graph(self, model, test_data):
        # Predict on one batch within a tf.function for performance.
        return model(test_data, training=False)

    def run_prediction(self, test_data, batch_size=None):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            # If test_data is a tf.data.Dataset, ensure it's mapped with proper casting
            if isinstance(test_data, tf.data.Dataset):
                test_data = test_data.map(self.cast_to_float32, num_parallel_calls=tf.data.AUTOTUNE)
            # Use the compiled model's predict method (or the _predict_graph if preferred)
            predictions = best_model.predict(test_data, batch_size=batch_size or self.batch_size)
            return predictions
        except IndexError:
            logger.info("No models found. Ensure tuning has been run successfully.")
        except Exception as e:
            logger.info(f"Error during prediction: {e}")
            return None

    def export_best_model(self, ftype='tf'):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.project_dir)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            logger.info(f"Exporting best model to {export_path}")
            if ftype == 'h5':
                best_model.save(export_path + self.modelname + '.h5')
                logger.info(f"Model saved to {export_path}{self.modelname}.h5")
            else:
                best_model.save(export_path + self.modelname + '.keras')
                logger.info(f"Model saved to {export_path}{self.modelname}.keras")
        except IndexError:
            logger.info("No models found to export.")
        except Exception as e:
            logger.info(f"Error saving the model: {e}")

    def check_and_load_model(self, lpbase_path, ftype='tf'):
        logger.info(f"Checking for model file at base_path {lpbase_path}")
        logger.info(f"Model name: {self.modelname}")
        if ftype == 'h5':
            localmodel = self.modelname + '.h5'
            model_path = os.path.join(lpbase_path, localmodel)
            logger.info(f"Model path h5 : {model_path}")
        if ftype == 'tf':
            localmodel = self.modelname + '.keras'
            model_path = os.path.join(lpbase_path, localmodel)
            logger.info(f"Model path keras : {model_path}")

        try:
            if os.path.exists(model_path) and (model_path.endswith('.h5') or model_path.endswith('.keras') or os.path.isdir(model_path)):
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
