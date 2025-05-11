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
import uuid  # Ensure uuid is imported for use in get_callbacks
# Machine Learning packages
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_DISABLE_POOL_ALLOCATOR"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from datetime import date
import numpy as np
from keras_tuner.engine.hyperparameters import HyperParameters
from tsMqlMLTuner.tsMqlMLOracleServer import OracleServer
from tsMqlMLTuner.tsMqlMLOracleClient import OracleClient

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

mixed_precision.set_global_policy('mixed_float16')
import gc
gc.collect()



class CMdtuner:
    def __init__(self, **kwargs):
        # Extract hypermodel parameters
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        logger.info(f"Hypermodel parameters: {self.hypermodel_params}")

        self.oracle = kwargs.get("oracle", None)

        base = self.hypermodel_params.get('base', {})
        self.mp_pl_platform_base       = base.get('mp_glob_base_platform_dir', None)
        self.checkpoint_filepath       = base.get('mp_glob_base_ml_checkpoint_filepath', None)
        self.modeldatapath             = base.get('mp_glob_sub_ml_src_modeldata', None)
        # Model path
        self.base_path                 = base.get('mp_glob_base_path', None)
        self.project_dir               = base.get('mp_glob_base_ml_project_dir', None)
        self.baseuniq                  = base.get('mp_glob_sub_ml_baseuniq', None)
        self.modelname                 = base.get('mp_glob_sub_ml_model_name', None)
        self.modelpath                 = os.path.join(self.project_dir, self.modelname)


        logger.info(f"TuneParams: mp_pl_platform_base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: checkpoint_filepath: {self.checkpoint_filepath}")
        logger.info(f"TuneParams: base_path          : {self.base_path}")
        logger.info(f"TuneParams: project_dir        : {self.project_dir}")
        logger.info(f"TuneParams: baseuniq           : {self.baseuniq}")
        logger.info(f"TuneParams: modeldatapath      : {self.modeldatapath}")
        logger.info(f"TuneParams: modelname          : {self.modelname}")
        logger.info(f"TuneParams: modelpath          : {self.modelpath}")
       
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

        # Ensure input_width and shift have valid numeric values
        self.input_width = mltune.get('input_width', 24)
        self.shift = mltune.get('shift', 24)

        # Final fallback in case they are explicitly None
        if self.input_width is None:
            self.input_width = 24
        if self.shift is None:
            self.shift = 24

        self.total_window_size = self.input_width + self.shift

        self.tune_new_entries       = mltune.get('tune_new_entries', True)
        self.allow_new_entries       = mltune.get('allow_new_entries', True)
        self.max_retries_per_trial   = mltune.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = mltune.get('max_consecutive_failed_trials', 3)
        self.executions_per_trial    = mltune.get('executions_per_trial', 1)
        self.overwrite               = mltune.get('overwrite', False)

        logger.info(f"Tuning parameters: today            : {self.today}")
        logger.info(f"Tuning parameters: seed             : {self.seed}")
        logger.info(f"Tuning parameters: tunemode         : {self.tunemode}")
        logger.info(f"Tuning parameters: tunemodeepochs   : {self.tunemodeepochs}")
        logger.info(f"Tuning parameters: batch_size       : {self.batch_size}")
        logger.info(f"Tuning parameters: epochs           : {self.epochs}")
        logger.info(f"Tuning parameters: num_trials       : {self.num_trials}")
        logger.info(f"Tuning parameters: max_epochs       : {self.max_epochs}")
        logger.info(f"Tuning parameters: min_epochs       : {self.min_epochs}")
        logger.info(f"Tuning parameters: hyperband_iterations: {self.hyperband_iterations}")
        logger.info(f"Tuning parameters: factor           : {self.factor}")
        logger.info(f"Tuning parameters: objective        : {self.objective}")
        logger.info(f"Tuning parameters: input_shape      : {self.input_shape}")
        logger.info(f"Tuning parameters: data_input_shape : {self.data_input_shape}")
        logger.info(f"Tuning parameters: multi_inputs     : {self.multi_inputs}")
        logger.info(f"Tuning parameters: multi_branches   : {self.multi_branches}")
        logger.info(f"Tuning parameters: multi_outputs    : {self.multi_outputs}")
        logger.info(f"Tuning parameters: label_columns    : {self.label_columns}")
        logger.info(f"Tuning parameters: shift           : {self.shift}")
        logger.info(f"Tuning parameters: input_width     : {self.input_width}")
        logger.info(f"Tuning parameters: total_window_size: {self.total_window_size}")
        logger.info(f"Tuning parameters: tune_new_entries : {self.tune_new_entries}")
        logger.info(f"Tuning parameters: allow_new_entries: {self.allow_new_entries}")
        logger.info(f"Tuning parameters: max_retries_per_trial: {self.max_retries_per_trial}")
        logger.info(f"Tuning parameters: max_consecutive_failed_trials: {self.max_consecutive_failed_trials}")
        logger.info(f"Tuning parameters: executions_per_trial: {self.executions_per_trial}")
        logger.info(f"Tuning parameters: overwrite        : {self.overwrite}")


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

        #Threading parameters
        self.use_multiprocessing = mltune.get('use_multiprocessing', True)
        self.workers = mltune.get('workers', 32)
      
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
        logger.info(f"Tuning parameters: 'use_multiprocessing': {self.use_multiprocessing}")
        logger.info(f"Tuning parameters: 'workers': {self.workers}")

       

        # Checkpoint parameters  
        self.checkpoint_dir = self.checkpoint_filepath
        self.overwrite = mltune.get('overwrite', False)
        self.chk_fullmodel = mltune.get('chk_fullmodel', True)
        self.chk_verbosity = mltune.get('chk_verbosity', 1)
        self.chk_mode = mltune.get('chk_mode', 'min')
        self.chk_monitor = mltune.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = mltune.get('chk_sav_freq', 'epoch')
        self.chk_patience = mltune.get('chk_patience', 10)
        self.save_best_only = mltune.get('save_best_only', True)

        logger.info(f"Checkpoint parameters: checkpoint_dir: {self.checkpoint_dir}")
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

       

        self.prepare_shapes()

        self.cnn_model         = mltune.get('cnn_model', True)
        self.lstm_model        = mltune.get('lstm_model', True)
        self.gru_model         = mltune.get('gru_model', True)
        self.transformer_model = mltune.get('transformer_model', True)
        self.multiactivate     = mltune.get('multiactivate', True)
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

        # run commands
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


    @property
    def hypermodel(self):
        return self.build_model


    def enable_debugging(self, kwargs):
        if kwargs.get('tf1', False):
            tf.debugging.set_log_device_placement(True)
        if kwargs.get('tf2', False):
            tf.debugging.enable_check_numerics()

    def prepare_shapes(self):
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")
        # Normalize 4D input shape (batch, time, features, channels) to 3D (time, features)
        if len(self.data_input_shape) == 4:
            self.data_input_shape = self.data_input_shape[1:3]  # Remove batch and channel dimensions
            logger.info(f"Adjusted 4D data input shape to 3D: {self.data_input_shape}")
        elif len(self.data_input_shape) == 2:
            self.data_input_shape = (*self.data_input_shape, 1)
        elif len(self.data_input_shape) == 3:
            self.data_input_shape = self.data_input_shape[1:]  # Remove batch dimension if present
            logger.info(f"Adjusted 3D data input shape: {self.data_input_shape}")
        self.main_input_shape = self.get_shape(self.data_input_shape)


    @staticmethod
    def get_shape(data_shape):
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D or 3D.")
        return tuple(data_shape)


    def initialize_tuner(self):
        hp = kt.HyperParameters()
        logger.info(f"Initializing tuner with hyperparameters... {hp}")

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
            hp.Int('num_cnn_layers', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'cnn_filters_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Int(f'cnn_kernel_size_{i}', min_value=2, max_value=5, step=1, default=3)
                hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])

            hp.Int('num_lstm_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'lstm_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Choice(f'lstm_activation_{i}', ['tanh', 'relu'])

            hp.Int('num_gru_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'gru_units_{i}', min_value=self.unitmin, max_value=self.unitmax, step=self.unitstep, default=self.defaultunits)
                hp.Choice(f'gru_activation_{i}', ['tanh', 'relu'])

            hp.Int('num_transformer_blocks', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'key_dim_{i}', min_value=self.trans_dim_min, max_value=self.trans_dim_max, step=self.trans_dim_step, default=self.trans_dim_default)
                hp.Int(f'num_heads_{i}', min_value=self.trans_heads_min, max_value=self.trans_heads_max, step=self.trans_heads_step, default=self.trans_heads_min)
                hp.Int(f'ff_dim_{i}', min_value=self.trans_ff_min, max_value=self.trans_ff_max, step=self.trans_ff_step, default=self.trans_ff_min)
                hp.Choice(f'transformer_activation_{i}', ['relu', 'tanh', 'gelu'])
        else:
            hp.Fixed('cnn_filters', 3)
            hp.Fixed('cnn_kernel_size', 3)

        hp.Int('dense_1_units', min_value=self.dense_units_min, max_value=self.dense_units_max, step=self.dense_units_step)

        tuner_classes = {
            'random': kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian': kt.BayesianOptimization
        }

        if self.tunemode not in tuner_classes:
            logger.error(f"Unsupported tuner type: {self.tunemode}")
            self.tuner = None
            return

        try:
            tuner_args = {
                "hypermodel": self.build_model,
                "hyperparameters": hp,
                "objective": self.objective,
                "directory": self.project_dir,
                "project_name": self.modelname,
                "overwrite": self.overwrite,
                "tune_new_entries": self.tune_new_entries,
                "allow_new_entries": self.allow_new_entries,
                "max_retries_per_trial": self.max_retries_per_trial,
                "max_consecutive_failed_trials": self.max_consecutive_failed_trials,
                "executions_per_trial": self.executions_per_trial,
            }

            if self.tunemode == 'hyperband':
                tuner_args.update({
                    "max_epochs": self.max_epochs,
                    "factor": self.factor,
                    "hyperband_iterations": self.hyperband_iterations
                })

            logger.info(f"Tuner arguments: {tuner_args}")
            self.tuner = tuner_classes[self.tunemode](**tuner_args)
            logger.info(f"Tuner initialized: {self.tunemode}")
            self.tuner._save_model = lambda: None  # Disable weight-saving to avoid file lock issues on Windows
            logger.info(f"Tuner model save method overridden to avoid file lock issues on Windows.")
            self.tuner.search_space_summary()

        except Exception as e:
            logger.error(f"Error initializing tuner: {e}")
            self.tuner = None


    def build_model(self, hp):
        shared_input = Input(shape=self.main_input_shape, name='shared_input')
        inputs = [] if self.multi_inputs else [shared_input]
        branches = []
        logger.info(f"Shared input shape: {shared_input.shape}, multi_inputs: {self.multi_inputs}")

        from tensorflow.keras.backend import int_shape

        # CNN Branch
        if self.cnn_model:
            cnn_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='cnn_input')
            if self.multi_inputs:
                inputs.append(cnn_input)
            cnn_branch = cnn_input

            shape = int_shape(cnn_branch)
            if len(shape) == 4:
                cnn_branch = Reshape((shape[1], shape[2]))(cnn_branch)
            elif len(shape) == 2:
                cnn_branch = Reshape((shape[1], 1))(cnn_branch)
            elif len(shape) == 3 and shape[-1] != 1:
                cnn_branch = Dense(1)(cnn_branch)

            if self.tunemode:
                for i in range(hp.values.get('num_cnn_layers')):
                    cnn_branch = Conv1D(
                        filters=hp.get(f'cnn_filters_{i}'),
                        kernel_size=hp.get(f'cnn_kernel_size_{i}'),
                        activation=hp.values.get(f'cnn_activation_{i}'),
                        padding='same'
                    )(cnn_branch)
                    cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
                    cnn_branch = LayerNormalization()(cnn_branch)
                    cnn_branch = Dropout(0.2)(cnn_branch)
            else:
                cnn_branch = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(cnn_branch)
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
            lstm_branch = lstm_input

            shape = int_shape(lstm_branch)
            if len(shape) == 4:
                lstm_branch = Reshape((shape[1], shape[2]))(lstm_branch)

            for i in range(hp.values.get('num_lstm_layers')):
                lstm_branch = LSTM(
                    units=hp.get(f'lstm_units_{i}'), activation=hp.get(f'lstm_activation_{i}'),
                    return_sequences=(i < hp.values.get('num_lstm_layers') - 1)
                )(lstm_branch)
                lstm_branch = LayerNormalization()(lstm_branch)
                lstm_branch = Dropout(0.2)(lstm_branch)

            branches.append(lstm_branch)

        # GRU Branch
        if self.gru_model:
            gru_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='gru_input')
            if self.multi_inputs:
                inputs.append(gru_input)
            gru_branch = gru_input

            shape = int_shape(gru_branch)
            if len(shape) == 4:
                gru_branch = Reshape((shape[1], shape[2]))(gru_branch)

            for i in range(hp.values.get('num_gru_layers')):
                gru_branch = GRU(
                    units=hp.get(f'gru_units_{i}'), activation=hp.get(f'gru_activation_{i}'),
                    return_sequences=(i < hp.values.get('num_gru_layers') - 1)
                )(gru_branch)
                gru_branch = LayerNormalization()(gru_branch)
                gru_branch = Dropout(0.2)(gru_branch)

            branches.append(gru_branch)

        # Transformer Branch
        if self.transformer_model:
            transformer_input = shared_input if not self.multi_inputs else Input(shape=self.main_input_shape, name='transformer_input')
            if self.multi_inputs:
                inputs.append(transformer_input)
            transformer_branch = transformer_input

            shape = int_shape(transformer_branch)
            if len(shape) == 4:
                transformer_branch = Reshape((shape[1], shape[2]))(transformer_branch)

            key_dim = hp.values.get('key_dim_0')
            num_heads = hp.values.get('num_heads_0')
            projected_dim = key_dim * num_heads

            transformer_branch = Dense(projected_dim)(transformer_branch)
            transformer_branch = self.transformer_block(transformer_branch, hp, 0, dim=projected_dim)
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)

            branches.append(transformer_branch)

        # Combine all branches
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        merged = Dense(512, activation='relu')(concatenated)
        dense_1 = Dense(
            units=hp.values.get('dense_1_units'), activation=hp.get('dense_1_activation') if self.tunemode else 'relu', kernel_regularizer=tf.keras.regularizers.l2(hp.values.get('l2_reg'))
        )(merged)
        dense_dropout = Dropout(0.2)(dense_1)
        output = Dense(1, activation='sigmoid')(dense_dropout)

        model = Model(inputs=inputs if self.multi_inputs else inputs[0], outputs=output)
        optimizer = self.get_optimizer(hp.values.get('optimizer'), hp.values.get('learning_rate')) if self.tunemode else Adam(learning_rate=1e-3)
        model.compile(optimizer=optimizer, loss=hp.values.get('loss') if self.tunemode else 'mse', metrics=[hp.get('metric') if self.tunemode else 'mse'])

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
        logger.info(f"Checkpoint filepath: {checkpoint_filepath}")
        if checkpoint_filepath and isinstance(checkpoint_filepath, pathlib.Path):
            checkpoint_filepath = str(checkpoint_filepath)
            logger.info(f"Converted checkpoint filepath to string: {checkpoint_filepath}")

        tuner_id = os.environ.get("TUNER_ID", f"worker_{uuid.uuid4().hex[:6]}")
        checkpoint_filepath = os.path.join(self.modeldatapath, f"{self.modelname}_{tuner_id}.keras")

        callbacks = [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=self.chk_verbosity, restore_best_weights=True),
            TensorBoard(log_dir=os.path.join(self.modeldatapath, 'tboard_logs')),
            ReduceLROnPlateau(monitor=self.objective, factor=0.1, patience=self.chk_patience, min_lr=1e-6, verbose=self.chk_verbosity)
        ]

        if tuner_id.lower() == "chief":
            callbacks.insert(1, ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=self.save_best_only, verbose=self.chk_verbosity))
        else:
            logger.info(f"Skipping ModelCheckpoint on worker: {tuner_id}")

        return callbacks

    def export_best_model(self, ftype='tf'):
        try:
            tuner_id = os.environ.get("TUNER_ID", "worker")
            if tuner_id.lower() != "chief":
                logger.info(f"Skipping export_best_model: not chief (TUNER_ID={tuner_id})")
                return

            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.project_dir, self.modelname)
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            logger.info(f"Exporting best model to {export_path}")
            if ftype == 'h5':
                export_filepath = export_path + '.h5'
                best_model.save(export_filepath)
                logger.info(f"Model saved to {export_filepath}")
            else:
                export_filepath = export_path + '.keras'
                best_model.save(export_filepath)
                logger.info(f"Model saved to {export_filepath}")
        except IndexError:
            logger.info("No models found to export.")
        except Exception as e:
            logger.info(f"Error saving the model: {e}")

 
    def transformer_block(self, x, hp, i=0, dim=None, use_positional_encoding=True):
        """Single transformer encoder block."""
        key_dim = hp.values.get(f'key_dim_{i}', 64)
        num_heads = hp.values.get(f'num_heads_{i}', 4)
        projected_dim = dim or (key_dim * num_heads)

        # Optional: Apply positional encoding safely
        if use_positional_encoding:
            x = AddPositionalEncoding(projected_dim)(x)

        # Multi-head attention block
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
        attn_output = tf.keras.layers.Dropout(0.1)(attn_output)
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward block
        ffn_output = tf.keras.layers.Dense(projected_dim * 2, activation='relu')(out1)
        ffn_output = tf.keras.layers.Dense(projected_dim)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)

        # Final residual connection
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    def run_search(self):
        logger.info("Running custom tuner search via OracleClient...")
        logger.debug(f"run_search: input_shape = {self.input_shape}")
        logger.debug(f"run_search: hypermodel_params keys = {list(self.hypermodel_params.get('mltune', {}).keys())}")

        if not self.oracle or not isinstance(self.oracle, OracleClient):
            raise RuntimeError("OracleClient not initialized in distributed mode.")

        while True:
            try:
                trial = self.oracle.get_trial()

                if not trial:
                    logger.info("No more trials received from OracleServer. Exiting.")
                    break

                trial_id = trial.get("trial_id")
                hp_config = trial.get("hyperparameters", {})

                if not trial_id:
                    logger.warning("Received trial without trial_id; skipping.")
                    continue

                if not isinstance(hp_config, dict) or len(hp_config) == 0:
                    logger.warning(f"Trial {trial_id} has empty hyperparameters. Marking as FAILED.")
                    self.oracle.update_trial_status(trial_id, "FAILED")
                    continue

                hp = self.tuner.oracle.hyperparameters.copy()
                hp.values = hp_config

                logger.info(f"Running trial {trial_id} with hyperparameters: {hp_config}")
                val_loss = self.objective(hp)

                logger.info(f"✅ Trial {trial_id} completed. val_loss={val_loss:.5f}")
                self.oracle.report_trial_result(trial_id, val_loss)

            except Exception as e:
                logger.error(f"❌ Exception during trial {trial_id if 'trial_id' in locals() else '[UNKNOWN]'}: {str(e)}")
                if "trial_id" in locals():
                    self.oracle.update_trial_status(trial_id, "FAILED")
                break

        logger.info("Custom tuner search completed.")
        return True


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

    def check_and_load_model(self, lpbase_path, ftype='tf'):
        tuner_id = os.environ.get("TUNER_ID", "worker")
        if tuner_id.lower() != "chief":
            logger.info(f"Skipping check_and_load_model: not chief (TUNER_ID={tuner_id})")
            return None

        logger.info(f"Checking for model file at base_path {lpbase_path}")
        logger.info(f"Model name: {self.modelname}")
        if ftype == 'h5':
            localmodel = self.modelname + '.h5'
            model_path = os.path.join(lpbase_path, localmodel)
            logger.info(f"Model path h5 : {model_path}")
        elif ftype == 'tf':
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

    def get_positional_encoding(self, seq_len, dim):
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, dim, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / dim))
        pe = tf.zeros((seq_len, dim))
        pe[:, 0::2] = tf.math.sin(position * div_term)
        pe[:, 1::2] = tf.math.cos(position * div_term)
        pe = tf.expand_dims(pe, axis=0)
        return tf.cast(pe, dtype=tf.float16 if mixed_precision.global_policy().compute_dtype == 'float16' else tf.float32)

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
