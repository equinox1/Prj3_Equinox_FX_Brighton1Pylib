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

# -- Set up global logging (from tsMqlSetup) --
from tsMqlSetup import CMqlSetup
clientlog_config = CMqlSetup()

# Retrieve global logfile path from environment variable
GLOBAL_LOGFILE_PATH = os.environ.get('GLOBAL_LOGFILE_PATH')
if GLOBAL_LOGFILE_PATH:
    clientlog_config.setup_logging(logfile=GLOBAL_LOGFILE_PATH)
else:
    clientlog_config.setup_logging() # Fallback to default if not provided
    print("WARNING: GLOBAL_LOGFILE_PATH not found in environment for tsMqlMLTunerMod. Using default logging.")

logger = logging.getLogger(__name__) # Get logger for this module
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
# CORRECTED: Changed to relative imports for OracleServer and OracleClient
from .tsMqlMLOracleServer import OracleServer
from .tsMqlMLOracleClient import OracleClient


from tsMqlSetup import CMqlSetup
# Initialize CMqlSetup for the launcher itself, to ensure logging is configured
# and setup_config is defined for any utility functions that might implicitly use it.
# Dynamically determine num_cores and num_threads for optimal performance.
# num_cores: Estimate physical cores. On systems with hyperthreading, this is often
#            half the logical core count (os.cpu_count()). If os.cpu_count() is not available
#            or is 1, default to 1.
# num_threads: Typically 1 per core for numerical workloads to avoid hyperthreading
#              contention, but can be set higher (e.g., 2) if testing proves beneficial.
_logical_cores = os.cpu_count() if os.cpu_count() is not None else 1
_estimated_physical_cores = _logical_cores // 2 if _logical_cores > 1 else 1

setup_config = CMqlSetup(
    loglevel='INFO',
    warn='ignore',
    precision='mixed_bfloat16',
    tfdebug=False,
    num_cores=_estimated_physical_cores,
    num_threads=1
)


from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

app_params = mql_overrides.env.all_params().get("app", {})
global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')


# -- end of logging setup ----
# Platform imports
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk         = run_platform.RunPlatform()
os_platform  = platform_checker.get_platform()
loadmql      = pchk.check_mql_state()


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
from tensorflow.keras.activations import get as get_activation
from keras_tuner.engine.hyperparameters import HyperParameters


import gc
gc.collect()



class CMdtuner:
    def __init__(self, **kwargs):
        # Extract hypermodel parameters
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        logger.info(f"Hypermodel parameters: {self.hypermodel_params}")

        self.oracle = kwargs.get("oracle_client", None) # Use oracle_client from kwargs

        # Get relevant parameters from hypermodel_params
        base_params = self.hypermodel_params.get('base', {})
        app_params = self.hypermodel_params.get('app', {})
        mltune_params = self.hypermodel_params.get('mltune', {})

        # Defensive check: Ensure mltune_params is a dictionary
        if not isinstance(mltune_params, dict):
            logger.error(f"mltune_params is not a dictionary: {type(mltune_params)}. Defaulting to empty dict.")
            mltune_params = {}

        # Determine project_dir: prioritize from base_params, then a sensible default
        self.project_dir = base_params.get('mp_glob_base_ml_project_dir')
        if self.project_dir is None:
            # Fallback to a default directory if not configured
            # Using current working directory + a default folder name
            self.project_dir = os.path.join(os.getcwd(), "tuner_projects")
            logger.warning(f"mp_glob_base_ml_project_dir not found in config. Using default: {self.project_dir}")
        
        # Determine modelname: prioritize from project_name kwarg, then app_params, then a default
        # The 'project_name' kwarg comes from CMdtunerSelector, which gets it from tsNeuroPredictWinMql_chief.py's MODEL_NAME
        self.modelname = kwargs.get('project_name', app_params.get('mp_glob_sub_ml_model_name', 'ts_mql_model'))
        if self.modelname is None: # Double check if it's still None
            self.modelname = 'default_model'
            logger.warning("Model name not found in config. Using default: 'default_model'")

        # Construct modelpath (for saving the final model, often the same as project_dir for tuner)
        self.modelpath = os.path.join(self.project_dir, self.modelname)
        logger.info(f"CMdtuner: Initialized with project_dir='{self.project_dir}' and modelname='{self.modelname}'")
        logger.info(f"CMdtuner: Constructed modelpath: {self.modelpath}")
       
        # Determine modeldatapath for checkpoints and other model-related files
        # Prioritize from base_params, fallback to project_dir
        self.modeldatapath = base_params.get('mp_glob_sub_ml_src_modeldata')
        if self.modeldatapath is None:
            self.modeldatapath = os.path.join(self.project_dir, "model_data")
            logger.warning(f"mp_glob_sub_ml_src_modeldata not found in config. Using default: {self.modeldatapath}")
        
        # Construct checkpoint_filepath
        self.checkpoint_filepath = os.path.join(self.modeldatapath, f"{self.modelname}_checkpoint.keras")
        logger.info(f"CMdtuner: Constructed modeldatapath: {self.modeldatapath}")
        logger.info(f"CMdtuner: Constructed checkpoint_filepath: {self.checkpoint_filepath}")


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
        self.today                   = mltune_params.get('today', '2025-03-16 17:27:46')
        # CORRECTED: Ensure tunemode is stored in lowercase to avoid lookup issues
        self.tunemode                = mltune_params.get('tunemode', 'Hyperband').lower() 
        # Explicitly assign seed to prevent AttributeError
        self.seed                    = mltune_params.get('seed', 42)
        self.tunemodeepochs          = mltune_params.get('tunemodeepochs', True)
        self.batch_size              = mltune_params.get('batch_size', 16)  # Reduced batch size
        self.epochs                  = mltune_params.get('epochs', 2)
        self.num_trials              = mltune_params.get('num_trials', 3)
        self.max_epochs              = mltune_params.get('max_epochs', 120)
        self.min_epochs              = mltune_params.get('min_epochs', 10)
        self.hyperband_iterations    = mltune_params.get('hyperband_iterations', 1)
        self.factor                  = mltune_params.get('factor', 10)
        self.objective               = mltune_params.get('objective', 'val_loss')
        self.input_shape             = kwargs.get('input_shape', None) # Get from kwargs passed by CMdtunerSelector
        # Pass num_classes from kwargs or derive from label_columns
        self.num_classes             = kwargs.get('num_classes', 1) 
        self.data_input_shape        = mltune_params.get('data_input_shape', None)
        self.multi_inputs            = mltune_params.get('multi_inputs', False)
        self.multi_branches          = mltune_params.get('multi_branches', True)
        self.multi_outputs           = mltune_params.get('multi_outputs', False)
        self.label_columns           = mltune_params.get('label_columns', None)
        self.shift                   = mltune_params.get('shift', 24)
        self.input_width             = mltune_params.get('input_width', 1440)

        # Ensure input_width and shift have valid numeric values
        self.input_width = mltune_params.get('input_width', 24)
        self.shift = mltune_params.get('shift', 24)

        # Final fallback in case they are explicitly None
        if self.input_width is None:
            self.input_width = 24
        if self.shift is None:
            self.shift = 24

        self.total_window_size = self.input_width + self.shift

        self.tune_new_entries       = mltune_params.get('tune_new_entries', True)
        self.allow_new_entries       = mltune_params.get('allow_new_entries', True)
        self.max_retries_per_trial   = mltune_params.get('max_retries_per_trial', 5)
        self.max_consecutive_failed_trials = mltune_params.get('max_consecutive_failed_trials', 3)
        self.executions_per_trial    = mltune_params.get('executions_per_trial', 1)
        self.overwrite               = mltune_params.get('overwrite', True)

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

        # Corrected: Derive num_classes if multi_outputs is True and label_columns are provided
        if self.multi_outputs and self.label_columns is not None:
            if isinstance(self.label_columns, (list, tuple)):
                self.num_classes = len(self.label_columns)
                logger.info(f"Updated num_classes based on label_columns: {self.num_classes}")
            else:
                logger.warning(f"multi_outputs is True but label_columns is not a list/tuple. num_classes remains {self.num_classes}")
        
        # New tuning parameters 
        self.unitmin         = mltune_params.get('unitmin', 32)
        self.unitmax         = mltune_params.get('unitmax', 512)
        self.unitstep        = mltune_params.get('unitstep', 32)
        self.defaultunits    = mltune_params.get('defaultunits', 128)
        self.all_modelscale  = mltune_params.get('all_modelscale', 8.0)
        self.cnn_modelscale  = mltune_params.get('cnn_modelscale', 8.0)
        self.lstm_modelscale = mltune_params.get('lstm_modelscale', 8.0)
        self.gru_modelscale  = mltune_params.get('gru_modelscale', 8.0)
        self.trans_modelscale = mltune_params.get('trans_modelscale', 8.0)
        self.transh_modelscale = mltune_params.get('transh_modelscale', 8.0)
        self.transff_modelscale = mltune_params.get('transff_modelscale', 8.0)
        self.dense_modelscale = mltune_params.get('dense_modelscale', 8.0)
        # Ensure division by zero is handled if scale is 0
        self.trans_dim_min      = mltune_params.get('trans_dim_min', 32 // int(self.trans_modelscale) if self.trans_modelscale != 0 else 32)
        self.trans_dim_max      = mltune_params.get('trans_dim_max', 256 // int(self.trans_modelscale) if self.trans_modelscale != 0 else 256)
        self.trans_dim_step     = mltune_params.get('trans_dim_step', 32 // int(self.trans_modelscale) if self.trans_modelscale != 0 else 32)
        self.trans_dim_default  = mltune_params.get('trans_dim_default', 64 // int(self.trans_modelscale) if self.trans_modelscale != 0 else 64)
        
        self.trans_heads_min    = mltune_params.get('trans_heads_min', 2)
        self.trans_heads_max    = mltune_params.get('trans_heads_max', 8)
        self.trans_heads_step   = mltune_params.get('trans_heads_step', 2)
        self.trans_ff_min       = mltune_params.get('trans_ff_min', int(64 // self.transff_modelscale) if self.transff_modelscale != 0 else 64)
        self.trans_ff_max       = mltune_params.get('trans_ff_max', int(512 // self.transff_modelscale) if self.transff_modelscale != 0 else 512)
        self.trans_ff_step      = mltune_params.get('trans_ff_step', int(64 // self.transff_modelscale) if self.transff_modelscale != 0 else 64)
        self.dense_units_min    = mltune_params.get('dense_units_min', int(32 // self.dense_modelscale) if self.dense_modelscale != 0 else 32)
        self.dense_units_max    = mltune_params.get('dense_units_max', int(128 // self.dense_modelscale) if self.dense_modelscale != 0 else 128)
        self.dense_units_step   = mltune_params.get('dense_units_step', int(32 // self.dense_modelscale) if self.dense_modelscale != 0 else 32)

        #Threading parameters - Removed as they are not expected by Keras Tuner's internal model.fit
        # self.use_multiprocessing = mltune_params.get('use_multiprocessing', True)
        # self.workers = mltune_params.get('workers', 32)
      
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
        # logger.info(f"Tuning parameters: 'use_multiprocessing': {self.use_multiprocessing}") # Removed
        # logger.info(f"Tuning parameters: 'workers': {self.workers}") # Removed

       

        # Checkpoint parameters  
        self.checkpoint_dir = os.path.dirname(self.checkpoint_filepath) # Use the directory of the checkpoint file
        self.overwrite = mltune_params.get('overwrite', True)
        self.chk_fullmodel = mltune_params.get('chk_fullmodel', True)
        self.chk_verbosity = mltune_params.get('chk_verbosity', 1)
        self.chk_mode = mltune_params.get('chk_mode', 'min')
        self.chk_monitor = mltune_params.get('chk_monitor', 'val_loss')
        self.chk_sav_freq = mltune_params.get('chk_sav_freq', 'epoch')
        self.chk_patience = mltune_params.get('chk_patience', 10)
        self.save_best_only = mltune_params.get('save_best_only', True)

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
        self.traindataset = kwargs.get('train_dataset') # Corrected to train_dataset
        self.valdataset   = kwargs.get('val_dataset')   # Corrected to val_dataset
        self.testdataset  = kwargs.get('test_dataset')  # Corrected to test_dataset (if used)
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

        self.cnn_model         = mltune_params.get('cnn_model', True)
        self.lstm_model        = mltune_params.get('lstm_model', True)
        self.gru_model         = mltune_params.get('gru_model', True)
        self.transformer_model = mltune_params.get('transformer_model', True)
        self.multiactivate     = mltune_params.get('multiactivate', True)
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
        if not self.input_shape: # Use self.input_shape which is passed from CMdtunerSelector
            raise ValueError("Input shape must be specified.")
        # Normalize 4D input shape (batch, time, features, channels) to 3D (time, features)
        # Assuming input_shape from kwargs is already (n_steps, n_features) or (n_steps, n_features, 1)
        if len(self.input_shape) == 3 and self.input_shape[-1] == 1:
            self.main_input_shape = self.input_shape[:2] # Remove the last dimension if it's 1
            logger.info(f"Adjusted 3D input shape (with channel 1) to 2D for model building: {self.main_input_shape}")
        elif len(self.input_shape) == 2:
            self.main_input_shape = self.input_shape
            logger.info(f"Using 2D input shape directly: {self.main_input_shape}")
        else:
            raise ValueError(f"Unsupported input shape: {self.input_shape}. Expected (time, features) or (time, features, 1).")
        
        logger.info(f"Main input shape for Keras model: {self.main_input_shape}")


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
        hp.Choice('metric', ['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity']) # Added kld and cosine_similarity
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

        # The self.tunemode is already converted to lowercase in __init__
        if self.tunemode not in tuner_classes: 
            logger.error(f"Unsupported tuner type: {self.tunemode}")
            self.tuner = None
            return

        try:
            tuner_args = {
                "hypermodel": self.build_model,
                "hyperparameters": hp,
                "objective": self.objective,
                "directory": self.project_dir, # Use the resolved project_dir
                "project_name": self.modelname, # Use the resolved modelname
                "overwrite": self.overwrite,
                "tune_new_entries": self.tune_new_entries,
                "allow_new_entries": self.allow_new_entries,
                "max_retries_per_trial": self.max_retries_per_trial,
                "max_consecutive_failed_trials": self.max_consecutive_failed_trials,
                "executions_per_trial": self.executions_per_trial,
            }

            if self.tunemode == 'hyperband': # Now self.tunemode is already lowercase
                tuner_args.update({
                    "max_epochs": self.max_epochs,
                    "factor": self.factor,
                    "hyperband_iterations": self.hyperband_iterations
                })

            logger.info(f"Tuner arguments: {tuner_args}")
            self.tuner = tuner_classes[self.tunemode](**tuner_args) # Use self.tunemode directly
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
            # Ensure the input to Conv1D is 3D (batch, steps, features)
            if len(shape) == 4: # (None, steps, features, 1) -> (None, steps, features)
                cnn_branch = Reshape((shape[1], shape[2]))(cnn_branch)
            elif len(shape) == 2: # (None, features) -> (None, features, 1) to make it 3D
                cnn_branch = Reshape((shape[1], 1))(cnn_branch)
            # If len(shape) == 3, it's already (None, steps, features) which is good for Conv1D

            if self.tunemode:
                num_cnn_layers = hp.values.get('num_cnn_layers', 1)
                for i in range(num_cnn_layers):
                    cnn_branch = Conv1D(
                        filters=hp.values.get(f'cnn_filters_{i}', 64),
                        kernel_size=hp.values.get(f'cnn_kernel_size_{i}', 3),
                        activation=hp.values.get(f'cnn_activation_{i}', 'relu'),
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
            if len(shape) == 4: # (None, steps, features, 1) -> (None, steps, features)
                lstm_branch = Reshape((shape[1], shape[2]))(lstm_branch)
            elif len(shape) == 2: # (None, features) -> (None, features, 1) to make it 3D
                lstm_branch = Reshape((shape[1], 1))(lstm_branch)

            num_lstm_layers = hp.values.get('num_lstm_layers', 1)
            for i in range(num_lstm_layers):
                lstm_branch = LSTM(
                    units=hp.values.get(f'lstm_units_{i}', 64),
                    activation=hp.values.get(f'lstm_activation_{i}', 'tanh'),
                    return_sequences=(i < num_lstm_layers - 1)
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
            if len(shape) == 4: # (None, steps, features, 1) -> (None, steps, features)
                gru_branch = Reshape((shape[1], shape[2]))(gru_branch)
            elif len(shape) == 2: # (None, features) -> (None, features, 1) to make it 3D
                gru_branch = Reshape((shape[1], 1))(gru_branch)

            num_gru_layers = hp.values.get('num_gru_layers', 1)
            for i in range(num_gru_layers):
                gru_branch = GRU(
                    units=hp.values.get(f'gru_units_{i}', 64),
                    activation=hp.values.get(f'gru_activation_{i}', 'tanh'),
                    return_sequences=(i < num_gru_layers - 1)
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
            if len(shape) == 4: # (None, steps, features, 1) -> (None, steps, features)
                transformer_branch = Reshape((shape[1], shape[2]))(transformer_branch)
            elif len(shape) == 2: # (None, features) -> (None, features, 1) to make it 3D
                transformer_branch = Reshape((shape[1], 1))(transformer_branch)


            key_dim = hp.values.get('key_dim_0', 64)
            num_heads = hp.values.get('num_heads_0', 4)
            projected_dim = key_dim * num_heads

            # Ensure the input to the transformer block has the correct dimension for positional encoding
            # If the input_shape is (steps, features), and features is not equal to projected_dim,
            # we need a Dense layer to project it to projected_dim before positional encoding.
            if self.main_input_shape[-1] != projected_dim:
                transformer_branch = Dense(projected_dim)(transformer_branch)

            transformer_branch = self.transformer_block(transformer_branch, hp, 0, dim=projected_dim)
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)

            branches.append(transformer_branch)

        # Combine branches and dense layers
        concatenated = Concatenate()(branches) if self.multi_branches and len(branches) > 1 else branches[0]
        
        # Ensure the output layer matches the number of classes (regression task)
        # For regression, num_classes is typically 1.
        output_units = self.num_classes if self.num_classes else 1
        # The activation function should be 'linear' for regression, and 'sigmoid' for binary classification.
        # If num_classes > 1 and it's a classification, use 'softmax'.
        # For this context, assuming regression or multi-output regression.
        output_activation = 'linear'
        if output_units > 1: # Could be multi-label binary or multi-class
             # If target is (None, 7) and multi_outputs is true, it's likely a multi-output regression.
             # If it was multi-class classification, it would likely be one-hot encoded and need softmax.
             # Given the problem's context often being time series prediction, linear is usually correct.
             output_activation = 'linear' # For multi-output regression
        elif self.hypermodel_params.get('is_binary_classification', False): # Add a param to distinguish binary
            output_activation = 'sigmoid' # For single-output binary classification

        merged = Dense(512, activation='relu')(concatenated)
        dense_1 = Dense(
            units=hp.values.get('dense_1_units', 64),
            activation=get_activation(hp.values.get('dense_1_activation', 'relu')),
            kernel_regularizer=tf.keras.regularizers.l2(hp.values.get('l2_reg', 1e-4))
        )(merged)
        dense_dropout = Dropout(0.2)(dense_1)
        # IMPORTANT FIX: Set the units of the final Dense layer to self.num_classes
        # This resolves the `target.shape=(None, 7), output.shape=(None, 1)` mismatch.
        output = Dense(self.num_classes, activation=output_activation)(dense_dropout)

        model = Model(inputs=inputs if self.multi_inputs else inputs[0], outputs=output)

        # Final compile
        optimizer = self.get_optimizer(hp.values.get('optimizer', 'adam'), hp.values.get('learning_rate', 1e-3))
        
        # Resolve the loss function
        loss_str = hp.values.get('loss', 'mse')
        try:
            if loss_str.lower() in ["mse", "mean_squared_error"]:
                resolved_loss = tf.keras.losses.MeanSquaredError()
            elif loss_str.lower() in ["mae", "mean_absolute_error"]:
                resolved_loss = tf.keras.losses.MeanAbsoluteError()
            elif loss_str.lower() in ["binary_crossentropy"]:
                # Ensure BinaryCrossentropy is appropriate for multi-output regression,
                # or adjust activation if it's binary classification.
                # If target is (None, 7) and loss is binary_crossentropy, it implies 7 independent binary predictions.
                # If it's multi-output regression, linear activation and MSE/MAE are more typical.
                # For now, keep it as is, assuming a multi-label binary classification or special case.
                resolved_loss = tf.keras.losses.BinaryCrossentropy()
            elif loss_str.lower() in ["mape", "mean_absolute_percentage_error"]:
                resolved_loss = tf.keras.losses.MeanAbsolutePercentageError()
            elif loss_str.lower() in ["msle", "mean_squared_logarithmic_error"]:
                resolved_loss = tf.keras.losses.MeanSquaredLogarithmicError()
            elif loss_str.lower() == "poisson":
                resolved_loss = tf.keras.losses.Poisson()
            elif loss_str.lower() in ["kld", "kl_divergence"]:
                resolved_loss = tf.keras.losses.KLDivergence()
            elif loss_str.lower() in ["cosine_similarity"]:
                resolved_loss = tf.keras.losses.CosineSimilarity()
            else:
                resolved_loss = tf.keras.losses.get(loss_str)
                if isinstance(resolved_loss, type):
                    resolved_loss = resolved_loss()
        except Exception as e:
            logger.error(f"Error resolving loss '{loss_str}' in build_model: {e}")
            resolved_loss = tf.keras.losses.MeanSquaredError() # Fallback

        # Resolve the metric function
        metric_str = hp.values.get('metric', 'mse')
        try:
            if metric_str.lower() in ["mse", "mean_squared_error"]:
                resolved_metric = tf.keras.metrics.MeanSquaredError()
            elif metric_str.lower() in ["mae", "mean_absolute_error"]:
                resolved_metric = tf.keras.metrics.MeanAbsoluteError()
            elif metric_str.lower() in ["accuracy"]:
                resolved_metric = tf.keras.metrics.Accuracy()
            elif metric_str.lower() in ["mape", "mean_absolute_percentage_error"]:
                resolved_metric = tf.keras.metrics.MeanAbsolutePercentageError()
            elif metric_str.lower() in ["msle", "mean_squared_logarithmic_error"]:
                resolved_metric = tf.keras.metrics.MeanSquaredLogarithmicError()
            elif metric_str.lower() == "poisson":
                resolved_metric = tf.keras.metrics.Poisson()
            elif metric_str.lower() in ["kld", "kl_divergence"]:
                resolved_metric = tf.keras.metrics.KLDivergence()
            elif metric_str.lower() in ["cosine_similarity"]:
                resolved_metric = tf.keras.metrics.CosineSimilarity()
            else:
                resolved_metric = tf.keras.metrics.get(metric_str)
                if isinstance(resolved_metric, type):
                    resolved_metric = resolved_metric()
        except Exception as e:
            logger.error(f"Error resolving metric '{metric_str}' in build_model: {e}")
            resolved_metric = tf.keras.metrics.MeanSquaredError() # Fallback

        model.compile(
            optimizer=optimizer,
            loss=resolved_loss,
            metrics=[resolved_metric]
        )

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
        optimizer_class = optimizers.get(optimizer_name.lower(), Adam)
        logger.debug(f"Using optimizer: {optimizer_name} → {optimizer_class}")
        return optimizer_class(learning_rate=learning_rate)

    def get_callbacks(self):
        tuner_id = os.environ.get("TUNER_ID", f"worker_{uuid.uuid4().hex[:6]}")
        # Use self.checkpoint_filepath which is already constructed in __init__
        checkpoint_filepath = self.checkpoint_filepath 

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor=self.chk_monitor, patience=self.chk_patience,
                                            verbose=self.chk_verbosity, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.modeldatapath, 'tboard_logs')),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=self.chk_monitor, factor=0.1,
                                                patience=self.chk_patience, min_lr=1e-6,
                                                verbose=self.chk_verbosity)
        ]

        if tuner_id.lower() == "chief":
            # Ensure the directory for the checkpoint exists
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            callbacks.insert(1, tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_best_only=self.save_best_only,
                                                                verbose=self.chk_verbosity))
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
            # Use self.modeldatapath for export path
            export_path = self.modeldatapath
            os.makedirs(os.path.dirname(export_path), exist_ok=True) # Ensure directory exists
            logger.info(f"Exporting best model to {export_path}")
            if ftype == 'h5':
                export_filepath = os.path.join(export_path, f"{self.modelname}.h5")
                best_model.save(export_filepath)
                logger.info(f"Model saved to {export_filepath}")
            else: # Default to .keras format
                export_filepath = os.path.join(export_path, f"{self.modelname}.keras")
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
        attn_output = tf.keras.layers.Dropout(0.2)(attn_output) # Increased dropout for attention
        out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward block
        ffn_output = tf.keras.layers.Dense(projected_dim * 2, activation='relu')(out1)
        ffn_output = tf.keras.layers.Dense(projected_dim)(ffn_output)
        ffn_output = tf.keras.layers.Dropout(0.2)(ffn_output) # Increased dropout for FFN

        # Final residual connection
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
    
   
    def _objective(self, hp):
        """Objective function for evaluating a trial's performance."""

        trial_id = os.environ.get("KERASTUNER_TRIAL_ID", "UNKNOWN")
        logger.info(f"[OBJECTIVE] Starting trial {trial_id} with hyperparameters: {hp.values}")

        # Resolve loss
        loss_str = hp.values.get("loss", "mse")
        try:
            if loss_str.lower() in ["mse", "mean_squared_error"]:
                self.loss = tf.keras.losses.MeanSquaredError()
            elif loss_str.lower() in ["mae", "mean_absolute_error"]:
                self.loss = tf.keras.losses.MeanAbsoluteError()
            elif loss_str.lower() in ["binary_crossentropy"]:
                self.loss = tf.keras.losses.BinaryCrossentropy()
            elif loss_str.lower() in ["mape", "mean_absolute_percentage_error"]:
                self.loss = tf.keras.losses.MeanAbsolutePercentageError()
            elif loss_str.lower() in ["msle", "mean_squared_logarithmic_error"]:
                self.loss = tf.keras.losses.MeanSquaredLogarithmicError()
            elif loss_str.lower() == "poisson":
                self.loss = tf.keras.losses.Poisson()
            elif loss_str.lower() in ["kld", "kl_divergence"]:
                self.loss = tf.keras.losses.KLDivergence()
            elif loss_str.lower() in ["cosine_similarity"]:
                self.loss = tf.keras.losses.CosineSimilarity()
            else:
                loss = tf.keras.losses.get(loss_str)
                self.loss = loss() if isinstance(loss, type) else loss
        except Exception as e:
            logger.error(f"[OBJECTIVE] Invalid loss '{loss_str}': {e}", exc_info=True)
            return float("inf")

        # Resolve metric
        metric_str = hp.values.get("metric", "mse")
        try:
            if metric_str.lower() in ["mse", "mean_squared_error"]:
                self.metric = tf.keras.metrics.MeanSquaredError()
            elif metric_str.lower() in ["mae", "mean_absolute_error"]:
                self.metric = tf.keras.metrics.MeanAbsoluteError()
            elif metric_str.lower() in ["accuracy"]:
                self.metric = tf.keras.metrics.Accuracy()
            elif metric_str.lower() in ["mape", "mean_absolute_percentage_error"]:
                self.metric = tf.keras.metrics.MeanAbsolutePercentageError()
            elif metric_str.lower() in ["msle", "mean_squared_logarithmic_error"]:
                self.metric = tf.keras.metrics.MeanSquaredLogarithmicError()
            elif metric_str.lower() == "poisson":
                self.metric = tf.keras.metrics.Poisson()
            elif metric_str.lower() in ["kld", "kl_divergence"]:
                self.metric = tf.keras.metrics.KLDivergence()
            elif metric_str.lower() in ["cosine_similarity"]:
                self.metric = tf.keras.metrics.CosineSimilarity()
            else:
                metric = tf.keras.metrics.get(metric_str)
                self.metric = metric() if isinstance(metric, type) else metric
        except Exception as e:
            logger.error(f"[OBJECTIVE] Invalid metric '{metric_str}': {e}", exc_info=True)
            return float("inf")

        logger.info(f"[OBJECTIVE] Using loss={self.loss}, metric={self.metric}")

        # Build and compile model
        try:
            model = self.build_model(hp)
            # When calling model.compile in _objective, ensure it uses the resolved
            # loss and metric objects, not string lookups, to match build_model
            model.compile(
                optimizer=self.get_optimizer(hp.values.get('optimizer', 'adam'), hp.values.get('learning_rate', 1e-3)), # Use hp values for optimizer
                loss=self.loss,
                metrics=[self.metric]
            )
        except Exception as e:
            logger.error(f"[OBJECTIVE] Failed to compile model for trial {trial_id}: {e}", exc_info=True)
            return float("inf")

        # Train model
        try:
            history = model.fit(
                self.traindataset,
                validation_data=self.valdataset,
                epochs=hp.values.get("epochs", 10),
                verbose=0,
                callbacks=self.get_callbacks()
            )
        except Exception as e:
            logger.error(f"[OBJECTIVE] Training failed for trial {trial_id}: {e}", exc_info=True)
            return float("inf")

        # Extract validation loss
        val_loss = history.history.get("val_loss", [None])[-1]
        if val_loss is None:
            logger.warning(f"[OBJECTIVE] Trial {trial_id} produced no val_loss.")
            return float("inf")

        logger.info(f"[OBJECTIVE] Trial {trial_id} completed with val_loss={val_loss:.6f}")
        return float(val_loss)


    def run(self):
        """
        Placeholder run method to orchestrate the tuning process.
        """
        if self.tuner is None:
            logger.error("Tuner is not initialized. Cannot run tuning.")
            return

        logger.info(f"Starting tuning process with mode: {self.tunemode}")
        try:
            # Removed use_multiprocessing and workers from tuner.search() arguments
            # as they are not expected by Keras Tuner's TensorFlowTrainer.
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                epochs=self.epochs, # Use base epochs for tuner.search
                callbacks=self.get_callbacks()
            )
            logger.info("Tuning process completed.")
            self.export_best_model() # Export best model after search
        except Exception as e:
            logger.error(f"Error during tuning process: {e}", exc_info=True)


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
            best_hps = self.tuner.get_best_hyperparameters(num_trials=1)
            if not best_hps:
                logger.info("No best hyperparameters found. Returning None.")
                return None
            return best_hps[0]
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

    def finalize_best_trial(self):
        try:
            best_trials = self.tuner.oracle.get_best_trials(num_trials=1)
            if not best_trials:
                logger.error("❌ No best trial found. Skipping training/export.")
                return None
            best_trial = best_trials[0]
            logger.info(f"✅ Best trial finalized: {best_trial.trial_id}")
            return best_trial
        except Exception as e:
            logger.error(f"❌ Failed to fetch or finalize best trial from OracleClient: {e}")
            return None