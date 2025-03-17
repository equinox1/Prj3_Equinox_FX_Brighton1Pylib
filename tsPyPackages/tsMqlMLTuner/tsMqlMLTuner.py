#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlMLTuner.py
File: tsPyPackages/tsMqlMLTuner/tsMqlMLTuner.py
Description: The Tuner method for the machine learning model.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1 (Updated and Fixed)
License: MIT License
"""

import logging
import os
import pathlib
logger = logging.getLogger(__name__)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk         = run_platform.RunPlatform()
os_platform  = platform_checker.get_platform()
loadmql      = pchk.check_mql_state()
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

from tensorflow.keras.losses import (MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError,
                                     MeanAbsolutePercentageError, MeanSquaredLogarithmicError,
                                     Poisson, KLDivergence, CosineSimilarity)
from tensorflow.keras.metrics import (MSE, MAE, MAPE, MSLE, Poisson, KLDivergence,
                                      CosineSimilarity, Accuracy)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_tuner as kt
import numpy as np
from datetime import date


class CMdtuner:
    def __init__(self, **kwargs):
        # Initialize hypermodel parameters from kwargs
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        logger.info(f"Hypermodel parameters: {self.hypermodel_params}")

        # Set base parameters from hypermodel_params['base']
        base = self.hypermodel_params.get('base', {})
        self.mp_pl_platform_base      = base.get('mp_glob_base_platform_dir', None)
        self.checkpoint_filepath       = base.get('mp_glob_base_ml_checkpoint_filepath', None)
        self.base_path                 = base.get('mp_glob_base_path', None)
        self.project_dir               = base.get('mp_glob_base_ml_project_dir', None)
        self.baseuniq                  = base.get('mp_glob_sub_ml_baseuniq', None)
        self.modeldatapath             = base.get('mp_glob_base_ml_modeldir', None)
        self.modelname                 = base.get('mp_glob_sub_ml_model_name', None)
        self.project_name              = base.get('mp_glob_sub_ml_model_name', None)

        logger.info(f"TuneParams: modeldatapath    : {self.modeldatapath}")
        logger.info(f"TuneParams: mp_pl_platform_base: {self.mp_pl_platform_base}")
        logger.info(f"TuneParams: base_path          : {self.base_path}")
        logger.info(f"TuneParams: project_dir        : {self.project_dir}")
        logger.info(f"TuneParams: baseuniq           : {self.baseuniq}")
        logger.info(f"TuneParams: modelname          : {self.modelname}")
        logger.info(f"TuneParams: checkpoint_filepath: {self.checkpoint_filepath}")

        # Validate required base parameters
        if not self.base_path:
            raise ValueError("The 'mp_glob_base_path' parameter must be provided in hypermodel_params.")
        if not self.project_name:
            raise ValueError("The 'mp_glob_base_ml_project_dir' parameter must be provided in hypermodel_params.")

        # Set data parameters from hypermodel_params['data']
        data = self.hypermodel_params.get('data', {})
        self.mp_data_load       = data.get('mp_data_load', True)
        self.mp_data_save       = data.get('mp_data_save', False)
        self.mp_data_rows       = data.get('mp_data_rows', 1000)
        self.mp_data_rowcount   = data.get('mp_data_rowcount', 10000)
        self.df1_mp_data_filter_int = data.get('df1_mp_data_filter_int', False)
        # (Additional data parameters can be aligned here as needed.)

        # Set ML parameters from hypermodel_params['ml']
        ml = self.hypermodel_params.get('ml', {})
        self.input_keyfeat         = ml.get('mp_ml_input_keyfeat', 'Close')
        self.feature1              = ml.get('feature1', 'Open')
        self.feature2              = ml.get('feature2', 'High')
        self.feature3              = ml.get('feature3', 'Low')
        self.feature4              = ml.get('feature4', 'Close')
        self.feature5              = ml.get('feature5', 'Volume')
        # (Additional ML parameters can be aligned similarly.)

        # Set Tuning parameters from hypermodel_params['mltune']
        mltune = self.hypermodel_params.get('mltune', {})
        self.today                   = mltune.get('today', '2025-03-16 17:27:46')
        self.seed                    = mltune.get('seed', 42)
        self.tunemode                = mltune.get('tunemode', 'Hyperband')
        self.tunemodeepochs          = mltune.get('tunemodeepochs', 100)
        self.batch_size              = mltune.get('batch_size', 32)
        self.epochs                  = mltune.get('epochs', 2)
        self.num_trials              = mltune.get('num_trials', 3)
        self.max_epochs              = mltune.get('max_epochs', 100)
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

        # Logging Tuning parameters
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

        # Datasets and casting mode
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

        if self.traindataset is not None:
            self.traindataset = self.traindataset.map(self.castval)
        if self.valdataset is not None:
            self.valdataset = self.valdataset.map(self.castval)
        if self.testdataset is not None:
            self.testdataset = self.testdataset.map(self.castval)

        # TensorFlow setup: Enable XLA and mixed precision
        tf.config.optimizer.set_jit(True)
        mixed_precision.set_global_policy('mixed_float16')
        self.strategy = tf.distribute.MirroredStrategy()
        logger.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

        # Enable TensorFlow debugging if specified
        self.tf1 = kwargs.get('tf1', False)
        self.tf2 = kwargs.get('tf2', False)
        self.enable_debugging(kwargs)

        # Prepare input shapes (ensure data_input_shape is set and adjusted)
        self.prepare_shapes()

        # Validate that at least one model branch is enabled
        self.cnn_model         = mltune.get('cnn_model', True)
        self.lstm_model        = mltune.get('lstm_model', True)
        self.gru_model         = mltune.get('gru_model', True)
        self.transformer_model = mltune.get('transformer_model', True)
        self.multiactivate     = mltune.get('multiactivate', True)
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

        # Initialize the tuner with hyperparameter definitions
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
        hp = kt.HyperParameters()
        # Define common hyperparameters
        hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd', 'nadam', 'adadelta', 'adagrad', 'adamax', 'ftrl'])
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5])
        hp.Choice('loss', ['binary_crossentropy', 'mse', 'mae', 'mape', 'msle', 'poisson', 'kld', 'cosine_similarity'])
        hp.Choice('dense_1_activation', ['relu', 'tanh'])
        hp.Choice('metric', ['accuracy', 'mae', 'mse', 'mape', 'msle', 'poisson', 'cosine_similarity'])
        hp.Float('l2_reg', min_value=1e-6, max_value=1e-2, sampling='log', default=1e-4)

        # Set epochs tuning based on tunemodeepochs
        if self.tunemodeepochs:
            hp.Int('epochs', min_value=self.min_epochs, max_value=self.max_epochs, step=1)
        else:
            hp.Fixed('epochs', self.min_epochs)

        if self.tunemode:
            # Tuning for CNN branch
            hp.Int('num_cnn_layers', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'cnn_filters_{i}', min_value=32, max_value=512, step=32, default=128)
                hp.Int(f'cnn_kernel_size_{i}', min_value=2, max_value=5, step=1, default=3)
                hp.Choice(f'cnn_activation_{i}', ['relu', 'tanh', 'selu', 'elu', 'linear', 'sigmoid', 'softmax', 'softplus'])
            # Tuning for LSTM branch
            hp.Int('num_lstm_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'lstm_units_{i}', min_value=32, max_value=128, step=32, default=64)
                hp.Choice(f'lstm_activation_{i}', ['tanh', 'relu'])
            # Tuning for GRU branch
            hp.Int('num_gru_layers', min_value=1, max_value=2, default=1)
            for i in range(2):
                hp.Int(f'gru_units_{i}', min_value=32, max_value=128, step=32, default=64)
                hp.Choice(f'gru_activation_{i}', ['tanh', 'relu'])
            # Tuning for Transformer branch
            hp.Int('num_transformer_blocks', min_value=1, max_value=3, default=1)
            for i in range(3):
                hp.Int(f'key_dim_{i}', min_value=32, max_value=256, step=32, default=64)
                hp.Int(f'num_heads_{i}', min_value=2, max_value=8, step=2, default=2)
                hp.Int(f'ff_dim_{i}', min_value=64, max_value=512, step=64, default=64)
                hp.Choice(f'transformer_activation_{i}', ['relu', 'tanh', 'gelu'])
        else:
            hp.Fixed('cnn_filters', 3)
            hp.Fixed('cnn_kernel_size', 3)

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
        try:
            if self.tunemode in tuner_classes:
                self.tuner = tuner_classes[self.tunemode](
                    hypermodel=self.build_model,
                    hyperparameters=hp,
                    hyperband_iterations=self.hyperband_iterations,
                    objective=self.objective,
                    max_epochs=self.max_epochs,
                    factor=self.factor,
                    directory=self.base_path,
                    project_name=self.modelname,
                    overwrite=True,
                    tune_new_entries=True,
                    allow_new_entries=True,
                    max_retries_per_trial=5,
                    max_consecutive_failed_trials=3,
                    executions_per_trial=1,
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
        inputs   = [] if self.multi_inputs else [shared_input]
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
                    filters     = hp.get(f'cnn_filters_{i}')
                    kernel_size = hp.get(f'cnn_kernel_size_{i}')
                    activation  = hp.get(f'cnn_activation_{i}')
                    cnn_branch  = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                    cnn_branch  = MaxPooling1D(pool_size=2)(cnn_branch)
                    cnn_branch  = LayerNormalization()(cnn_branch)
                    cnn_branch  = Dropout(0.2)(cnn_branch)
            else:
                filters     = hp.get('cnn_filters')
                kernel_size = hp.get('cnn_kernel_size')
                activation  = 'relu'
                cnn_branch  = Conv1D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same')(cnn_branch)
                cnn_branch  = MaxPooling1D(pool_size=2)(cnn_branch)
                cnn_branch  = LayerNormalization()(cnn_branch)
                cnn_branch  = Dropout(0.2)(cnn_branch)
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
                    units      = hp.get(f'lstm_units_{i}')
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
                    units      = hp.get(f'gru_units_{i}')
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
            num_transformer_blocks = hp.get('num_transformer_blocks') if self.tunemode else 1
            for i in range(num_transformer_blocks):
                transformer_branch = self.transformer_block(transformer_branch, hp, i, hp.get(f'key_dim_{i}', 64))
            transformer_branch = GlobalAveragePooling1D()(transformer_branch)
            transformer_branch = Dense(64, activation=hp.get('dense_1_activation') if self.tunemode else 'relu')(transformer_branch)
            transformer_branch = Dropout(0.2)(transformer_branch)
            branches.append(transformer_branch)

        # Concatenate branches if multiple are enabled
        concatenated = Concatenate()(branches) if self.multi_branches else branches[0]
        dense_1 = Dense(
            units=hp.Int('dense_1_units', min_value=32, max_value=128, step=32, default=80),
            activation=hp.get('dense_1_activation') if self.tunemode else 'relu',
            kernel_regularizer=tf.keras.regularizers.l2(hp.get('l2_reg'))
        )(concatenated)
        dense_dropout = Dropout(0.2)(dense_1)
        output = Dense(1, activation="linear" if 1 > 1 else "sigmoid")(dense_dropout)

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

        # Compile model with appropriate loss, metric, and optimizer
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
            loss=hp.get('loss') if self.tunemode else 'mse',
            steps_per_execution=50,
            metrics=[hp.get('metric')] if self.tunemode else ['mse']
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
            'adamax':  tf.keras.optimizers.Adamax,
            'ftrl':    tf.keras.optimizers.Ftrl
        }
        return optimizers[optimizer_name](learning_rate=learning_rate)

    def get_callbacks(self):
        checkpoint_filepath = self.checkpoint_filepath
        if checkpoint_filepath and isinstance(checkpoint_filepath, pathlib.Path):
            checkpoint_filepath = str(checkpoint_filepath)
        if checkpoint_filepath and not checkpoint_filepath.endswith('.keras'):
            checkpoint_filepath += '.keras'
        return [
            EarlyStopping(monitor=self.objective, patience=3, verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, verbose=1) if checkpoint_filepath else None,
            TensorBoard(log_dir=os.path.join(self.base_path, 'logs')),
            ReduceLROnPlateau(monitor=self.objective, factor=0.1, patience=3, min_lr=1e-6, verbose=1)
        ]

    def transformer_block(self, inputs, hp, block_num, dim):
        key_dim    = hp.get(f'key_dim_{block_num}')
        num_heads  = hp.get(f'num_heads_{block_num}')
        ff_dim     = hp.get(f'ff_dim_{block_num}')
        activation = hp.get(f'transformer_activation_{block_num}')
        attn_out   = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
        attn_out   = LayerNormalization()(inputs + attn_out)
        ffn_out    = Dense(ff_dim, activation=activation)(attn_out)
        ffn_out    = Dropout(0.2)(ffn_out)
        ffn_out    = Dense(inputs.shape[-1])(ffn_out)
        ffn_out    = LayerNormalization()(attn_out + ffn_out)
        ffn_out    = Dropout(0.2)(ffn_out)
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
                verbose=1,
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
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
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
        logger.info(f"Checking for model file at base_path {lpbase_path}")
        logger.info(f"Project name: {self.project_name}")
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
