import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import keras_tuner as kt

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import os

class CMdtuner:
    def __init__(self, lv_X_train, lv_y_train, lv_X_test, lv_y_test, lp_cnn_model, lp_lstm_model, lp_gru_model, lp_transformer_model, lp_run_single_input_model, lp_run_single_input_submodels, 
                 lp_shapes_and_features, lp_objective, lp_max_epochs, lp_factor, lp_seed, lp_hyperband_iterations,
                 lp_tune_new_entries, lp_allow_new_entries, lp_max_retries_per_trial, lp_max_consecutive_failed_trials,
                 lp_validation_split, lp_epochs, lp_batch_size, lp_dropout, lp_oracle, lp_hypermodel, lp_max_model_size, lp_optimizer,
                 lp_loss, lp_metrics, lp_distribution_strategy, lp_directory, lp_project_name, lp_logger, lp_tuner_id,
                 lp_overwrite, lp_executions_per_trial, lp_chk_fullmodel, lp_chk_verbosity, lp_chk_mode, lp_chk_monitor, lp_chk_sav_freq, lp_checkpoint_filepath, lp_modeldatapath):
        # Set the input data
        self.X_train = lv_X_train
        self.y_train = lv_y_train
        self.X_test = lv_X_test
        self.y_test = lv_y_test 
        self.cnn_model = lp_cnn_model
        self.lstm_model = lp_lstm_model
        self.gru_model = lp_gru_model
        self.transformer_model = lp_transformer_model
        self.run_single_input_model = lp_run_single_input_model
        self.run_single_input_submodels = lp_run_single_input_submodels
        self.shapes_and_features = lp_shapes_and_features
        self.objective = lp_objective
        self.max_epochs = lp_max_epochs
        self.factor = lp_factor
        self.seed = lp_seed
        self.hyperband_iterations = lp_hyperband_iterations
        self.tune_new_entries = lp_tune_new_entries 
        self.allow_new_entries = lp_allow_new_entries
        self.max_retries_per_trial = lp_max_retries_per_trial   
        self.max_consecutive_failed_trials = lp_max_consecutive_failed_trials
        self.validation_split = lp_validation_split 
        self.epochs = lp_epochs
        self.batch_size = lp_batch_size
        self.dropout = lp_dropout   
        self.oracle = lp_oracle
        self.hypermodel = lp_hypermodel
        self.max_model_size = lp_max_model_size
        self.optimizer = lp_optimizer
        self.loss = lp_loss
        self.metrics = lp_metrics
        self.distribution_strategy = lp_distribution_strategy
        self.directory = lp_directory
        self.project_name = lp_project_name
        self.logger = lp_logger
        self.tuner_id = lp_tuner_id
        self.overwrite = lp_overwrite
        self.executions_per_trial = lp_executions_per_trial
        self.chk_fullmodel = lp_chk_fullmodel
        self.chk_verbosity = lp_chk_verbosity
        self.chk_mode = lp_chk_mode
        self.chk_monitor = lp_chk_monitor
        self.chk_sav_freq = lp_chk_sav_freq
        self.checkpoint_filepath = lp_checkpoint_filepath
        self.callback_checkpoint_filepath = None
        self.modeldatapath = lp_modeldatapath
        self.tunefilename = os.path.join(self.modeldatapath, self.directory)

        self.tuner = kt.Hyperband(self.build_model,
                     objective='val_mean_absolute_error',
                     max_epochs=2,
                     factor=3,
                     directory='c:\\tmp\\keras_tuner_dir',
                     project_name='forex_price_forecasting')
      

        self.tuner.search_space_summary()

        self.tuner.search(self.X_train, self.y_train,
                          epochs=2,
                          validation_data=(self.X_test, self.y_test),
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

    def build_model(self, hp):
        inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
        
        # CNN branch
        x_cnn = Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32), 
                       kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1), 
                       activation='relu')(inputs)
        x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = Flatten()(x_cnn)
        
        # LSTM branch
        x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
        x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32))(x_lstm)
        
        # GRU branch
        x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
        x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32))(x_gru)
        
        # Transformer branch
        x_trans = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), 
                                     key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(inputs, inputs)
        x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
        x_trans = GlobalAveragePooling1D()(x_trans)
        
        # Concatenate the outputs of the branches
        combined = Concatenate()([x_cnn, x_lstm, x_gru, x_trans])
        x = Dense(50, activation='relu')(combined)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss=MeanSquaredError(), 
                      metrics=[MeanAbsoluteError()])
        
        return model