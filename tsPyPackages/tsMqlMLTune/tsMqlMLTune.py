import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import ml_dtypes
import ml_dtypes as mld

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
                 lp_validation_split, lp_epochs, lp_batch_size, lp_dropout, lp_oracle,lp_activation1,lp_activation2,lp_activation3,lp_activation4, lp_hypermodel, lp_max_model_size, lp_optimizer,
                 lp_loss, lp_metrics, lp_distribution_strategy, lp_directory,lp_basepath, lp_project_name, lp_logger, lp_tuner_id,
                 lp_overwrite, lp_executions_per_trial, lp_chk_fullmodel, lp_chk_verbosity, lp_chk_mode, lp_chk_monitor, lp_chk_sav_freq, lp_chk_patience,lp_checkpoint_filepath, lp_modeldatapath):
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
        self.activation1 = lp_activation1
        self.activation2 = lp_activation2
        self.activation3 = lp_activation3
        self.activation4 = lp_activation4
        self.hypermodel = lp_hypermodel
        self.max_model_size = lp_max_model_size
        self.optimizer = lp_optimizer
        self.loss = lp_loss
        self.metrics = lp_metrics
        self.distribution_strategy = lp_distribution_strategy
        self.directory = lp_directory
        self.basepath = lp_basepath
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
        self.chk_patience = lp_chk_patience
        self.checkpoint_filepath = lp_checkpoint_filepath
        self.callback_checkpoint_filepath = None
        self.modeldatapath = lp_modeldatapath
        self.tunefilename = os.path.join(self.modeldatapath, self.directory)
        
        
        self.tuner = kt.Hyperband(
                     hypermodel=self.build_model,
                     objective=self.objective,
                     max_epochs=self.max_epochs,
                     executions_per_trial=self.executions_per_trial,
                     directory=self.basepath,
                     project_name=self.project_name,
                     overwrite=self.overwrite,
                     factor=self.factor
                     )



        self.tuner.search_space_summary()

        self.tuner.search(self.X_train, self.y_train,
                          epochs=2,
                          validation_data=(self.X_test, self.y_test),
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor=self.chk_monitor, patience=self.chk_patience)])

    def build_model(self, hp):
        
        cnn_inputs, lstm_inputs, gru_inputs, transformer_inputs, single_inputs = None, None, None, None, None
        x_cnn, x_lstm, x_gru,x_trans = None, None, None, None

        for model_type, (input_shape, features) in self.shapes_and_features.items():
            if model_type == 'cnn':
                features_tuple = (features,)
                cnn_inputs = Input(shape=(input_shape,) + features_tuple)
                print("model cnn_inputs:", cnn_inputs)
            elif model_type == 'lstm':
                features_tuple = (features,)
                lstm_inputs = Input(shape=(input_shape,) + features_tuple)
                print("model lstm_inputs:", lstm_inputs)
            elif model_type == 'gru':
                features_tuple = (features,)
                gru_inputs = Input(shape=(input_shape,) + features_tuple)
                print("model gru_inputs:", gru_inputs)
            elif model_type == 'transformer':
                features_tuple = (features,)
                transformer_inputs = Input(shape=(input_shape,) + features_tuple)
                print("model transformer_inputs:", transformer_inputs)
            elif model_type == 'single_input':
                features_tuple = (features,)
                single_inputs = Input(shape=(input_shape,) + features_tuple)
                print("model inputs:", single_inputs)

        # CNN branch
        x_cnn = Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32), 
                       kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1), 
                       activation=self.activation1)(cnn_inputs)
        x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
        x_cnn = Flatten()(x_cnn)
        
        # LSTM branch
        x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)(lstm_inputs)
        x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32))(x_lstm)
        
        # GRU branch
        x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32), return_sequences=True)(gru_inputs)
        x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32))(x_gru)
        
        # Transformer branch
        x_trans = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), 
                                     key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(transformer_inputs, transformer_inputs)
        x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
        x_trans = GlobalAveragePooling1D()(x_trans)
        
        # Concatenate the outputs of the branches
        combined = Concatenate()([x_cnn, x_lstm, x_gru, x_trans])
        x = Dense(50, activation='relu')(combined)
        x = Dropout(0.3)(x)
        output = Dense(1, activation=self.activation2)(x)
        
        if self.run_single_input_model:
            model = Model(inputs=single_inputs, outputs=output)
            print("Running single input model", single_inputs)
        else:
            inputs = [cnn_inputs, lstm_inputs, gru_inputs, transformer_inputs]
            model = Model(inputs=inputs, outputs=output)
            print("Running multi input model", inputs)
        
       # print("Model Parameters: inputs: ", inputs., "outputs: ", output)

        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                      loss=MeanSquaredError(), 
                      metrics=[MeanAbsoluteError()])
        
        return model