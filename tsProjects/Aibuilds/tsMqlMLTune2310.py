import tensorflow as tf

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError

import keras_tuner as kt
from keras_tuner import Hyperband

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
import os  # Import the os module

class CMdtuner:
    def __init__(self, lv_X_train, lv_y_train, lp_cnn_model, lp_lstm_model, lp_gru_model, lp_transformer_model,lp_run_single_input_model,lp_run_single_input_submodels, 
                 lp_shapes_and_features, lp_objective, lp_max_epochs, lp_factor, lp_seed, lp_hyperband_iterations,
                 lp_tune_new_entries, lp_allow_new_entries, lp_max_retries_per_trial, lp_max_consecutive_failed_trials,
                 lp_validation_split, lp_epochs, lp_batch_size, lp_dropout, lp_oracle, lp_hypermodel, lp_max_model_size, lp_optimizer,
                 lp_loss, lp_metrics, lp_distribution_strategy, lp_directory, lp_project_name, lp_logger, lp_tuner_id,
                 lp_overwrite, lp_executions_per_trial, lp_chk_fullmodel,lp_chk_verbosity,lp_chk_mode,lp_chk_monitor,lp_chk_sav_freq,lp_checkpoint_filepath,lp_modeldatapath):
        # Set the input data
        self.X_train = lv_X_train
        self.y_train = lv_y_train
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
        self.callback_checkpoint_filepath=None
        self.modeldatapath = lp_modeldatapath
        self.tunefilename=os.path.join(self.modeldatapath,self.directory)
        # Initialize the Keras Tuner
        
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            executions_per_trial=self.executions_per_trial,
            directory=self.tunefilename,
            project_name=self.project_name,
            overwrite=self.overwrite,
            factor=self.factor
        )

    def build_model(self, hp):
        print("Building model with hp:", hp)
        inputs = []
        layers = []
        single_input = []

        # example working shape inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
        print("Single Input Shape:", single_input)
        # Define input shapes for the models
        for model_type, (input_shape, features) in self.shapes_and_features.items():
            if model_type == 'cnn' and self.cnn_model:
                input_tensor = Input(shape=(input_shape[1], features))
                inputs.append(input_tensor)
                single_input.append(input_tensor)
                print("Initial Input Shape:", input_tensor.shape)
                
                if self.run_single_input_submodels:
                    inputs=single_input
                    print("Initial Input CNN Shape from run single input submodels:", input_tensor.shape)
                else:
                    inputs.append(input_tensor)
                    print("Initial Input CNN Shape from run multi input submodels:", input_tensor.shape)

                 #override
                inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)

                # CNN branch
                x_cnn = Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32),
                             kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1), 
                             activation='relu')(input_tensor)
                x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
                x_cnn = Flatten()(x_cnn)
                #x_cnn = Dropout(self.dropout)(x_cnn)
                layers.append(x_cnn)
                

            elif model_type == 'lstm' and self.lstm_model:
                input_tensor = Input(shape=(input_shape[1], features))

                if self.run_single_input_submodels:
                    inputs=single_input
                    print("Initial Input LSTM Shape from run single input submodels:", input_tensor.shape)
                else:
                    #inputs.append(input_tensor)
                    print("Initial Input LSTM Shape from run multi input submodels:", input_tensor.shape)

                #override
                inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
                # LSTM Layers
                x_lstm = LSTM(hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(input_tensor)
                x_lstm = LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32))(x_lstm)
                #x_lstm = Flatten()(x_lstm)
                #x_lstm = Dropout(self.dropout)(x_lstm)
                layers.append(x_lstm)

            elif model_type == 'gru' and self.gru_model:
                input_tensor = Input(shape=(input_shape[1], features))
 
                if self.run_single_input_submodels:
                    inputs=single_input
                    print("Initial Input GRU Shape from run single input submodels:", input_tensor.shape)
                else:
                   # inputs.append(input_tensor)
                    print("Initial Input GRU Shape from run multi input submodels:", input_tensor.shape)

                 #override
                inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
                # GRU Layers
                x_gru = GRU(hp.Int('gru_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(input_tensor)
                x_gru = GRU(hp.Int('gru_units_2', min_value=32, max_value=128, step=32))(x_gru)
                x_gru = Flatten()(x_gru)
                #x_gru = Dropout(self.dropout)(x_gru)
                layers.append(x_gru)

            elif model_type == 'transformer' and self.transformer_model:
                input_tensor = Input(shape=(input_shape[1], features))
                
                if self.run_single_input_submodels:
                    inputs=input_tensor
                    print("Initial Input TRAN Shape from run single input submodels:", input_tensor.shape)
                else:
                    #inputs.append(input_tensor)
                    print("Initial Input TRAN Shape from run multi input submodels:", input_tensor.shape)

                 #override
                inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
                # Transformer Layers
                x_trans = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1),
                                        key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(input_tensor, input_tensor)
                x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
                x_trans = GlobalAveragePooling1D()(x_trans)
                #x_trans = Dropout(self.dropout)(x_trans)
                layers.append(x_trans)


            print("Checker Model types:", model_type)

        # Concatenate Layers if multiple models are enabled
        if len(layers) > 1:
            x = Concatenate()(layers)
        elif len(layers) == 1:
            x = layers[0]
        else:
            raise ValueError("At least one model (cnn_model, lstm_model, gru_model, transformer_model) must be enabled.")

    
        # Concatenate the outputs of the branches
        combined = Concatenate()([x_cnn, x_lstm, x_gru, x_trans])
        x = Dense(50, activation='relu')(combined)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='linear')(x)
        
        #Input is created from the concat of each input append and output x from each layers append 
        if self.run_single_input_model:
                model = Model(inputs=single_input, outputs=output)
                print("Running single input model",inputs)
        else:
                model = Model(inputs=inputs, outputs=output)
                print("Running multi input model",inputs)

        print("Layer Output Shape:", layers)
        print ("Combined Output Shape:", combined.shape)  
        print("Input Shape:", inputs) 
        print("Output Shape:", output.shape)
        
        #override
        inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)

        # Define model building function
        model = Model(inputs=inputs, outputs=output)
        # Compile the model with learning rate tuning
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)
        #model.compile(optimizer=Adam(learning_rate=learning_rate),loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])
        model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss=MeanSquaredError(), 
                  metrics=[MeanAbsoluteError()])
        #model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy', AUC()])
        return model

    def run_tuner(self):
        # Define a ModelCheckpoint callback
        if self.chk_fullmodel:
            self.callback_checkpoint_filepath= os.path.join(self.checkpoint_filepath, 'Eq_best_model.keras')
            print("Checkpoint Filepath:", self.callback_checkpoint_filepath)
            print("Callback Checkpoint Filepath:", self.callback_checkpoint_filepath)
            fp_save_best_only = True
            fp_save_weights_only = False
            fp_save_freq = self.chk_sav_freq
            fp_initial_value_threshold = None
        else:
            self.callback_checkpoint_filepath = os.path.join(self.checkpoint_filepath, 'Eq_checkpoint.weights.h5')
            print("Checkpoint Filepath:", self.checkpoint_filepath)
            print("Callback Checkpoint Filepath:", self.callback_checkpoint_filepath)
            fp_save_best_only = True
            fp_save_weights_only = True
            fp_save_freq = self.chk_sav_freq
            fp_initial_value_threshold = None 

        # Define a ModelCheckpoint callback

        checkpoint_callback = ModelCheckpoint(
            filepath=self.callback_checkpoint_filepath,
            monitor=self.chk_monitor,
            verbose=self.chk_verbosity,          
            save_best_only=fp_save_best_only,
            mode=self.chk_mode,
            save_weights_only=fp_save_weights_only,
            save_freq=fp_save_freq,
            initial_value_threshold=fp_initial_value_threshold)
        
        self.tuner.search_space_summary()
        # Run the hyperparameter tuning
        self.tuner.search(self.X_train, self.y_train,
                          validation_split=self.validation_split,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])
                          #callbacks=[EarlyStopping(monitor=self.chk_monitor, patience=3), checkpoint_callback])
        

        # Get the best model
        best_model = self.tuner.get_best_models()
        #best_params = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        # best_model[0].summary()
        return best_model