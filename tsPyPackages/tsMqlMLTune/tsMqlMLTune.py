import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import keras_tuner as kt
from keras_tuner import HyperModel

# Class for running the tuner
class CMdtuner:
    def __init__(self, X_train, y_train, cnn_model, lstm_model, gru_model, transformer_model, lstm_shape, lstm_features, cnn_shape, cnn_features, gru_shape, gru_features, transformer_shape, transformer_features, objective, max_trials, executions_per_trial, directory, project_name, validation_split, epochs, batch_size, factor, channels):
        self.X_train = X_train
        self.y_train = y_train
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.transformer_model = transformer_model
        self.lstm_shape = lstm_shape
        self.lstm_features = lstm_features
        self.cnn_shape = cnn_shape
        self.cnn_features = cnn_features
        self.gru_shape = gru_shape
        self.gru_features = gru_features
        self.transformer_shape = transformer_shape
        self.transformer_features = transformer_features
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory
        self.project_name = project_name
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.factor = factor
        self.channels = channels
        
        # Define the base models target shape
        self.tuner = kt.Hyperband(
            self.build_model,
            objective=self.objective,
            max_epochs=self.epochs,
            executions_per_trial=self.executions_per_trial,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=True,
            factor=self.factor
        )

    def build_model(self, hp):
        print("Building model with hp:", hp)

        x_cnn = None
        x_lstm = None
        x_gru = None
        x_transformer = None

        cnninputs = None
        lstminputs = None
        gruinputs = None
        transformerinputs = None

        # Define the input shapes
        if self.cnn_model:
            cnninputs = Input(shape=(self.cnn_shape[1], 1))  # Define the input shape explicitly
            print("Set cnninputs shape:", cnninputs.shape)
        
        if self.lstm_model:
            lstminputs = Input(shape=(self.lstm_shape[1], 1))  # Define the input shape explicitly
            print("Set lstminputs shape:", lstminputs.shape)
        
        if self.gru_model:
            gruinputs = Input(shape=(self.gru_shape[1], 1))
            print("Set gruinputs shape:", gruinputs.shape)
        
        if self.transformer_model:
            transformerinputs = Input(shape=(self.transformer_shape[1], 1))  # Define the input shape explicitly
            print("Set transformerinputs shape:", transformerinputs.shape)
        
        # Convolutional layers (search space example)
        if self.cnn_model:
            x_cnn = Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
                          kernel_size=hp.Int('conv_kernel_size', min_value=3, max_value=7, step=2),
                          activation='relu')(cnninputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Dropout(0.2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
        
        # LSTM Recurrent layers to capture temporal dependencies
        if self.lstm_model: 
            x_lstm = LSTM(hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(lstminputs)
            x_lstm = LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32))(x_lstm)
            x_lstm = Dropout(0.2)(x_lstm)
            x_lstm = Flatten()(x_lstm)
        
        # GRU layers
        if self.gru_model:
            x_gru = GRU(hp.Int('gru_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(gruinputs)
            x_gru = GRU(hp.Int('gru_units_2', min_value=32, max_value=128, step=32))(x_gru)
            x_gru = Dropout(0.2)(x_gru)
            x_gru = Flatten()(x_gru)
        
        # Transformer layers
        if self.transformer_model:
            x_transformer = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(transformerinputs)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = Dropout(0.2)(x_transformer)
            x_transformer = GlobalAveragePooling1D()(x_transformer)    
        
        # Concatenate all the layers
        concat_layers = [layer for layer in [x_cnn, x_lstm, x_gru, x_transformer] if layer is not None]
        if len(concat_layers) > 1:
            x = Concatenate()(concat_layers)
        elif len(concat_layers) == 1:
            x = concat_layers[0]
        else:
            raise ValueError("At least one model (cnn_model, lstm_model, gru_model, transformer_model) must be enabled.")

        # Dense layer for final prediction
        x = Dense(1, activation='sigmoid')(x)

        # Create the model
        model = Model(inputs=[input for input in [cnninputs, lstminputs, gruinputs, transformerinputs] if input is not None], outputs=x)

        # Compile the model
        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])
                
        return model

    def run_tuner(self):
        self.tuner.search(self.X_train, self.y_train, validation_split=self.validation_split, epochs=self.epochs, batch_size=self.batch_size)
        best_model = self.tuner.get_best_models()[0]
        return best_model