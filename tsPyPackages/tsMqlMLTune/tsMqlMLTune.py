import tensorflow as tf
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy
from keras.metrics import AUC
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from keras_tuner import HyperModel

# Class for running the tuner
class CMdtuner:
    def __init__(self, X_train, y_train, lstm_shape, cnn_shape, gru_shape, transformer_shape, objective, max_trials, executions_per_trial, directory, project_name, validation_split, epochs, batch_size, factor, channels):
        self.X_train = X_train
        self.y_train = y_train
        self.lstm_shape = lstm_shape
        self.cnn_shape = cnn_shape
        self.gru_shape = gru_shape
        self.transformer_shape = transformer_shape
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
            max_consecutive_failed_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=True,
            factor=self.factor,
            max_epochs=self.epochs
        )

    def build_model(self, hp):
        model = Sequential()
        print("Building model with hp:", hp)
        print("Model lstm_input:", self.lstm_shape)
               
        lstminputs = Input(shape=self.lstm_shape)  # Define the input shape explicitly
        lstminputs = Input(shape=(60,1))

        # Convolutional layers (search space example)
        x = Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
               kernel_size=hp.Int('conv_kernel_size', min_value=3, max_value=7, step=2),
               activation='relu')(lstminputs)
        x = MaxPooling1D(pool_size=2)(x)
       
        # Recurrent layers to capture temporal dependencies
        x = (LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True))(x)
        x = (LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32)))(x)

        # Flatten the output from the recurrent layers
        x = (Flatten())(x)

        # Dense layer for final prediction
        x = (Dense(1, activation='sigmoid'))(x)
        x = Dropout(0.2)(x)

        # Create the model
        model = Model(inputs=lstminputs, outputs=x)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        return model


    def run_tuner(self):
        self.tuner.search(self.X_train, self.y_train, validation_split=self.validation_split, epochs=self.epochs, batch_size=self.batch_size)
        best_model = self.tuner.get_best_models()[0]
        return best_model