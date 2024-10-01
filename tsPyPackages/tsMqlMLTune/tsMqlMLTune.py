#+------------------------------------------------------------------+
#|                                                   tsMqlMLTune.pyw
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------
import numpy as np
import pandas as pd
import tabulate
import torch
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#+-------------------------------------------------------------------
# import ai packages scikit-learns
#+-------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
#+-------------------------------------------------------------------
# import keras package
#+-------------------------------------------------------------------
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import keras
from keras import layers
from keras import ops
#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
import keras_tuner as kt

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from keras_tuner import HyperModel
from keras_tuner import RandomSearch
#-----------------------------------------------i
# Class CMqlmlsetuptune
#+--------------------------------------------------
class CMqlmlsetuptune:
    def __init__(self, hp ,df,X=None, y=None, X_train=None, X_train_scaled=None, X_test=None, X_test_scaled=None, y_train=None, y_test=None):
        if X is None:
            X = []
        if y is None:
            y = []
        if X_train is None:
            X_train = []
        if X_train_scaled is None:
            X_train_scaled = []
        if X_test is None:
            X_test = []
        if X_test_scaled is None:
            X_test_scaled = []
        if y_train is None:
            y_train = []
        if y_test is None:
            y_test = []
        self._df = df
    @property
    def df(self):
        return self.df

    @df.setter
    def dfx(self, value):
        self._df = value
    @property
    def X(self):
        return self.X
    @X.setter
    def X(self, value):
        value = self._X = []
        self._X = value
    @property
    def y(self):
        return self.y
    @y.setter
    def y(self, value):
        value = self._y = []
        self._y = value
    @property
    def X_train(self):
        return self.X_train
    @X_train.setter
    def X_train(self, value):
        self._X_train = value
    @property
    def X_test(self):
        return self.X_test
    @X_test.setter
    def X_test(self, value):
        self._X_test = value
    @property
    def X_train_scaled(self):
        return self.X_train_scaled
    @X_train_scaled.setter

    def X_train_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.fit_transform(self.X_train)
        self._X_train_scaled = value

    @property
    def X_test_scaled(self):
        return self.X_test_scaled
    @X_test_scaled.setter
    def X_test_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.transform(self.X_test)
        self._X_test_scaled = value

    @property
    def y_train(self):
        return self.y_train
    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def y_test(self):
        return self.y_test
    @y_test.setter
    def y_test(self, value):
        self._y_test = value

  
#--------------------------------------------------------------------
# create method  "dl_tune_neuro_ensemble".
# class: cmqlmlsetup
# usage: mql data
# \pdl_tune_neuro_network
#--------------------------------------------------------------------  
    def dl_tune_neuro_ensemble (self,lp_seq,lp_filt,lp_pool,lp_ksizelp_k_reg, X_train, y_train,optimizer='adam',lp_act1 = 'relu', lp_act2 = 'linear',lp_act3 = 'sigmoid',lp_metric = 'accuracy', lp_loss1='mean_squared_error',lp_loss2 = 'binary_crossentropy',cells1=128,cells2=256, cells3=128,cells4=64,cells5=1):
        # Example data
        #X_train = np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
        #y_train = np.random.randint(2, size=(1000,))  # Binary target
        # Define input shape
        #input_shape = (100, 1)

        # Run the tuner
        # best_model = run_tuner(input_shape, X_train, y_train)
        # Print the summary of the best model
        # best_model.summary()

        # Optionally: train the best model on the full dataset
        # best_model.fit([X_train, X_train, X_train, X_train], y_train, epochs=10, batch_size=32)
        return self

#--------------------------------------------------------------------
# create method  "dl_build_neuro_ensemble".
# class: cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
#Base Models:

#LSTM Model: Captures long-term temporal dependencies.
#1D CNN Model: Extracts local spatial features.
#GRU Model: Another RNN variant that captures temporal patterns.
#Transformer Model: Uses self-attention to capture global dependencies in the sequence.

#Key Components:
#Base Models:

#LSTM: Captures temporal dependencies.
#1D CNN: Handles spatial dependencies.
#GRU: A recurrent network like LSTM but more computationally efficient.
#Transformer: Uses attention mechanisms to capture long-range dependencies.
#Keras Tuner:

#The EnsembleHyperModel class defines the structure of the ensemble model and uses hp (hyperparameters) to tune different aspects like the number of units, layers, and learning rate.
#RandomSearch is the Keras Tuner method used to try different sets of hyperparameters and find the optimal configuration.
#Hyperparameters:

#For each model (LSTM, CNN, GRU, Transformer), we tune parameters such as the number of units, filter sizes, and dense layer configurations.
#We also tune hyperparameters for the final dense layers and the learning rate of the optimizer.
#Input Shape:

#The input shape is (100, 1) for each model, meaning 100 time steps with 1 feature. This can be adjusted depending on your data.
#Training and Searching:

#The tuner will search through different configurations, train the models, and evaluate them on validation data to find the best hyperparameters.
#You can modify max_trials and executions_per_trial to control the number of trials and model evaluations.
#Benefits of Using Keras Tuner:
#Automates hyperparameter tuning to find the best architecture and hyperparameters.
#Helps optimize the ensemble model to achieve better performance on your dataset.
#This ensemble setup provides a powerful method to capture complex relationships in sequence data and can be adapted to other types of problems with appropriate modifications.
 
# Transformer Block definition (used later in the transformer model)
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define a HyperModel class for the Keras Tuner
class EnsembleHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape
    
    def build(self, hp):
    
    ### Base Models ###
        
    # LSTM Model
        def create_lstm_model():
            inputs = Input(shape=self.input_shape)
            units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
            x = LSTM(units)(inputs)
            x = Dense(hp.Int('lstm_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # 1D CNN Model
        def create_cnn_model():
            inputs = Input(shape=self.input_shape)
            filters = hp.Int('cnn_filters', min_value=32, max_value=128, step=32)
            kernel_size = hp.Choice('cnn_kernel_size', values=[3, 5])
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # GRU Model
        def create_gru_model():
            inputs = Input(shape=self.input_shape)
            units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
            x = GRU(units)(inputs)
            x = Dense(hp.Int('gru_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        # Transformer Model
        def create_transformer_model():
            inputs = Input(shape=self.input_shape)
            embed_dim = hp.Int('transformer_embed_dim', min_value=32, max_value=128, step=32)
            num_heads = hp.Int('transformer_num_heads', min_value=2, max_value=8, step=2)
            ff_dim = hp.Int('transformer_ff_dim', min_value=32, max_value=128, step=32)
            x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(inputs)
            x = GlobalAveragePooling1D()(x)
            x = Dense(hp.Int('transformer_dense_units', min_value=32, max_value=128, step=32), activation='relu')(x)
            return Model(inputs, x)
        
        ### Ensemble Model ###
        # Instantiate base models
        lstm_model = create_lstm_model()
        cnn_model = create_cnn_model()
        gru_model = create_gru_model()
        transformer_model = create_transformer_model()
      
        # Concatenate the outputs from each base model
        combined_output = concatenate([lstm_model.output, cnn_model.output, gru_model.output, transformer_model.output])
        
        # Add Dense layers to learn from the combined outputs
        x = Dense(hp.Int('ensemble_dense_units', min_value=64, max_value=256, step=64), activation='relu')(combined_output)
        x = Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
        output = Dense(1, activation='sigmoid')(x)
        
       # Create and compile the model
        model = Model(inputs=[lstm_model.input, cnn_model.input, gru_model.input, transformer_model.input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),loss='binary_crossentropy', metrics=['accuracy'])    
        return model

    # Set up the Keras Tuner
def run_tuner(input_shape, X_train, y_train):
    tuner = RandomSearch(
        EnsembleHyperModel(input_shape=input_shape),
        objective='val_accuracy',
        max_trials=10,  # Number of hyperparameter sets to try
        executions_per_trial=1,  # Number of models to build and evaluate for each trial
        directory='ensemble_tuning',
        project_name='ensemble_model'
    )
    
    # Train the tuner
    tuner.search([X_train, X_train, X_train, X_train], y_train, validation_split=0.2, epochs=10, batch_size=32)
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print the best hyperparameters
    print(f"Best hyperparameters: {best_hp.values}")
    
    # Return the best model
    return tuner.get_best_models(num_models=1)[0]

