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
    def __init__(self,X_train,y_train,lp_shape,lp_k_reg,lp_objective = 'val_accuracy',optimizer='adam',lp_act1 = 'relu', lp_act2 = 'linear',lp_act3 = 'sigmoid',lp_metric = 'accuracy', lp_loss1='mean_squared_error',lp_loss2 = 'binary_crossentropy',lp_min=32,lp_max=128,lp_step=2,lp_hmin=8,lp_hmax=128,lp_hstep=2,lp_imin=64,lp_imax=256,lp_istep=64,lp_jmin=0.5,lp_jmax=0.2,lp_jstep=0.1):
        self._X_train = X_train
        self._y_train = y_train
        self._lp_shape = lp_shape
        self._lp_objective = lp_objective
        self._lp_k_reg = lp_k_reg
        self._lp_act1 = lp_act1
        self._lp_act2 = lp_act2
        self._lp_act3 = lp_act3
        self._lp_metric = lp_metric
        self._lp_loss1 = lp_loss1
        self._lp_loss2 = lp_loss2
        self._lp_min = lp_min
        self._lp_max = lp_max
        self._lp_step = lp_step
        self._lp_hmin = lp_hmin
        self._lp_hmax = lp_hmax
        self._lp_hstep = lp_hstep
        self._lp_imin = lp_imin
        self._lp_imax = lp_imax
        self._lp_istep = lp_istep
        self._lp_jmin = lp_jmin
        self._lp_jmax = lp_jmax
        self._lp_jstep = lp_jstep
        self._optimizer = optimizer

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def lp_shape(self):
        return self._lp_shape

    @lp_shape.setter
    def lp_shape(self, value):
        self._lp_shape = value

    @property
    def lp_objective(self):
        return self._lp_objective

    @lp_objective.setter
    def lp_objective(self, value):
        self._lp_objective = value

    @property
    def lp_k_reg(self):
        return self._lp_k_reg

    @lp_k_reg.setter
    def lp_k_reg(self, value):
        self._lp_k_reg = value

    @property
    def lp_act1(self):
        return self._lp_act1

    @lp_act1.setter
    def lp_act1(self, value):
        self._lp_act1 = value

    @property
    def lp_act2(self):
        return self._lp_act2

    @lp_act2.setter
    def lp_act2(self, value):
        self._lp_act2 = value

    @property
    def lp_act3(self):
        return self._lp_act3

    @lp_act3.setter
    def lp_act3(self, value):
        self._lp_act3 = value

    @property
    def lp_metric(self):
        return self._lp_metric

    @lp_metric.setter
    def lp_metric(self, value):
        self._lp_metric = value

    @property
    def lp_loss1(self):
        return self._lp_loss1

    @lp_loss1.setter
    def lp_loss1(self, value):
        self._lp_loss1 = value

    @property
    def lp_loss2(self):
        return self._lp_loss2

    @lp_loss2.setter
    def lp_loss2(self, value):
        self._lp_loss2 = value

    @property
    def lp_min(self):
        return self._lp_min

    @lp_min.setter
    def lp_min(self, value):
        self._lp_min = value

    @property
    def lp_max(self):
        return self._lp_max

    @lp_max.setter
    def lp_max(self, value):
        self._lp_max = value

    @property
    def lp_step(self):
        return self._lp_step

    @lp_step.setter
    def lp_step(self, value):
        self._lp_step = value

    @property
    def lp_hmin(self):
        return self._lp_hmin

    @lp_hmin.setter
    def lp_hmin(self, value):
        self._lp_hmin = value

    @property
    def lp_hmax(self):
        return self._lp_hmax

    @lp_hmax.setter
    def lp_hmax(self, value):
        self._lp_hmax = value

    @property
    def lp_hstep(self):
        return self._lp_hstep

    @lp_hstep.setter
    def lp_hstep(self, value):
        self._lp_hstep = value

    @property
    def lp_imin(self):
        return self._lp_imin

    @lp_imin.setter
    def lp_imin(self, value):
        self._lp_imin = value

    @property
    def lp_imax(self):
        return self._lp_imax

    @lp_imax.setter
    def lp_imax(self, value):
        self._lp_imax = value

    @property
    def lp_istep(self):
        return self._lp_istep

    @lp_istep.setter
    def lp_istep(self, value):
        self._lp_istep = value

    @property
    def lp_jmin(self):
        return self._lp_jmin

    @lp_jmin.setter
    def lp_jmin(self, value):
        self._lp_jmin = value

    @property
    def lp_jmax(self):
        return self._lp_jmax

    @lp_jmax.setter
    def lp_jmax(self, value):
        self._lp_jmax = value

    @property
    def lp_jstep(self):
        return self._lp_jstep

    @lp_jstep.setter
    def lp_jstep(self, value):
        self._lp_jstep = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

#--------------------------------------------------------------------
# create method  "dl_tune_neuro_ensemble".
# class: cmqlmlsetup
# usage: mql data
# \pdl_tune_neuro_network
#--------------------------------------------------------------------  
    def dl_tune_neuro_ensemble(self):
        # Example data
        # X_train = np.random.rand(1000, 100, 1)  # 1000 samples, 100 time steps, 1 feature
        # y_train = np.random.randint(2, size=(1000,))  # Binary target
        # Define input shape
        # input_shape = (100, 1)

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
    def __init__(self, embed_dim, num_heads, ff_dim,lp_act1 = 'relu'):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation=lp_act1), Dense(embed_dim)]
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
            units = hp.Int('lstm_units',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step)
            x = LSTM(units)(inputs)
            x = Dense(hp.Int('lstm_dense_units', min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step), activation=self.lp_act1)(x)
            return Model(inputs, x)
        
        # 1D CNN Model
        def create_cnn_model():
            inputs = Input(shape=self.input_shape)
            filters = hp.Int('cnn_filters',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step)
            kernel_size = hp.Choice('cnn_kernel_size', values=[3, 5])
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation=self.lp_act1)(inputs)
            x = MaxPooling1D(pool_size=2)(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step), activation=self.lp_act1)(x)
            return Model(inputs, x)
        
        # GRU Model
        def create_gru_model():
            inputs = Input(shape=self.input_shape)
            units = hp.Int('gru_units',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step)
            x = GRU(units)(inputs)
            x = Dense(hp.Int('gru_dense_units',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step), activation=self.lp_act1)(x)
            return Model(inputs, x)
        
        # Transformer Model
        def create_transformer_model():
            inputs = Input(shape=self.input_shape)
            embed_dim = hp.Int('transformer_embed_dim',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step)
            num_heads = hp.Int('transformer_num_heads', min_value=self.lp_hmin, max_value=self.lp_hmax, step=self.lp_hstep)
            ff_dim = hp.Int('transformer_ff_dim',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step)
            x = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)(inputs)
            x = GlobalAveragePooling1D()(x)
            x = Dense(hp.Int('transformer_dense_units',  min_values=self.lp_min, max_value=self.lp_max, step=self.lp_step), activation=self.lp_act1)(x)
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
        x = Dense(hp.Int('ensemble_dense_units', min_value=self.lp_imin, max_value=self.lp_imax, step=self.lp_istep), activation='relu')(combined_output)
        x = Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1))(x)
        output = Dense(1, activation=self.lp_act2)(x)
        
       # Create and compile the model
        model = Model(inputs=[lstm_model.input, cnn_model.input, gru_model.input, transformer_model.input], outputs=output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),loss='binary_crossentropy', metrics=['accuracy'])    
        return model

    # Set up the Keras Tuner
def run_tuner(input_shape, X_train, y_train,lp_objective,lp_validation_split=0.2, lp_epochs=10, lp_batch_size=32,lp_num_trials=1,lp_num_models=1):
    tuner = RandomSearch(
        EnsembleHyperModel(input_shape=input_shape),
        objective=lp_objective,
        max_trials=10,  # Number of hyperparameter sets to try
        executions_per_trial=1,  # Number of models to build and evaluate for each trial
        directory='ensemble_tuning',
        project_name='ensemble_model'
    )
    
    # Train the tuner
    tuner.search([X_train, X_train, X_train, X_train], y_train, lp_validation_split,lp_epochs, lp_batch_size,lp_num_trials)
    
    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(lp_num_trials=1)[0]
    
    # Print the best hyperparameters
    print(f"Best hyperparameters: {best_hp.values}")
    
    # Return the best model
    return tuner.get_best_models(num_models=lp_num_models)[0]

