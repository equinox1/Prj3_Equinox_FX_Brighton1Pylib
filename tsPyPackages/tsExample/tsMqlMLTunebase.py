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
#-----------------------------------------------
# Class CMqlmlsetuptune
#+--------------------------------------------------
class CMqlmlsetuptune:
    def __init__(self, hp, df, X=None, y=None, X_train=None, X_train_scaled=None, X_test=None, X_test_scaled=None, y_train=None, y_test=None):
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
        self._X = X
        self._y = y
        self._X_train = X_train
        self._X_train_scaled = X_train_scaled
        self._X_test = X_test
        self._X_test_scaled = X_test_scaled
        self._y_train = y_train
        self._y_test = y_test

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property
    def X_test(self):
        return self._X_test

    @X_test.setter
    def X_test(self, value):
        self._X_test = value

    @property
    def X_train_scaled(self):
        return self._X_train_scaled

    @X_train_scaled.setter
    def X_train_scaled(self, value):
        scaler = StandardScaler()
        self._X_train_scaled = scaler.fit_transform(self._X_train)

    @property
    def X_test_scaled(self):
        return self._X_test_scaled

    @X_test_scaled.setter
    def X_test_scaled(self, value):
        scaler = StandardScaler()
        self._X_test_scaled = scaler.transform(self._X_test)

    @property
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property
    def y_test(self):
        return self._y_test

    @y_test.setter
    def y_test(self, value):
        self._y_test = value

    #--------------------------------------------------------------------
    # create method  "model_builder"
    # class:cmqlmlsetup
    # usage: mql data
    # /pdl_build_neuro_network
    #--------------------------------------------------------------------
    def model_builder(hp):
        '''
        Args:
        hp - Keras tuner object
        '''
        # Initialize the Sequential API and start stacking the layers
        model = tf.keras.models.Sequential()
        
        # Tune the number of units in the first Dense layer Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', name='dense_1'))

        # Add next layers
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        # Tune the learning rate for the optimizer Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        return model

    def run_model_tune(self, x_train, y_train):
        # Instantiate the tuner
        tuner = kt.Hyperband(self.model_builder,
                             objective='val_accuracy',  # objective to optimize
                             max_epochs=10,
                             factor=3,  # factor which you have seen above
                             directory='dir',  # directory to save logs
                             project_name='equinox1')
        # hypertuning settings
        tuner.search_space_summary()

        # Search space summary
        # Default search space size: 2
        # units (Int)
        # {'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': None}
        # learning_rate (Choice)
        # {'default': 0.01, 'conditions': [], 'values': [0.01, 0.001, 0.0001], 'ordered': True}
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Perform hypertuning
        tuner.search(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[stop_early])
        best_hp = tuner.get_best_hyperparameters()[0]

        # Build the model with the optimal hyperparameters
        h_model = tuner.hypermodel.build(best_hp)
        h_model.summary()
        h_model.fit(x_train, y_train, epochs=10, validation_split=0.2)

        # Evaluate the model
        h_eval_dict = h_model.evaluate(x_train, y_train, return_dict=True)
        return h_eval_dict



        #--------------------------------------------------------------------
# create method  "dl_model_tune_build"
# class:cmqlmlsetup
# usage: mql data
# /pdl_build_neuro_network
#--------------------------------------------------------------------
# Method to build the model for Keras Tuner
    def dl_model_tune_build(self,lp_X_train_scaled, lp_X_test_scaled, optimizer='adam', loss='mean_squared_error', k_reg=0.001):
        tmodel = tf.keras.models.Sequential()

        # Input layer
        hp = kt.HyperParameters()

        tmodel.add(tf.keras.layers.Flatten(input_shape=(lp_X_train_scaled.shape[1],)))
        
        # Hidden Layer 1: Tune the number of units between 32-512
        hpunits1 = hp.Int('hpunits1', min_value=32, max_value=512, step=32)
        tmodel.add(tf.keras.layers.Dense(units=hpunits1, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(k_reg)))

        # Hidden Layer 2: Tune the number of units between 32-512
        hpunits2 = hp.Int('hpunits2', min_value=32, max_value=512, step=32)
        tmodel.add(tf.keras.layers.Dense(units=hpunits2, activation='relu',
                                         kernel_regularizer=tf.keras.regularizers.l2(k_reg)))

        # Output Layer: Linear activation for regression
        tmodel.add(tf.keras.layers.Dense(1, activation='linear'))

        # Learning rate tuning
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

        tmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                       loss=loss,
                       metrics=['mean_squared_error'])  # For regression

        return tmodel


#--------------------------------------------------------------------
# create method  "dl_model_tune_run"
# class:cmqlmlsetup
# usage: mql data
# /pdl_build_neuro_network
#--------------------------------------------------------------------           
# Method to run the Keras Tuner
    def dl_model_tune_run(self,lp_X_train_scaled, lp_X_test_scaled, optimizer='adam', loss='mean_squared_error', k_reg=0.001):
        # Build tuner
        
        tuner = kt.Hyperband(self.dl_model_tune_build(self),
                             objective='val_loss',  # Use 'val_loss' for regression
                             max_epochs=10,
                             factor=3,
                             directory='my_dir',
                             project_name='intro_to_kt')

        # Early stopping to prevent overfitting
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        # Perform the search for the best hyperparameters
        tuner.search(lp_X_train_scaled, self.label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
            The hyperparameter search is complete. The optimal number of units in the first densely-connected
            layer is {best_hps.get('hpunits1')} and the optimal learning rate for the optimizer
            is {best_hps.get('learning_rate')}.
        """)

        # Build the model with the optimal hyperparameters and train it
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(lp_X_train_scaled, self.label_train, epochs=50, validation_split=0.2)

        # Get the best epoch based on validation loss
        val_loss_per_epoch = history.history['val_loss']
        best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
        print('Best epoch: %d' % (best_epoch,))

        # Rebuild the model and retrain it on the optimal number of epochs
        hypermodel = tuner.hypermodel.build(best_hps)
        hypermodel.fit(lp_X_train_scaled, self.label_train, epochs=best_epoch, validation_split=0.2)

        # Evaluate the model on test data
        eval_result = hypermodel.evaluate(lp_X_test_scaled, self.label_test)
        print("[test loss, mean squared error]:", eval_result)
        return eval_result
    

    #--------------------------------------------------------------------
# create method  "dl_model_performance"
# class:cmqlmlsetup
# usage: mql data
# /pdl_build_neuro_network
#--------------------------------------------------------------------
#Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values.
# The lower the MSE, the better the model.

#Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values.
# Like MSE, lower values indicate better model performance.

#R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
# dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates
# a perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the
#target variable. Negative values indicate poor model performance.
    def dl_model_performance(self,lp_model,lp_X_train_scaled ,lp_X_test_scaled):
        # sourcery skip: instance-method-first-arg-name
        lp_X_train_scaled= lp_X_train_scaled[:len(lp_X_test_scaled)]  # Truncate 'x' to match 'y'
        # Calculate and print mean squared error
        mse = mean_squared_error(lp_X_train_scaled, lp_model.predict(lp_X_test_scaled))
        print(f"/nMean Squared Error: {mse}")
        # Calculate and print mean absolute error
        mae = mean_absolute_error(lp_X_train_scaled, lp_model.predict(lp_X_test_scaled))
        print(f"/nMean Absolute Error: {mae}")
        # Calculate and print R2 Score
        r2 = r2_score(lp_X_train_scaled, lp_model.predict(lp_X_test_scaled))
        print(f"/nR2 Score: {r2}")
        return r2
