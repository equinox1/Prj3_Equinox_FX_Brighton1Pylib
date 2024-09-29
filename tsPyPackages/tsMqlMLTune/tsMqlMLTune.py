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
#+-------------------------------------------------------------------
# import ai packages scikit-learns
#+-------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
#+-------------------------------------------------------------------
# import keras package
#+-------------------------------------------------------------------
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
#======================================================
# import ai packages tensorflow and keras libraries
#======================================================
import tensorflow as tf
import keras_tuner as kt
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
# create method  "model_builder"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
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


    
    def run_model_tune(self , x_train, y_train):
        # Instantiate the tuner
        model_builder = self.model_builder
        tuner = kt.Hyperband(model_builder, # the hypermodel
            objective='val_accuracy', # objective to optimize
            max_epochs=10,
            factor=3, # factor which you have seen above 
            directory='dir', # directory to save logs 
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
        best_hp=tuner.get_best_hyperparameters()[0]

        # Build the model with the optimal hyperparameters
        h_model = tuner.hypermodel.build(best_hps)
        h_model.summary()
        h_model.fit(x_train, x_test, epochs=10, validation_split=0.2)

        # Evaluate the model
        h_eval_dict = h_model.evaluate(img_test, label_test, return_dict=True)
        return h_eval_dict