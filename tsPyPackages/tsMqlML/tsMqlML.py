#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
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
import matplotlib.pyplot as plt
import pandas as pd

#+-------------------------------------------------------------------
# import ai packages scikit-learns
#+-------------------------------------------------------------------
import scipy
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import  KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================

import keras
from keras import Sequential
from keras import layers
from keras import ops

from tensorflow.python.keras import layers

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.regularizers import l2
#-----------------------------------------------
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self, lp_df, lv_X=None, lv_y=None, lv_X_train=None, lv_X_train_scaled=None, lv_X_test=None, lv_X_test_scaled=None, lv_y_train=None, lv_y_test=None):
        if lv_X is None:
            lv_X = []
        if lv_y is None:
            lv_y = []
        if lv_X_train is None:
            lv_X_train = []
        if lv_X_train_scaled is None:
            lv_X_train_scaled = []
        if lv_X_test is None:
            lv_X_test = []
        if lv_X_test_scaled is None:
            lv_X_test_scaled = []
        if lv_y_train is None:
            lv_y_train = []
        if lv_y_test is None:
            lv_y_test = []
        self._lp_df = lp_df
    @property
    def lp_df(self):
        return self.lp_df

    @lp_df.setter
    def lp_dfx(self, value):
        self._lp_df = value
    @property
    def lv_X(self):
        return self.lv_X
    @lv_X.setter
    def lv_X(self, value):
        value = self._lv_X = []
        self._lv_X = value
    @property
    def lv_y(self):
        return self.lv_y
    @lv_y.setter
    def lv_y(self, value):
        value = self._lv_y = []
        self._lv_y = value
    @property
    def lv_X_train(self):
        return self.lv_X_train
    @lv_X_train.setter
    def lv_X_train(self, value):
        self._lv_X_train = value
    @property
    def lv_X_test(self):
        return self.lv_X_test
    @lv_X_test.setter
    def lv_X_test(self, value):
        self._lv_X_test = value
    @property
    def lv_X_train_scaled(self):
        return self.lv_X_train_scaled
    @lv_X_train_scaled.setter

    def lv_X_train_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.fit_transform(self.lv_X_train)
        self._lv_X_train_scaled = value

    @property
    def lv_X_test_scaled(self):
        return self.lv_X_test_scaled
    @lv_X_test_scaled.setter
    def lv_X_test_scaled(self, value):
        scaler = StandardScaler()
        value = scaler.transform(self.lv_X_test)
        self._lv_X_test_scaled = value

    @property
    def lv_y_train(self):
        return self.lv_y_train
    @lv_y_train.setter
    def lv_y_train(self, value):
        self._lv_y_train = value

    @property
    def lv_y_test(self):
        return self.lv_y_test
    @lv_y_test.setter
    def lv_y_test(self, value):
        self._lv_y_test = value
#--------------------------------------------------------------------
# create method  "dl_split_data_sets".
# class:cmqlmlsetup
# usage: mql data
# \pdlsplit data
#--------------------------------------------------------------------
    def dl_split_data_sets(self, lp_df, lv_X=None, lv_y=None, lp_test_size=0.2, lp_shuffle = False, lp_prog = 1):
        if lv_X is None:
            lv_X = []
        if lv_y is None:
            lv_y = []
        lv_X = lp_df[['close']]
        lv_y = lp_df['target']

        lv_X_train,lv_y_train,lv_X_test,lv_y_test =  train_test_split(lv_X, lv_y, test_size=lp_test_size, shuffle=lp_shuffle)
        if lp_prog == 1:
            return lv_X_train
        if lp_prog == 2:
            return lv_y_train
        if lp_prog == 3:
            return lv_X_test
        if lp_prog == 4:
            return lv_y_test
#--------------------------------------------------------------------
# create method  "dl_train_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model_scaled(self):
        # meta names
        scaler = StandardScaler()
        return scaler.fit_transform(self)

#--------------------------------------------------------------------
# create method  "dl_test_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#----------test----------------------------------------------------
    def dl_test_model_scaled(self):
        # meta names
        scaler = StandardScaler()
        return scaler.fit_transform(self)


#--------------------------------------------------------------------
# create method  "dl_build_neuro_network".
# class: cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_build_neuro_network(self, lp_X_train=None, lp_optimizer = 'adam', lp_loss = 'mean_squared_error'):
        if lp_X_train is None:
            lp_X_train = []
        # Build a neural network model
        model=None
        model = Sequential()
        model.add(
                Dense(
                128,
                activation='relu',
                input_shape=(lp_X_train.shape[1],),
                kernel_regularizer=l2(self),
            )
        )
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(self)))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(self)))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(self)))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        #model=model.compile(optimizer=lp_optimizer, loss=lp_loss)
        return model


# create method  "dl_train_model".
# class:cmqlmlsetup  usage: mql data
# \param  var
#--------------------------------------------------------------------

    def dl_train_model(self, lp_X_train_scaled=None, lp_y_train=None, lp_epoch = 1, lp_batch_size = 256, lp_validation_split = 0.2, lp_verbose =1):
        if lp_X_train_scaled is None:
            lp_X_train_scaled = []
        if lp_y_train is None:
            lp_y_train = []
        lp_X_train_scaled = np.stack(lp_X_train_scaled)
        lp_y_train = np.stack(lp_y_train)
        self.fit(
            lp_X_train_scaled,
            lp_y_train,
            epochs=lp_epoch,
            batch_size=lp_batch_size,
            validation_split=lp_validation_split,
            verbose=lp_verbose,
        )
        return self


#--------------------------------------------------------------------
# create method  "dl_predict_network"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_predict_values(self, lp_model, lp_seconds = 60):
        # Use the model to predict the next N instances
        lv_X_predict=[]
        lv_X_predict_scaled=[]

        lv_predictions = pd.DataFrame()
        lv_predictions=[]
        scaler = StandardScaler()
        # Empty DataFrame
        lv_X_predict = self.tail(lp_seconds)[['close']]
        lv_X_predict_scaled = scaler.transform(lv_X_predict)
        return lp_model.predict(lv_X_predict_scaled)

#--------------------------------------------------------------------
# create method  "dl_model_performance"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
#Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values.
# The lower the MSE, the better the model.

#Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values.
# Like MSE, lower values indicate better model performance.

#R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
# dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates
# a perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the
#target variable. Negative values indicate poor model performance.
    def dl_model_performance(self, lp_model, lp_X_test_scaled):
        # Calculate and print mean squared error
        mse = mean_squared_error(self, lp_model.predict(lp_X_test_scaled))
        print(f"\nMean Squared Error: {mse}")
        # Calculate and print mean absolute error
        mae = mean_absolute_error(self, lp_model.predict(lp_X_test_scaled))
        print(f"\nMean Absolute Error: {mae}")
        # Calculate and print R2 Score
        r2 = r2_score(self, lp_model.predict(lp_X_test_scaled))
        print(f"\nR2 Score: {r2}")
        return r2