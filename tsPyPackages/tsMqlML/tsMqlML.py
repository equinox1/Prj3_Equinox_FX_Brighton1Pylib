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
import pandas as pd
import tabulate
#+-------------------------------------------------------------------
# import ai packages scikit-learns
#+-------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split

#======================================================
# import ai packages tensorflow and keras libraries
#======================================================

import tensorflow as tf; tf.keras
from tensorflow.keras.regularizers import l2


#-----------------------------------------------
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self, df, X=None, y=None, X_train=None, X_train_scaled=None, X_test=None, X_test_scaled=None, y_train=None, y_test=None):
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
# create method  "dl_split_data_sets".
# class:cmqlmlsetup
# usage: mql data
# \pdlsplit data

#--------------------------------------------------------------------
    def dl_split_data_sets(df, X, y, test_size=0.2, shuffle = False, prog = 1):
        # sourcery skip: instance-method-first-arg-name
        X = df[['close']]
        y = df['target']
        # Split the data into training and testing sets
        X_train=[]
        X_test=[]
        y_train=[]
        y_test=[]
        X_train,y_train,X_test,y_test =  train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        if prog == 1:
            return X_train
        if prog == 2:
            return y_train
        if prog == 3:
            return X_test
        if prog == 4:
            return y_test
#--------------------------------------------------------------------
# create method  "dl_train_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model_scaled(df):
        # sourcery skip: instance-method-first-arg-name
        # meta names
        scaler = StandardScaler()
        return scaler.fit_transform(df)

#--------------------------------------------------------------------
# create method  "dl_test_model_scaled".
# class:cmqlmlsetup
# usage: mql data
# \param  var
#----------test----------------------------------------------------
    def dl_test_model_scaled(df):
        # sourcery skip: instance-method-first-arg-name
        scaler = StandardScaler()
        return scaler.fit_transform(df)


#--------------------------------------------------------------------
# create method  "dl_build_neuro_network".
# class: cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_build_neuro_network(p_k_reg, X_train, optimizer='adam', loss='mean_squared_error'):
        # sourcery skip: instance-method-first-arg-name
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(p_k_reg)),
                tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=l2(p_k_reg)),
                tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(p_k_reg)),
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=l2(p_k_reg)),
                tf.keras.layers.Dense(1, activation='linear')
            ])
            
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            return model


# create method  "dl_train_model".
# class:cmqlmlsetup  usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model(lp_model,lp_X_train_scaled, lp_y_train, epoch = 1, batch_size = 256, validation_split = 0.2, verbose =1):
        # sourcery skip: instance-method-first-arg-name
        lp_X_train_scaled= lp_X_train_scaled[:len(lp_y_train)]  # Truncate 'x' to match 'y'
        lp_model.fit(
            lp_X_train_scaled,
            lp_y_train,
            epochs=epoch,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose
        )
        return lp_model


#--------------------------------------------------------------------
# create method  "dl_predict_network"
# class:cmqlmlsetup
# usage: mql data
# \pdl_build_neuro_network
#--------------------------------------------------------------------
    def dl_predict_values(df, model, seconds = 60):
        # sourcery skip: instance-method-first-arg-name
        # Use the model to predict the next N instances
        X_predict=[]
        X_predict_scaled=[]
        predictions = pd.DataFrame()
        predictions=[]
        scaler = StandardScaler()
        # Empty DataFrame
        X_predict = df.tail(seconds)[['close']]
        X_predict_scaled = scaler.transform(X_predict)
        return model.predict(X_predict_scaled)

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
    def dl_model_performance(df, model, X_test_scaled):
        # sourcery skip: instance-method-first-arg-name
        # Calculate and print mean squared error
        mse = mean_squared_error(df, model.predict(X_test_scaled))
        print(f"\nMean Squared Error: {mse}")
        # Calculate and print mean absolute error
        mae = mean_absolute_error(df, model.predict(X_test_scaled))
        print(f"\nMean Absolute Error: {mae}")
        # Calculate and print R2 Score
        r2 = r2_score(df, model.predict(X_test_scaled))
        print(f"\nR2 Score: {r2}")
        return r2