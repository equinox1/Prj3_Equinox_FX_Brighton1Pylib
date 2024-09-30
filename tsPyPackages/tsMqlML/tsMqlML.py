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
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
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
# create method  "dl_split_data_sets".
# class:cmqlmlsetup
# usage: mql data
# \pdlsplit data

#--------------------------------------------------------------------
    def dl_split_data_sets(self,df, X, y, test_size=0.2, shuffle = False, prog = 1):
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
"""
Key Parameters
units: The number of neurons or units in the dense layer. This is the dimensionality of the output space.
activation: The activation function to apply to the output. 

Common activation functions include:
====================================
relu (Rectified Linear Unit)
sigmoid
softmax
tanh
linear (no activation)
use_bias: Boolean, whether to include a bias term in the layer (default: True).

kernel_initializer: Specifies the initializer for the kernel weights matrix, which
determines how the weights are initialized before training.
bias_initializer: Specifies the initializer for the bias vector.
kernel_regularizer: Optional regularizer function applied to the kernel weights matrix 
(for regularization like L1 or L2).
bias_regularizer: Optional regularizer applied to the bias vector.
activity_regularizer: Regularizer applied to the output (activation) of the layer.
kernel_constraint: Constraint function applied to the kernel weights matrix.
bias_constraint: Constraint function applied to the bias vector.
"""

    def dl_build_neuro_network(self,lp_k_reg, X_train, y_train,optimizer='adam',lp_act1 = 'relu', lp_act2 = 'linear',lp_metric = 'accuracy', lp_loss='mean_squared_error',cells1=128,cells2=256, cells3=128,cells4=64,cells5=1):
        # sourcery skip: instance-method-first-arg-name
        bmodel = tf.keras.models.Sequential([
            tf.keras.layers.Dense(cells1, activation=lp_act1, input_shape=(X_train.shape[1],), kernel_regularizer=tf.l2(lp_k_reg)),
            tf.keras.layers.Dense(cells2, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
            tf.keras.layers.Dense(cells3, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
            tf.keras.layers.Dense(cells4, activation=lp_act1, kernel_regularizer=tf.l2(lp_k_reg)),
            tf.keras.layers.Dense(cells5, activation=lp_act2)
           ])
        bmodel.compile(optimizer=optimizer, loss=fn_loss, metrics=[lp_metric])
        return bmodel

#--------------------------------------------------------------------
# create method  "dl_lstm_model".
# class:cmqlmlsetup  usage: mql data
# \param  var
#--------------------------------------------------------------------
#model = tf.keras.models.Sequential([
#  tf.keras.layers.Conv1D(filters=32, kernel_size=3,
#                      strides=1, padding="causal",
#                      activation="relu",
#                      input_shape=[None, 1]),
#  tf.keras.layers.LSTM(32, return_sequences=True),
#  tf.keras.layers.LSTM(32, return_sequences=True),
#  tf.keras.layers.Dense(1),
#  tf.keras.layers.Lambda(lambda x: x * 200)
#])

optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset,epochs=500)

#--------------------------------------------------------------------
# create method  "dl_train_model".
# class:cmqlmlsetup  usage: mql data
# \param  var
#--------------------------------------------------------------------
    def dl_train_model(self,lp_model,lp_X_train_scaled, lp_y_train, epoch = 1, batch_size = 256, validation_split = 0.2, verbose =1):
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
    def dl_predict_values(self,df, model, seconds = 60):
        # sourcery skip: instance-method-first-arg-name
        # Use the model to predict the next N instances
        X_predict=[]
        X_predict_scaled=[]
        predictions = pd.DataFrame()
        predictions=[]
        scaler = StandardScaler()
        # Empty DataFrame
        print("dftail:",df.tail(seconds)[['close']].values)
        X_predict = df.tail(seconds)[['close']].values
        scaler.fit(X_predict)
        X_predict_scaled = scaler.transform(X_predict)
        return model.predict(X_predict_scaled)

