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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

#-----------------------------------------------i
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self, X=None, y=None, X_train=None, X_train_scaled=None, X_test=None, X_test_scaled=None, y_train=None, y_test=None):
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
        self.X = X
        self.y = y
        self.X_train = X_train
        self.X_train_scaled = X_train_scaled
        self.X_test = X_test
        self.X_test_scaled = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test

    #--------------------------------------------------------------------
    # create method  "dl_split_data_sets".
    # class:cmqlmlsetup
    # usage: mql data
    # \pdlsplit data
    #--------------------------------------------------------------------
    def dl_split_data_sets(self, df, test_size=0.2, shuffle=False):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input df must be a pandas DataFrame")
        df=pd.DataFrame(df)
        X = df[['close']]  # Ensure X is a DataFrame
        y = df['target']   # Ensure y is a Series
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
        return X_train, X_test, y_train, y_test
    #--------------------------------------------------------------------
    # create method  "dl_train_model_scaled".
    # class:cmqlmlsetup
    # usage: mql data
    # \param  var
    #--------------------------------------------------------------------
    def dl_train_model_scaled(self, lpdf):
        scaler = StandardScaler()
        return scaler.fit_transform(lpdf)

    #--------------------------------------------------------------------
    # create method  "dl_test_model_scaled".
    # class:cmqlmlsetup
    # usage: mql data
    # \param  var
    #--------------------------------------------------------------------
    def dl_test_model_scaled(self, lpdf):
        scaler = StandardScaler()
        return scaler.fit_transform(lpdf)

    #--------------------------------------------------------------------
    # create method  "dl_model_performance"
    # class:cmqlmlsetup
    # usage: mql data
    # \pdl_build_neuro_network
    #--------------------------------------------------------------------
    def model_performance(self, model, X_test, y_test):
        # Predict the model
        predictions = model.predict(X_test)

        # If predictions are probabilities, convert them to class labels
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)

        # Determine if the problem is binary or multi-class
        average_type = 'binary' if len(set(y_test)) == 2 else 'weighted'

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average=average_type)
        recall = recall_score(y_test, predictions, average=average_type)
        f1 = f1_score(y_test, predictions, average=average_type)

        # Print performance metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1