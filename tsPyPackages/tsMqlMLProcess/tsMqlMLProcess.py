""#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr  # Ensure this exists
from tsMqlMLParams import CMqlEnvMLParams  # Ensure this exists
from tsMqlMLTunerParams import CMqlEnvMLTunerParams  # Ensure this exists

# Set up logger if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info("Running on: %s, loadmql state: %s", os_platform, loadmql)

class CDMLProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        self.env = CMqlEnvMgr()
        self.mlconfig = CMqlEnvMLParams()
        self.ml_tune_config = CMqlEnvMLTunerParams()
        
        self.mt5 = None
        if loadmql:
            try:
                import MetaTrader5 as mt5
                self.mt5 = mt5
                if not self.mt5.initialize():
                    logger.error("Failed to initialize MetaTrader5: %s", self.mt5.last_error())
            except ImportError:
                logger.error("MetaTrader5 module not found. Exiting...")
                sys.exit(1)

        self._set_envmgr_params(kwargs)
        self._set_ml_params(kwargs)
        self._set_global_parameters(kwargs)

    def _set_envmgr_params(self, kwargs):
        try:
            self.params = self.env.all_params()
        except Exception as e:
            logger.critical("Failed to initialize CMqlEnvMgr: %s", e)
            self.params = {}

        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

    def _set_ml_params(self, kwargs):
        try:
            self.FEATURES_PARAMS = self.mlconfig.get_features_params() or {}
            self.WINDOW_PARAMS = self.mlconfig.get_window_params() or {}
            self.DEFAULT_PARAMS = self.mlconfig.get_default_params() or {}
            self.TUNER_DEFAULT_PARAMS = self.ml_tune_config.get_default_params() or {}
        except Exception as e:
            logger.error("Error loading ML parameters: %s", e)
            self.FEATURES_PARAMS, self.WINDOW_PARAMS, self.DEFAULT_PARAMS, self.TUNER_DEFAULT_PARAMS = {}, {}, {}, {}

    def _set_global_parameters(self, kwargs):
        self.timeval = kwargs.get('timeval', 1)
        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)
        
        self.feature1 = self.FEATURES_PARAMS.get("Feature1", "KeyFeature")
        self.feature1_scaled = self.FEATURES_PARAMS.get("Feature1_scaled", "KeyFeature_Scaled")
        self.label = self.FEATURES_PARAMS.get("Label1", "Label")

    def get_feature_columns(self,feature_name = "Feature1"):
        return_value = self.FEATURES_PARAMS.get(feature_name, None)
        if return_value is not None:
            return [f"{return_value}"]

   
    def get_scaled_feature_columns(self,feature_name = "Feature1_Scaled"):
         return_value = self.FEATURES_PARAMS.get(feature_name, None)
         if return_value is not None:
               return [f"{return_value}"]

    def get_label_columns(self,label_name = "Label"):
         return_value = self.FEATURES_PARAMS.get(label_name, None)
         if return_value is not None:
               return [f"{return_value}"]


    def create_XY_unscaled_feature_sequence(self,data, target_col, window_size):
         """
         Create feature sequences (X) and target values (y) from time series data.
         
         Parameters:
         - data (array-like: np.ndarray, pd.DataFrame, or list): The input time series data.
         - target_col (str): The column name of the target variable.
         - window_size (int): The number of past observations to use for predicting the next value.
         
         Returns:
         - X (np.ndarray): Feature sequences of shape (num_samples, window_size, num_features).
         - y (np.ndarray): Target values of shape (num_samples,).
         """
         
         # Ensure window_size is an integer
         window_size = int(window_size)
         
         # Validate data type
         if not isinstance(data, (np.ndarray, pd.DataFrame, list)):
            raise ValueError(f"Expected data to be array-like, but got {type(data)}")
         
         # If data is a DataFrame, extract target column index and convert to NumPy array
         if isinstance(data, pd.DataFrame):
            if isinstance(target_col, list):
                  if len(target_col) != 1:
                     raise ValueError(f"Expected a single column name, but got {target_col}")
                  target_col = target_col[0]  # Extract the string from the list
            
            if target_col not in data.columns:
                  raise ValueError(f"Column '{target_col}' not found in DataFrame.")
            
            target_col_index = data.columns.get_loc(target_col)
            data = data.to_numpy()  # Convert DataFrame to NumPy array
         else:
            target_col_index = target_col  # If data is not a DataFrame, assume it's an index
         
         # Create feature sequences (X) and target values (y)
         X, y = [], []
         for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size, target_col_index])  # Predicting target column
         
         return np.array(X), np.array(y)

       
    def create_Xy_scaled_feature_sequence(self, df, past_window, future_window, feature_column='Close', target_column='Close_Scaled'):
         X, Y = [], []
         
         # Ensure past_window and future_window are integers
         past_window = int(past_window)
         future_window = int(future_window)
         
         for i in range(len(df) - past_window - future_window):
            past = df.loc[i:i + past_window - 1, target_column].values
            future = df.loc[i + past_window + future_window - 1, feature_column]
            X.append(past)
            Y.append(future)
         
         return np.array(X), np.array(Y)


    def create_ml_window(self, timeval):
        """Select the data file based on the DataFrame name."""
        features_count = len(self.FEATURES_PARAMS.get("mp_ml_input_features", {}))
        labels_count = len(self.FEATURES_PARAMS.get("mp_ml_output_label", {}))
       
        past_width = int(self.WINDOW_PARAMS.get("pasttimeperiods", 24)) * timeval
        future_width = int(self.WINDOW_PARAMS.get("futuretimeperiods", 24)) * timeval
        pred_width = int(self.WINDOW_PARAMS.get("predtimeperiods", 1)) * timeval

        logger.info("Past Width: %s, Future Width: %s, Prediction Width: %s, Features Count: %s, Labels Count: %s", past_width, future_width, pred_width, features_count, labels_count)

        return past_width, future_width, pred_width, features_count, labels_count

    def split_dataset(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state)
        val_test_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_test_ratio, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_data(self, X):
         """ Convert timestamp columns to numeric format if they exist and ensure the data is numeric. """
         if isinstance(X, pd.DataFrame):  
            for col in X.select_dtypes(include=['datetime64', 'object']).columns:
                  X[col] = pd.to_datetime(X[col], errors='coerce').astype(int) // 10**9  # Convert to Unix timestamp
            logger.info("Dataframe converted to Unix timestamp.")
         
         # Convert to numpy array and enforce dtype
         if isinstance(X, pd.Series):
            X = X.to_numpy()
         
         if isinstance(X, np.ndarray):
            X = X.astype(np.float32)  # Ensure proper dtype for TensorFlow
         
         return X

    def convert_to_tfds(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, batch_size=32, shuffle=True):
         X_train = self.preprocess_data(X_train)
         y_train = self.preprocess_data(y_train)

         X_val = self.preprocess_data(X_val) if X_val is not None else None
         y_val = self.preprocess_data(y_val) if y_val is not None else None

         X_test = self.preprocess_data(X_test) if X_test is not None else None
         y_test = self.preprocess_data(y_test) if y_test is not None else None

         logger.info(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
         logger.info(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")

         logger.info(f"X_train type: {type(X_train)}, dtype: {X_train.dtype if isinstance(X_train, np.ndarray) else 'N/A'}")
         logger.info(f"y_train type: {type(y_train)}, dtype: {y_train.dtype if isinstance(y_train, np.ndarray) else 'N/A'}")

         y_train = np.array(y_train).astype(np.float32)
         y_val = np.array(y_val).astype(np.float32) if y_val is not None else None
         y_test = np.array(y_test).astype(np.float32) if y_test is not None else None
         
         if np.isnan(X_train).any() or np.isnan(y_train).any():
            logger.info("Warning: NaN values detected in the dataset!")
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)

         train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
         if shuffle:
            train_ds = train_ds.shuffle(buffer_size=len(X_train))
         train_ds = train_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

         val_ds = None
         test_ds = None

         if X_val is not None and y_val is not None:
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

         if X_test is not None and y_test is not None:
            test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
            test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

         return train_ds, val_ds, test_ds


    def evaluate_model(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)
        average_type = 'binary' if len(set(y_test)) == 2 else 'weighted'
        
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average=average_type),
            "recall": recall_score(y_test, predictions, average=average_type),
            "f1_score": f1_score(y_test, predictions, average=average_type)
        }

        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
