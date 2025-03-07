#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.2
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
from datetime import datetime
import textwrap
from tabulate import tabulate

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")


class CDMLProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        # Initialize print parameters
        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)

        # Local data parameters
        self.lp_utc_from = kwargs.get('lp_utc_from', datetime.utcnow())
        self.lp_utc_to = kwargs.get('lp_utc_to', datetime.utcnow())
        self.mp_unit = kwargs.get('UNIT', {})

        # Initialize run state parameters
        self._initialize_mql()
        self._set_envmgr_params(kwargs)
        self._set_global_parameters(kwargs)
        # Set primary symbol and timeframe before ML features are set.
        self.lp_app_primary_symbol = kwargs.get(
            'lp_app_primary_symbol',
            self.params.get('app', {}).get('mp_app_primary_symbol', 'EURUSD')
        )
        self.lp_timeframe = kwargs.get(
            'lp_timeframe',
            self.params.get('data', {}).get('mp_data_timeframe', 'H4')
        )
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}, Timeframe: {self.lp_timeframe}")
        
        self._set_ml_features(kwargs)

    def _initialize_mql(self):
        """Initialize MetaTrader5 module if available."""
        self.os_platform = platform_checker.get_platform()
        self.loadmql = pchk.check_mql_state()
        logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")

        if self.loadmql:
            try:
                global mt5
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MetaTrader5. Error: {mt5.last_error()}")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5: {e}")

    def _set_envmgr_params(self, kwargs):
        """Extract environment parameters."""
        override_config = CMqlOverrides()
        self.params = override_config.env.all_params()
        logger.info(f"All Parameters: {self.params}")
        self.params_sections = self.params.keys()
        logger.info(f"PARAMS SECTIONS: {self.params_sections}")

        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        # Implementation of global parameter setting as required.
        pass

    def _set_ml_features(self, kwargs):
        """Extract and set machine learning features."""
        # Get feature configuration dictionary, if available
        self.ml_features_config = self.ml_params.get('mp_features', {})

        # Explicitly get Feature1 column from ml_params or fallback to configuration or default value.
        self.feature1 = self.ml_params.get('Feature1', self.ml_features_config.get('Feature1', 'Feature1'))
        if self.feature1 is None:
            self.feature1 = 'Feature1'
        logger.info("Feature1: %s", self.feature1)

        # Explicitly get the scaled Feature1 column
        self.feature1_scaled = self.ml_params.get('Feature1_scaled', self.ml_features_config.get('Feature1_scaled', 'Feature1_Scaled'))
        if self.feature1_scaled is None:
            self.feature1_scaled = 'Feature1_Scaled'
        logger.info("Feature1_scaled: %s", self.feature1_scaled)

        # Explicitly get the label column
        self.label = self.ml_params.get('Label1', self.ml_features_config.get('Label1', 'Label'))
        if self.label is None:
            self.label = 'Label'
        logger.info("Label: %s", self.label)

        # Set input keys for the machine learning pipeline
        self.mp_ml_input_keyfeat = self.feature1
        self.mp_ml_input_keyfeat_scaled = self.feature1_scaled
        self.mp_ml_input_label = self.label

        logger.info("Machine learning features configuration: %s", self.ml_features_config)
        logger.info("Machine learning input key feature: %s", self.mp_ml_input_keyfeat)
        logger.info("Machine learning input key feature scaled: %s", self.mp_ml_input_keyfeat_scaled)
        logger.info("Machine learning input label: %s", self.mp_ml_input_label)

        # File parameters
        self.rownumber = self.ml_params.get('mp_rownumber', False)
        self.mp_data_filename1 = self.params.get('data', {}).get('mp_data_filename1', 'default1.csv')
        self.mp_data_filename2 = self.params.get('data', {}).get('mp_data_filename2', 'default2.csv')

        logger.info("Data filename1: %s", self.mp_data_filename1)
        logger.info("Data filename2: %s", self.mp_data_filename2)
        logger.info("Row number: %s", self.rownumber)

        # Machine learning parameters
        self.lookahead_periods = self.params.get('ml', {}).get('mp_lookahead_periods', 1)
        self.ma_window = self.params.get('ml', {}).get('mp_ml_tf_ma_windowin', 10)
        self.hl_avg_col = self.params.get('ml', {}).get('mp_ml_hl_avg_col', 'HL_Avg')
        self.ma_col = self.params.get('ml', {}).get('mp_ml_ma_col', 'MA')
        self.returns_col = self.params.get('ml', {}).get('mp_ml_returns_col', 'Returns')
        self.shift_in = self.params.get('ml', {}).get('mp_ml_tf_shiftin', 1)

        # Fixed keys (removed extra quotes)
        self.run_avg = self.params.get('ml', {}).get('mp_ml_run_avg', False)
        self.run_avg_scaled = self.params.get('ml', {}).get('mp_ml_run_avg_scaled', False)
        self.log_stationary = self.params.get('ml', {}).get('mp_ml_log_stationary', False)
        self.remove_zeros = self.params.get('ml', {}).get('mp_ml_remove_zeros', False)

        self.last_col = self.params.get('ml', {}).get('mp_ml_last_col', False)
        self.last_col_scaled = self.params.get('ml', {}).get('mp_ml_last_col_scaled', False)
        self.first_col = self.params.get('ml', {}).get('mp_ml_first_col', False)
        self.mp_ml_dropna = self.params.get('ml', {}).get('mp_ml_dropna', False)
        self.mp_ml_dropna_scaled = self.params.get('ml', {}).get('mp_ml_dropna_scaled', False)

        self.create_label = self.params.get('ml', {}).get('mp_ml_create_label', False)
        self.create_label_scaled = self.params.get('ml', {}).get('mp_ml_create_label_scaled', False)

        logger.info("Lookahead periods: %s", self.lookahead_periods)
        logger.info("Moving average window: %s", self.ma_window)
        logger.info("High-low average column: %s", self.hl_avg_col)
        logger.info("Moving average column: %s", self.ma_col)
        logger.info("Returns column: %s", self.returns_col)
        logger.info("Shift in: %s", self.shift_in)
        logger.info("Run average: %s", self.run_avg)
        logger.info("Run average scaled: %s", self.run_avg_scaled)
        logger.info("Log stationary: %s", self.log_stationary)
        logger.info("Remove zeros: %s", self.remove_zeros)
        logger.info("Last column: %s", self.last_col)
        logger.info("Last column scaled: %s", self.last_col_scaled)
        logger.info("First column: %s", self.first_col)
        logger.info("Create label: %s", self.create_label)
        logger.info("Create label scaled: %s", self.create_label_scaled)

        # Data parameters
        self.rownumber = self.params.get('data', {}).get('mp_data_rownumber', False)
        self.lp_data_rows = kwargs.get('lp_data_rows', self.params.get('data', {}).get('mp_data_rows', 1000))
        self.lp_data_rowcount = kwargs.get('lp_data_rowcount', self.params.get('data', {}).get('mp_data_rowcount', 10000))

        # Derived filenames
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_data_path', 'Mql5Data'))
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

    def get_feature_columns(self, feature_name="Feature1"):
        return_value = self.ml_features_config.get(feature_name, None)
        if return_value is not None:
            return [f"{return_value}"]

    def get_scaled_feature_columns(self, feature_name="Feature1_Scaled"):
        return_value = self.ml_features_config.get(feature_name, None)
        if return_value is not None:
            return [f"{return_value}"]

    def get_label_columns(self, label_name="Label"):
        return_value = self.ml_features_config.get(label_name, None)
        if return_value is not None:
            return [f"{return_value}"]

    def create_XY_unscaled_feature_sequence(self, data, target_col, window_size):
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
        
        # Use default if target_col is None
        if target_col is None:
            if hasattr(self, 'mp_ml_input_keyfeat') and self.mp_ml_input_keyfeat is not None:
                target_col = self.mp_ml_input_keyfeat
            else:
                target_col = "Feature1"
            logger.warning("No target column specified. Falling back to default: %s", target_col)
        
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
            # If data is not a DataFrame, assume target_col is an index
            try:
                target_col_index = int(target_col)
            except ValueError:
                raise ValueError("When data is not a DataFrame, target_col must be an integer index.")
        
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
        features_count = len(self.ml_features_config.get("mp_ml_input_features", {}))
        labels_count = len(self.ml_features_config.get("mp_ml_output_label", {}))
       
        past_width = int(self.ml_features_config.get("pasttimeperiods", 24)) * timeval
        future_width = int(self.ml_features_config.get("futuretimeperiods", 24)) * timeval
        pred_width = int(self.ml_features_config.get("predtimeperiods", 1)) * timeval

        logger.info("Past Width: %s, Future Width: %s, Prediction Width: %s, Features Count: %s, Labels Count: %s",
                    past_width, future_width, pred_width, features_count, labels_count)

        return past_width, future_width, pred_width, features_count, labels_count

    def split_dataset(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=None):
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=random_state)
        val_test_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_test_ratio, random_state=random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def preprocess_data(self, X):
        """Convert timestamp columns to numeric format if they exist and ensure the data is numeric."""
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
