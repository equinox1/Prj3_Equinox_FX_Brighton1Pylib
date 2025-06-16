#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.4 (Corrected ML parameter loading from mltune_params)
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

from tsMqlSetup import CMqlSetup # Correctly import the class

    
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import textwrap
from tabulate import tabulate
from pathlib import Path

import logging

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides

# -- start of logging setup --
from tsMqlOverrides import CMqlOverrides
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get('mltune', {})
base_params = mql_overrides.env.all_params().get('base', {}) # Get base parameters

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__, 
    loglevel='INFO',
    logfile='tsneuropredict_app.log'
)



# Global tuner configuration
TUNER_ID = os.environ.get('TUNER_ID', 'default_worker')
LOGDIR = Path(app_params.get('LOGDIR', 'Logdir')) # Correctly get LOGDIR as Path
ORACLE_DIR = LOGDIR / "tsOracle" # Oracle working directory
MODEL_DIR = Path(base_params.get('mp_glob_sub_ml_src_modeldata', 'PythonLib/tsModelData')) # Path to save models
MODEL_NAME = tune_params.get('ml_model_name', 'tsneuromodel')

class CDMLProcess:
    def __init__(self, df: pd.DataFrame, input_key_feature: str = 'Close', label_key_feature: str = 'Label', history_size: int = 5, **kwargs):
        """
        Initializes the CDMLProcess with a DataFrame and key feature names.

        Args:
            df (pd.DataFrame): The input DataFrame.
            input_key_feature (str): The name of the primary feature column to be used as input.
            label_key_feature (str): The name of the column to be used as the label (target).
            history_size (int): The number of past observations to use for each sample (sequence length).
            **kwargs: Arbitrary keyword arguments.
        """
        if not isinstance(df, pd.DataFrame) or df.empty:
            logger.error("Input 'df' must be a non-empty Pandas DataFrame.")
            raise ValueError("Input 'df' must be a non-empty Pandas DataFrame.")

        self.df = df.copy()  # Work on a copy to avoid modifying the original DataFrame
        self.input_key_feature = input_key_feature
        self.label_key_feature = label_key_feature
        self.history_size = history_size # N past observations
        self.output_size = 1 # Predict 1 future step (e.g., next candle's value or return)
        self.num_features = None  # Will be set after feature selection
        self.train_split_ratio = kwargs.get('train_split_ratio', 0.8)
        self.val_split_ratio = kwargs.get('val_split_ratio', 0.1) # New parameter for validation split
        self.test_split_ratio = kwargs.get('test_split_ratio', 0.1) # Remaining for test
        self.scaler = StandardScaler()

        # Load environment parameters using CMqlOverrides
        self.env = CMqlEnvMgr() # Assuming CMqlEnvMgr is accessible
        self.params = self.env.all_params()
        
        # Access specific parameter sections
        self.app_params = self.params.get("app", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {}) # This might be empty if no 'ml' section in config
        self.mltune_params = self.params.get("mltune", {}) # This is where ML params are expected
        self.base_params = self.params.get("base", {})


        # Set ML features and other parameters from config
        self._set_ml_features()

        # Ensure MetaTrader5 is initialized
        self._initialize_mql()
        
        logger.info("CDMLProcess initialized.")
        logger.info(f"Input Key Feature: {self.input_key_feature}")
        logger.info(f"Label Key Feature: {self.label_key_feature}")
        logger.info(f"History Size: {self.history_size}")


    def _initialize_mql(self):
        """Initialize MetaTrader5 if available.
           This method assumes MetaTrader5 is already initialized by the calling script
           and only focuses on importing the module if loadmql is True.
        """
        # pchk, os_platform, loadmql are assumed to be initialized globally
        # or passed as args if not globally available.
        # Given they are defined outside the class, they should be accessible.
        pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = pchk.check_mql_state()
        logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")
        if self.loadmql: # Use the globally determined loadmql state
            try:
                global mt5
                import MetaTrader5 as mt5 # Import mt5
                # Removed the mt5.initialize() call from here.
                # It is assumed that mt5.initialize() is called once at the
                # entry point of the application (e.g., in chief/worker scripts)
                logger.info("MetaTrader5 module expected to be initialized by caller and imported.")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5: {e}")
        else:
            logger.info("MetaTrader5 module not to be loaded based on loadmql state.")

    def _set_envmgr_params(self):
        """No need to re-set env params here, already done in __init__."""
        pass

    def _set_ml_features(self):
        """Sets ML related features and parameters from loaded configuration."""
        # Use self.mltune_params for ML feature configuration as per config.yaml structure
        self.ml_features_config = self.mltune_params.get('mp_features', {}) # If 'mp_features' is ever nested under mltune
        logger.info(f"ML Features Config (from mltune_params): {self.ml_features_config}")

        # Override input_key_feature and label_key_feature if provided in mltune_params
        # If not, they will retain the defaults from __init__
        self.input_key_feature = self.mltune_params.get('mp_ml_input_keyfeat', self.input_key_feature)
        self.label_key_feature = self.mltune_params.get('mp_ml_input_label', self.label_key_feature)
        logger.info(f"Input Key Feature: {self.input_key_feature}")
        # Note: mp_ml_input_keyfeat_scaled is likely just a name in config, not meant to change input_key_feature directly.
        logger.info(f"Input Key Feature Scaled (from config): {self.mltune_params.get('mp_ml_input_keyfeat_scaled', 'Close_Scaled')}")
        logger.info(f"Input Label: {self.label_key_feature}")

        # Features for processing - now correctly pulling from mltune_params
        self.run_hl_avg = self.mltune_params.get('mp_ml_run_avg', False)
        self.run_ma = self.mltune_params.get('mp_ml_run_ma', False)
        self.run_returns = self.mltune_params.get('mp_ml_run_returns', False)
        self.run_returns_scaled = self.mltune_params.get('mp_ml_run_returns_scaled', False)
        self.run_returns_shifted = self.mltune_params.get('mp_ml_run_returns_shifted', False)
        self.run_returns_shifted_scaled = self.mltune_params.get('mp_ml_run_returns_shifted_scaled', False)
        self.run_label = self.mltune_params.get('mp_ml_run_label', False)
        self.run_label_scaled = self.mltune_params.get('mp_ml_run_label_scaled', False)
        self.run_label_shifted = self.mltune_params.get('mp_ml_run_label_shifted', False)
        # This one defaults to True in the chief/worker scripts if not found, but it should come from config.
        # Ensure your config.yaml has 'mp_ml_run_label_shifted_scaled: True' under ML_TUNING_PARAMS.
        self.run_label_shifted_scaled = self.mltune_params.get('mp_ml_run_label_shifted_scaled', True) 
        self.mp_ml_log_stationary = self.mltune_params.get('mp_ml_log_stationary', False)
        self.mp_ml_remove_zeros = self.mltune_params.get('mp_ml_remove_zeros', False)
        self.mp_ml_last_col = self.mltune_params.get('mp_ml_last_col', False)
        self.mp_ml_last_col_scaled = self.mltune_params.get('mp_ml_last_col_scaled', False)
        self.mp_ml_first_col = self.mltune_params.get('mp_ml_first_col', False)
        self.mp_ml_dropna = self.mltune_params.get('mp_ml_dropna', True)
        self.mp_ml_dropna_scaled = self.mltune_params.get('mp_ml_dropna_scaled', True)

        # Column names from config - now correctly pulling from mltune_params
        self.hl_avg_col = self.mltune_params.get('mp_ml_hl_avg_col', 'HL_Avg')
        self.ma_col = self.mltune_params.get('mp_ml_ma_col', 'SMA')
        self.returns_col = self.mltune_params.get('mp_ml_returns_col', 'LogReturns')
        self.returns_col_scaled = self.mltune_params.get('mp_ml_returns_col_scaled', 'LogReturns_Scaled')
        self.label_col = self.mltune_params.get('mp_ml_label_col', 'Label')
        self.label_col_scaled = self.mltune_params.get('mp_ml_label_col_scaled', 'Label_Scaled')
        self.label_shifted_col = self.mltune_params.get('mp_ml_label_shifted_col', 'Label_Shifted')
        self.label_shifted_scaled_col = self.mltune_params.get('mp_ml_label_shifted_scaled_col', 'Label_Shifted_Scaled')
        
        # Window sizes for ML features - now correctly pulling from mltune_params
        self.past_window = self.mltune_params.get('mp_ml_tf_past_window', 24)
        self.future_window = self.mltune_params.get('mp_ml_tf_future_window', 24)
        self.ma_window = self.mltune_params.get('mp_ml_tf_ma_windowin', 24)
        self.shift_in = self.mltune_params.get('mp_ml_tf_shiftin', 1)
        self.prediction_window = self.mltune_params.get('mp_ml_tf_prediction_window', 1) # For target label


        # Ensure original columns exist or log a warning
        if 'Open' not in self.df.columns:
            logger.warning("Original 'Open' column not found in DataFrame. Some features might not be computable.")
        if 'High' not in self.df.columns:
            logger.warning("Original 'High' column not found in DataFrame. Some features might not be computable.")
        if 'Low' not in self.df.columns:
            logger.warning("Original 'Low' column not found in DataFrame. Some features might not be computable.")
        if 'Close' not in self.df.columns:
            logger.warning("Original 'Close' column not found in DataFrame. Some features might not be computable.")

        logger.info("Machine learning features configured.")

    def create_sequences(self, data, history_size):
        """
        Create sequences from the data for time series forecasting.

        Args:
            data (np.array): The input data (features).
            history_size (int): The number of past observations to use for each sample.

        Returns:
            np.array: A 3D numpy array of sequences (samples, history_size, num_features).
        """
        xs = []
        for i in range(len(data) - history_size + 1): # Adjusted range to ensure enough data for sequences
            x = data[i:(i + history_size)]
            xs.append(x)
        return np.array(xs)

    def create_labels(self, data, history_size, prediction_window):
        """
        Create labels (targets) from the data, shifted by prediction_window.

        Args:
            data (np.array): The target data.
            history_size (int): The length of the input sequences.
            prediction_window (int): How many steps into the future to predict.

        Returns:
            np.array: A 1D numpy array of labels.
        """
        ys = []
        # Ensure there are enough future data points for the label
        for i in range(len(data) - history_size - prediction_window + 1):
            y = data[i + history_size + prediction_window -1] # Adjusted to correctly get the future label
            ys.append(y)
        return np.array(ys)
    

    def create_shifted_scaled_label(self, df: pd.DataFrame, source_col: str, new_label_col: str, shift: int, scaler: StandardScaler) -> pd.DataFrame:
        """
        Creates a shifted and scaled label column based on the future percentage change
        of a source column.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            source_col (str): The column to base the label on (e.g., 'Close').
            new_label_col (str): The name for the new shifted and scaled label column.
            shift (int): The number of periods to shift for the future percentage change.
            scaler (StandardScaler): The scaler to use for scaling the label.
            
        Returns:
            pd.DataFrame: DataFrame with the new label column.
        """
        if source_col not in df.columns:
            logger.warning(f"Source column '{source_col}' not found for label creation. Skipping.")
            return df

        # Calculate future percentage change
        # Ensure numeric type and handle potential NaNs before calculating
        temp_series = pd.to_numeric(df[source_col], errors='coerce')
        temp_series = temp_series.ffill().bfill() # Fill any NaNs to prevent issues in pct_change

        # Calculate percentage change and then shift it *backwards* by `shift`
        # so that the value aligns with the current row's features.
        # If shift = 1, the label for row `i` will be the % change from `i` to `i+1`.
        # Ensure it's numeric before scaling.
        # Add a small epsilon to avoid division by zero if values can be zero.
        epsilon = 1e-9
        future_pct_change = (temp_series.shift(-shift) / (temp_series + epsilon) - 1) * 100
        
        # Scale the future percentage change
        # Reshape for scaler (needs 2D array)
        scaled_label = scaler.fit_transform(future_pct_change.values.reshape(-1, 1)).flatten()
        
        df[new_label_col] = scaled_label
        
        logger.info(f"Created shifted and scaled label: '{new_label_col}' based on '{source_col}'.")
        return df


    def process_ml_data(self) -> pd.DataFrame:
        """
        Applies various ML-related transformations to the DataFrame based on configuration.
        This includes creating features, scaling, and handling missing values.
        """
        ldf = self.df.copy() # Use a copy for processing
        
        # Ensure numeric types for relevant columns before calculations
        numeric_cols_for_processing = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols_for_processing:
            if col in ldf.columns:
                ldf[col] = pd.to_numeric(ldf[col], errors='coerce')
                # Fill NaNs in these columns before calculations to avoid propagate
                ldf[col].fillna(ldf[col].mean(), inplace=True) 

        # Create HL_Avg column if enabled
        if self.run_hl_avg and 'High' in ldf.columns and 'Low' in ldf.columns:
            ldf[self.hl_avg_col] = (ldf['High'] + ldf['Low']) / 2
            logger.info(f"Created HL_Avg column: {self.hl_avg_col}.")

        # Create SMA column if enabled
        if self.run_ma and self.input_key_feature in ldf.columns:
            ldf[self.ma_col] = ldf[self.input_key_feature].rolling(window=self.ma_window, min_periods=1).mean()
            logger.info(f"Created SMA column with window {self.ma_window}.")
        
        # Create LogReturns column if enabled
        if self.run_returns and self.input_key_feature in ldf.columns:
            # Add a small epsilon to avoid log(0) issues
            epsilon = 1e-9
            ldf[self.returns_col] = np.log(ldf[self.input_key_feature] / ldf[self.input_key_feature].shift(self.shift_in) + epsilon)
            logger.info(f"Created LogReturns column.")

        # Create Label column if enabled (shifted and scaled)
        # This will be the primary label for the model
        if self.run_label_shifted_scaled and self.input_key_feature in ldf.columns:
            ldf = self.create_shifted_scaled_label(ldf, self.input_key_feature, self.label_shifted_scaled_col, self.prediction_window, self.scaler)
            self.label_key_feature = self.label_shifted_scaled_col # Ensure the label_key_feature is updated
        elif self.run_label and self.input_key_feature in ldf.columns:
            # Fallback for simpler label if shifted_scaled is not enabled
            ldf[self.label_col] = ldf[self.input_key_feature].shift(-self.prediction_window)
            logger.info(f"Created basic label column: {self.label_col}.")
            self.label_key_feature = self.label_col

        if self.label_key_feature not in ldf.columns:
            logger.warning(f"Label column '{self.label_key_feature}' not found or created. Ensure it exists in input data or is generated by other means.")

        # Handle NaNs before scaling
        if self.mp_ml_dropna:
            initial_rows = len(ldf)
            ldf.dropna(inplace=True)
            if len(ldf) < initial_rows:
                logger.info(f"Dropped {initial_rows - len(ldf)} rows containing NaN values.")
        
        # Identify numerical columns for scaling, excluding the datetime index if any
        numerical_cols = ldf.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude the label column if it's already scaled or should not be scaled with features
        if self.label_key_feature in numerical_cols:
            numerical_cols.remove(self.label_key_feature)
        
        # Apply scaling to selected numerical columns
        if not ldf[numerical_cols].empty:
            ldf[numerical_cols] = self.scaler.fit_transform(ldf[numerical_cols])
            for col in numerical_cols:
                ldf.rename(columns={col: f"{col}_Scaled"}, inplace=True)
                logger.info(f"Scaled column: {col} -> {col}_Scaled")
            
        # Dynamically select features for model input
        # Start with base features, then add engineered features if they exist
        selected_features = []
        if f"{self.input_key_feature}_Scaled" in ldf.columns:
            selected_features.append(f"{self.input_key_feature}_Scaled")
        elif self.input_key_feature in ldf.columns: # If not scaled, use original
            selected_features.append(self.input_key_feature)

        if self.run_returns and f"{self.returns_col}_Scaled" in ldf.columns:
            selected_features.append(f"{self.returns_col}_Scaled")
        elif self.run_returns and self.returns_col in ldf.columns:
            selected_features.append(self.returns_col)

        if self.run_ma and f"{self.ma_col}_Scaled" in ldf.columns:
            selected_features.append(f"{self.ma_col}_Scaled")
        elif self.run_ma and self.ma_col in ldf.columns:
            selected_features.append(self.ma_col)
        
        self.features_for_model = selected_features
        logger.info(f"Final features selected for model input: {self.features_for_model}")

        # Final dropna after all feature engineering
        if self.mp_ml_dropna_scaled:
            initial_rows = len(ldf)
            ldf.dropna(inplace=True)
            if len(ldf) < initial_rows:
                logger.info(f"Dropped {initial_rows - len(ldf)} rows containing NaN values after scaling.")

        self.df = ldf # Update the internal DataFrame
        return ldf


    def create_datasets(self):
        """
        Creates X (features) and y (labels) datasets from the processed DataFrame.
        This method must be called AFTER process_ml_data to ensure features and labels are ready.
        """
        # Ensure the label column exists before proceeding
        if self.label_key_feature not in self.df.columns:
            logger.error(f"Required label column '{self.label_key_feature}' not found in the processed DataFrame. Cannot create X, y sequences.")
            return None, None
        
        # Ensure selected features are in the DataFrame and are numeric
        for feature in self.features_for_model:
            if feature not in self.df.columns:
                logger.error(f"Selected feature '{feature}' not found in the processed DataFrame.")
                return None, None
            if not pd.api.types.is_numeric_dtype(self.df[feature]):
                logger.error(f"Feature '{feature}' is not numeric. Cannot create sequences.")
                return None, None
        
        # Extract features and labels
        X_data = self.df[self.features_for_model].values
        y_data = self.df[self.label_key_feature].values

        # Remove any NaN values that might have been introduced by shifting/label creation
        # This is a critical step to ensure that create_sequences and create_labels work correctly
        # and that the shapes match.
        combined_data = pd.DataFrame(X_data, columns=self.features_for_model)
        combined_data[self.label_key_feature] = y_data
        combined_data.dropna(inplace=True)

        if combined_data.empty:
            logger.error("Combined data (features and labels) is empty after dropping NaNs. Cannot create sequences.")
            return None, None

        X_data_clean = combined_data[self.features_for_model].values
        y_data_clean = combined_data[self.label_key_feature].values


        # Create sequences for X and corresponding labels for y
        # Adjusting the range to align X and y correctly
        X_sequences = self.create_sequences(X_data_clean, self.history_size)
        y_labels = self.create_labels(y_data_clean, self.history_size, self.prediction_window)

        # After creating sequences and labels, their lengths must match
        # The number of samples for X_sequences is `len(data) - history_size + 1`
        # The number of samples for y_labels is `len(data) - history_size - prediction_window + 1`
        # To align them, we need to take the minimum length.
        min_samples = min(len(X_sequences), len(y_labels))
        
        if min_samples == 0:
            logger.error("No samples available after aligning X and y sequences. Check history_size and prediction_window relative to data length.")
            return None, None

        X_final = X_sequences[:min_samples]
        y_final = y_labels[:min_samples]


        self.num_features = X_final.shape[2] if X_final.ndim == 3 else X_final.shape[1]
        logger.info(f"X (features) shape: {X_final.shape}, y (labels) shape: {y_final.shape}")
        
        return X_final, y_final

    def prepare_tensorflow_datasets(self, X, y, shuffle_buffer_size=1000):
        """
        Splits data into train, validation, and test sets and prepares TensorFlow datasets.

        Args:
            X (np.array): Features data.
            y (np.array): Labels data.
            shuffle_buffer_size (int): Buffer size for shuffling training data.

        Returns:
            tuple: (train_dataset, val_dataset, test_dataset) as tf.data.Dataset objects.
        """
        if X is None or y is None or X.size == 0 or y.size == 0:
            logger.error("Input X or y is empty for TensorFlow dataset preparation.")
            return None, None, None

        # Calculate split sizes
        total_samples = len(X)
        train_samples = int(total_samples * self.train_split_ratio)
        val_samples = int(total_samples * self.val_split_ratio)
        # Test samples take the remainder
        test_samples = total_samples - train_samples - val_samples

        # Ensure no negative counts
        if train_samples < 0: train_samples = 0
        if val_samples < 0: val_samples = 0
        if test_samples < 0: test_samples = 0
        
        if train_samples + val_samples + test_samples == 0:
            logger.error("Insufficient data to create train, validation, and test sets with the given ratios.")
            return None, None, None


        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Train samples: {train_samples}")
        logger.info(f"Validation samples: {val_samples}")
        logger.info(f"Test samples: {test_samples}")

        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_samples, random_state=self.mltune_params.get('seed', 42)
        )
        # Re-calculate val_size relative to the remaining train_val set
        val_size_relative = val_samples / (train_samples + val_samples) if (train_samples + val_samples) > 0 else 0

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_relative, random_state=self.mltune_params.get('seed', 42)
        )
        
        # Convert to tf.data.Dataset
        batch_size = self.mltune_params.get('batch_size', 32)
        buffer_size = tf.data.AUTOTUNE # Use tf.data.AUTOTUNE for optimal performance

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size).batch(batch_size).prefetch(buffer_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(buffer_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(buffer_size)

        logger.info("TensorFlow datasets successfully prepared and batched.")
        return train_dataset, val_dataset, test_dataset


    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluates the trained model and computes various metrics.
        
        Args:
            model (tf.keras.Model): The trained TensorFlow Keras model.
            X_test (np.array): Test features.
            y_test (np.array): True test labels.
        
        Returns:
            dict: A dictionary of evaluation metrics.
        """
        try:
            # Predict on the test set
            predictions = model.predict(X_test)
            
            # For regression, predictions will likely be 2D (samples, 1). Flatten for metric calculation.
            # For classification, you might need argmax if output is probabilities/logits.
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            elif predictions.ndim > 1: # For multi-class classification, get class with highest probability
                predictions = predictions.argmax(axis=1)

            # Ensure y_test is also flattened if it's 2D
            if y_test.ndim > 1 and y_test.shape[1] == 1:
                y_test_flat = y_test.flatten()
            else:
                y_test_flat = y_test

            # Determine average type for precision/recall/f1-score for classification tasks.
            # For regression, these metrics are not typically used, or require thresholding.
            # Assuming if the label is continuous, it's regression; if discrete, it's classification.
            # This logic might need refinement based on your specific problem (regression vs classification).
            if pd.api.types.is_numeric_dtype(y_test_flat) and len(np.unique(y_test_flat)) > 2:
                # This suggests regression or multi-class classification where direct metrics are needed.
                # Mean Squared Error (MSE), Mean Absolute Error (MAE), R2 Score are common for regression.
                metrics = {
                    "mse": mean_squared_error(y_test_flat, predictions),
                    "mae": mean_absolute_error(y_test_flat, predictions),
                    "r2_score": r2_score(y_test_flat, predictions)
                }
                logger.info(f"Regression Metrics: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2_score']:.4f}")
            else: # Likely classification if few unique values
                average_type = 'binary' if len(set(y_test_flat)) == 2 else 'weighted'
                metrics = {
                    "accuracy": accuracy_score(y_test_flat, predictions),
                    "precision": precision_score(y_test_flat, predictions, average=average_type, zero_division=0),
                    "recall": recall_score(y_test_flat, predictions, average=average_type, zero_division=0),
                    "f1_score": f1_score(y_test_flat, predictions, average=average_type, zero_division=0)
                }
                logger.info(f"Classification Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            return {}