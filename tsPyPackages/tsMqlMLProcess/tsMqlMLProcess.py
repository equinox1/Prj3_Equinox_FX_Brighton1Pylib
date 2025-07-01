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

# Load configuration
mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

# Corrected: Assign the logger instance to the 'logger' variable
logger = logging.getLogger(__name__)


class CDMLProcess:
    def __init__(self, df, all_params=None, project_dir=None): # Modified signature
        self.df = df
        self.all_params = all_params if all_params is not None else {} # Store all_params
        self.project_dir = project_dir # Store project_dir
        
        # Extract input_key_feature, label_key_feature, history_size from all_params
        # Providing default values in case they are not found in all_params
        app_params_local = self.all_params.get("app", {})
        data_params_local = self.all_params.get("data", {}) # Assuming data parameters are under 'data' key

        self.input_key_feature = app_params_local.get('mp_ml_input_keyfeat', 'Close') # Default value
        self.label_key_feature = app_params_local.get('mp_ml_input_label', 'Label') # Default value
        self.history_size = data_params_local.get('mp_data_history_size', 5) # Default value
        
        self.scaler_X = StandardScaler() # For scaling input features
        self.scaler_y = StandardScaler() # For scaling labels

    def create_datasets(self):
        """
        Processes the input DataFrame to create sequences for X and corresponding y targets.
        Handles scaling of both features and labels.
        """
        logger.info("Creating datasets...")

        # Ensure required columns exist
        if self.input_key_feature not in self.df.columns:
            logger.error(f"Input feature '{self.input_key_feature}' not found in DataFrame columns.")
            raise KeyError([self.input_key_feature])

        if self.label_key_feature not in self.df.columns:
            logger.warning(f"Label feature '{self.label_key_feature}' not found. Auto-generating using diff_pct fallback.")
            if 'Close' in self.df.columns:
                self.df[self.label_key_feature] = self.df['Close'].pct_change().shift(-1)
            else:
                logger.error("Cannot auto-generate label: 'Close' column missing.")
                raise KeyError([self.label_key_feature])

        # Drop rows with missing values for these key columns
        processed_df = self.df.dropna(subset=[self.input_key_feature, self.label_key_feature])

        # Extract features and labels
        features = processed_df[[self.input_key_feature]].values
        labels = processed_df[[self.label_key_feature]].values

        # Scale features and labels
        scaled_features = self.scaler_X.fit_transform(features)
        scaled_labels = self.scaler_y.fit_transform(labels)

        X_sequences = []
        y_targets = []

        # Create sequences
        for i in range(len(scaled_features) - self.history_size):
            X_sequences.append(scaled_features[i:i + self.history_size])
            y_targets.append(scaled_labels[i + self.history_size])

        X = np.array(X_sequences)
        y = np.array(y_targets)

        # Reshape X for LSTM if needed
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)

        logger.info(f"Created X shape: {X.shape}, y shape: {y.shape}")
        return X, y


    def process_ml_data(self):
        """
        Main method to process data for ML. It replaces the direct call to create_datasets
        and also returns input_shape and num_classes for model building.
        """
        logger.info("Starting ML data processing...")
        X, y = self.create_datasets() # Use the existing create_datasets logic

        if X is None or y is None or X.size == 0 or y.size == 0:
            logger.error("Generated X or y dataset is empty during ML data processing.")
            return None, None, None, None

        input_shape = (X.shape[1], X.shape[2]) if X.ndim == 3 else (X.shape[1], 1)
        # For regression, num_classes is typically 1 (the output dimension)
        # For classification, it would be the number of unique labels.
        # Assuming regression for now based on 'Dense(1)' in build_and_compile_model.
        num_classes = y.shape[1] if y.ndim > 1 else 1

        logger.info(f"ML data processing complete. Input shape: {input_shape}, Num classes: {num_classes}")
        return X, y, input_shape, num_classes

    def split_dataset(self, X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        """
        Splits data into training, validation, and test sets.
        """
        logger.info("Splitting datasets...")
        # Ensure splits sum up to 1 (or handle slight deviations for floating point)
        if not np.isclose(train_size + val_size + test_size, 1.0):
            logger.warning(f"Train/Val/Test sizes ({train_size}, {val_size}, {test_size}) do not sum to 1.0. Adjusting.")
            total_sum = train_size + val_size + test_size
            train_size /= total_sum
            val_size /= total_sum
            test_size /= total_sum

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=random_state)
        # Recalculate test_size for the second split relative to X_temp
        new_test_size = test_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=new_test_size, random_state=random_state)
        
        logger.info(f"Datasets split. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_tensorflow_datasets(self, X, y, batch_size=32, shuffle_buffer=1000):
        """
        Converts raw data (X, y numpy arrays) into tf.data.Dataset objects for training,
        validation, and testing, ensuring targets are included within the dataset elements.
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        logger.info("Preparing TensorFlow datasets...")
        
        # Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # Create tf.data.Dataset objects where each element is an (X, y) tuple
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Apply shuffling, batching, and prefetching for optimal performance
        train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info("TensorFlow datasets prepared successfully.")
        return train_dataset, val_dataset, test_dataset

    def evaluate_model(self, model, X_test, y_test, model_type="regression"): # Added X_test, y_test params
        """
        Evaluates the trained model on the test dataset.
        """
        logger.info(f"Evaluating model ({model_type})...")
        try:
            # Predict on the test data
            y_pred = model.predict(X_test)

            # Inverse transform predictions and true labels if scalers were used
            y_test_rescaled = self.scaler_y.inverse_transform(y_test)
            y_pred_rescaled = self.scaler_y.inverse_transform(y_pred)

            metrics = {}
            if model_type == "regression":
                mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
                mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
                r2 = r2_score(y_test_rescaled, y_pred_rescaled)
                
                metrics = {
                    "mse": mse,
                    "mae": mae,
                    "r2_score": r2
                }
                logger.info(f"Regression Metrics: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
            
            elif model_type == "classification":
                # For classification, assuming y_pred are probabilities or logits
                # Convert predictions to class labels (e.g., for binary classification with threshold 0.5)
                predictions = (y_pred_rescaled > 0.5).astype(int) # Adjust threshold as needed
                
                # Ensure y_test_rescaled is also in integer format if it's class labels
                y_true_labels = y_test_rescaled.astype(int)

                # Flatten arrays if they are multi-dimensional
                y_true_flat = y_true_labels.flatten()
                predictions_flat = predictions.flatten()

                # Choose average type: 'binary', 'micro', 'macro', 'weighted'
                average_type = 'binary' if len(np.unique(y_true_flat)) == 2 else 'weighted'
                
                metrics = {
                    "accuracy": accuracy_score(y_true_flat, predictions_flat),
                    "precision": precision_score(y_true_flat, predictions_flat, average=average_type, zero_division=0),
                    "recall": recall_score(y_true_flat, predictions_flat, average=average_type, zero_division=0),
                    "f1_score": f1_score(y_true_flat, predictions_flat, average=average_type, zero_division=0)
                }
                logger.info(f"Classification Metrics: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")

            return metrics
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}", exc_info=True)
            return {}

    def create_tf_datasets(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32, shuffle_buffer=1000):
        """
        Converts raw data into tf.data.Dataset objects for training, validation, and testing.
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # This method is redundant if prepare_tensorflow_datasets is used.
        # It's kept for backward compatibility if other parts of your code call it.
        logger.info("Creating TensorFlow datasets for training, validation, and testing...")
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        train_dataset = train_dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        logger.info("TensorFlow datasets created successfully.")
        return train_dataset, val_dataset, test_dataset