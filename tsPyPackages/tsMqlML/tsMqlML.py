#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"

from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, mean_squared_error, mean_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense, 
                                      LSTM, GRU, Dropout, concatenate, LayerNormalization, 
                                      MultiHeadAttention, GlobalAveragePooling1D)
import os
import numpy as np
import posixpath
from datetime import date
#=====================================================
# Class CMqlml
#=====================================================
class CMqlmlsetup:
    def __init__(self, **kwargs):
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.label_width = kwargs.get('label_width', 1)
        self.train_df = kwargs.get('train_df', None)
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df', None)
        self.label_columns = kwargs.get('label_columns', None)
        self.batch_size = kwargs.get('batch_size', 32)
        self.column_indices = {}
        self.label_columns_indices = {}
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def dl_split_ml_sets(self, X, y, train_split=0.8, shuffle=False):
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

    def create_Xy(self, data, target_column, lookback_minutes=60):
        # Create the input features (X) and target values (y)
            X, y = [], []
            for i in range(len(data) - lookback_minutes * 2):
                X.append(data[target_column].iloc[i:i + lookback_minutes].values)
                y.append(data[target_column].iloc[i + lookback_minutes])

            X = np.array(X)
            y = np.array(y)
            return X, y

    
    def create_Xy_time_windows1(self,data, window_size = 24, target_steps=1):
        """
        Creates input (X) and target (y) datasets from a time series.
        
        Parameters:
        - data: The time-series data (1D array or list).
        - window_size: The number of time steps for each input window.
        - target_steps: The number of steps ahead for the target.

        Returns:
        - X: 2D array of shape (num_samples, window_size).
        - y: 1D or 2D array of shape (num_samples, target_steps).
        """
        print("create_Xy_time_windows:window_size", window_size)
        print("create_Xy_time_windows:target_steps", target_steps)
      
        X, y = [], []
        for i in range(len(data) - window_size - target_steps + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + target_steps])
        return np.array(X), np.array(y)


    # STEP: Create input (X) and target (Y) tensors
    def create_Xy_time_windows2(self,data, time_steps, future_steps):
        """
        Generate input-output pairs from the time-series data.
        :param data: DataFrame with scaled features and target.
        :param time_steps: Number of past time steps to use as input.
        :param future_steps: Number of future steps to predict.
        :return: Numpy arrays for X (input) and y (output).
        """
        X, y = [], []
        for i in range(len(data) - time_steps - future_steps + 1):
            X.append(data.iloc[i:i + time_steps, :-1].values)  # All columns except target
            y.append(data.iloc[i + time_steps:i + time_steps + future_steps, -1].values)  # Target column
        return np.array(X), np.array(y)


    def create_Xy_time_windows3(self,df, past_window,future_window, target_column='Close_Scaled',feature_column='Close'):
        # Create the X and Y datasets
        print(df.columns)  # Check available columns
    
        print("create_Xy_time_windows3:past_window", past_window)
        print("create_Xy_time_windows3:future_window", future_window)
        print("create_Xy_time_windows3:target_column", target_column)
        print("create_Xy_time_windows3:feature_column", feature_column)

        X, Y = [], []
        for i in range(len(df) - past_window - future_window):
            past = df[i:i + past_window][target_column].values
            future = df[i + past_window:i + past_window + future_window][feature_column].values[-1]  # Use last value in the future window as the label
            X.append(past)
            Y.append(future)
        return np.array(X), np.array(Y)

    def align_to_batch_size(self,X, y, batch_size):
        """
        Ensure data size is divisible by batch size.
        :param X: Input array.
        :param y: Target array.
        :param batch_size: Desired batch size.
        :return: Resized X and y arrays.
        """
        usable_size = len(X) - (len(X) % batch_size)
        return X[:usable_size], y[:usable_size]  


    def multi_input_preprocess(self, data):
        if not isinstance(data, dict):
            raise ValueError("Expected data to be a dictionary")
        
        x = {
            "cnn_input": data.get("cnn_features", None),
            "lstm_input": data.get("lstm_features", None),
            "gru_input": data.get("gru_features", None),
            "transformer_input": data.get("transformer_features", None),
        }
        y = data.get("target", None)
        return x, y

    def convert_to_dict(self, element):
        # Example conversion: Adapt based on the structure of your dataset
        return {
            "cnn_features": element[0],  # Assuming this is the first feature
            "lstm_features": element[1],  # Assuming this is the second feature
            "gru_features": element[2],  # Assuming this is the third feature
            "transformer_features": element[3],  # Assuming this is the fourth feature
            "target": element[4],  # Assuming this is the label/target
        }


    def create_tf_dataset(self,features, labels=None, batch_size=32, shuffle=False):
        
        # Convert to tensors

        if isinstance(features, np.ndarray):
            # Handle standard single-input case
            dataset = tf.data.Dataset.from_tensor_slices((features, labels) if labels is not None else features)

        # Shuffle and batch the dataset
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(features) if isinstance(features, tf.Tensor) else features.shape[0])
        dataset = dataset.batch(batch_size)
        return dataset

    
    def scale_data(self, data, fit=False, scaler=None):
        if scaler is None:
            scaler = StandardScaler()
        return scaler.fit_transform(data) if fit else scaler.transform(data), scaler

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

#=====================================================
# Class CMqlWindowGenerator
#=====================================================
class CMqlWindowGenerator:
    def __init__(self, **kwargs):
        self.input_width = kwargs.get('input_width', 24)
        self.shift = kwargs.get('shift', 24)
        self.label_width = kwargs.get('label_width', 1)
        self.train_df = kwargs.get('train_df', None)
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df', None)
        self.label_columns = kwargs.get('label_columns', None)
        self.batch_size = kwargs.get('batch_size', 32)
        self.column_indices = {}
        self.label_columns_indices = {}
        self.total_window_size = self.input_width + self.shift
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        print("winobj:input_width", self.input_width)
        print("winobj:label_width", self.label_width)
        print("winobj:total_window_size", self.total_window_size)

        # Store the raw data.
        self.train_df = self.train_df
        self.val_df = self.val_df
        self.test_df = self.test_df

        # Work out the label column indices.
        
        
        if self.label_columns is not None:
            self.label_columns_indices = {name: i for i, name in 
                                         enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in 
                               enumerate(self.train_df.columns)}

        # Define slices and indices
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def plot(self, inputs, labels, model=None, plot_col='close', max_subplots=3):
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            label_col_index = self.column_indices.get(plot_col, None)
            if label_col_index is not None:
                plt.scatter(self.label_indices, labels[n, :, label_col_index],
                            edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        plt.show()

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,)

        ds = ds.map(self.split_window)

        return ds


    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def slice_window(self, df, window_size, shift_size):
        if isinstance(df, pd.DataFrame):
            df = df.values
        
        # Convert Timestamp objects to strings
        df = np.array([[str(cell) if isinstance(cell, pd.Timestamp) else cell for cell in row] for row in df])
        
        slices = [df[i:i + window_size] for i in range(0, len(df) - window_size + 1, shift_size)]
        min_length = min(len(s) for s in slices)
        slices = [s[:min_length] for s in slices]
        slices = tf.stack(slices)
        return slices


    def create_tf_datasets(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
        train_dataset = m1.create_tf_dataset(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_dataset = m1.create_tf_dataset(X_val, y_val, batch_size=batch_size, shuffle=False)
        test_dataset = m1.create_tf_dataset(X_test, y_test, batch_size=batch_size, shuffle=False)
        return train_dataset, val_dataset, test_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

class CMqlEnvML:
   def __init__(self, **kwargs):
         self.kwargs = kwargs
         #Windowing parameters
         self.mp_ml_cfg_period = kwargs.get('mp_ml_cfg_period', 24)
         self.mp_ml_cfg_period1 = kwargs.get('mp_ml_cfg_period1', 24) #Hours
         self.mp_ml_cfg_period2 = kwargs.get('mp_ml_cfg_period2', 6)  #Hours
         self.mp_ml_cfg_period3 = kwargs.get('mp_ml_cfg_period3', 1)  #Hours
         self.mp_ml_tf_ma_windowin = kwargs.get('mp_ml_tf_ma_windowing', 24)
         self.mp_ml_tf_shiftin = kwargs.get('mp_ml_tf_shiftin', 1) 
         
         #Average column names
         self.mp_ml_hl_avg_col = kwargs.get('mp_ml_hl_avg_col', 'HLAvg')
         self.mp_ml_ma_col = kwargs.get('mp_ml_ma_col', 'SMA')
         self.mp_ml_returns_col = kwargs.get('mp_ml_returns_col', 'LogReturns')
         self.mp_ml_returns_col_scaled = kwargs.get('mp_ml_returns_col_scaled', 'LogReturns_scaled')
         self.mp_ml_create_label = kwargs.get('mp_ml_create_label', False)
         self.mp_ml_create_label_scaled = kwargs.get('mp_ml_create_label_scaled', False)

          # feature Processing
         self.mp_ml_input_keyfeat = kwargs.get('mp_ml_input_keyfeat', {'Close'})
         self.mp_ml_output_label = kwargs.get('mp_ml_output_label', self.mp_ml_input_keyfeat)
         self.mp_ml_input_keyfeat_scaled = {feat + '_Scaled' for feat in self.mp_ml_input_keyfeat}  # the feature to predict
         self.mp_ml_output_label_scaled = {targ + '_Scaled' for targ in self.mp_ml_output_label}  # the label shifted to predict
         self.mp_ml_output_label_count=len(self.mp_ml_output_label)

         self.window_params = {
            "mp_ml_cfg_period": self.mp_ml_cfg_period,
            "lookahead_periods": self.mp_ml_cfg_period,
            "mp_ml_cfg_period1": self.mp_ml_cfg_period1,
            "mp_ml_cfg_period2": self.mp_ml_cfg_period2,
            "mp_ml_cfg_period3": self.mp_ml_cfg_period3,
            "mp_ml_tf_ma_window": self.mp_ml_tf_ma_windowin,
            "mp_ml_tf_shiftin": self.mp_ml_tf_shiftin,
         }

         self.average_params = {
            "mp_ml_hl_avg_col": self.mp_ml_hl_avg_col,
            "mp_ml_ma_col": self.mp_ml_ma_col,
            "mp_ml_returns_col": self.mp_ml_returns_col,
            "mp_ml_returns_col_scaled": self.mp_ml_returns_col_scaled,
            "mp_ml_create_label": self.mp_ml_create_label,
            "mp_ml_create_label_scaled": self.mp_ml_create_label_scaled,
         }  
 

         self.feature_ml_params = {
            "mp_ml_input_keyfeat": self.mp_ml_input_keyfeat,
            "mp_ml_output_label": self.mp_ml_output_label,
            "mp_ml_input_keyfeat_scaled": self.mp_ml_input_keyfeat_scaled,
            "mp_ml_output_label_scaled": self.mp_ml_output_label_scaled,
            "mp_ml_output_label_count": self.mp_ml_output_label_count,
        }


   def get_params(self):
        """Returns a dictionary of all set parameters."""
        return {
            **self.window_params,
            **self.average_params,
            **self.feature_ml_params,
         }

