#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------
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

#=====================================================
# Class CMqlmlsetup
#=====================================================
class CMqlmlsetup:
    def __init__(self):
        pass

    def dl_split_data_sets(self, X, y, train_split=0.8, shuffle=False):
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        return X_train, X_test, y_train, y_test

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
        
        self.label_columns = self.label_columns
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

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)
