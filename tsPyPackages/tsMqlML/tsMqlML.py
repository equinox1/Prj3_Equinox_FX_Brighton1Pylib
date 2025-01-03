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
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D

#----------------------------------------------------
# Class CMqlmlsetup
#+--------------------------------------------------
class CMqlmlsetup:
    def __init__(self):
        pass

    #--------------------------------------------------------------------
    # create method  "dl_split_data_sets".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def dl_split_data_sets(self, X,y, train_split= 0.8,test_split= 0.2, shuffle=False):
        # Splitting data into training and testing sets
        train_size = int(len(X) * train_split)
        test_size = len(X) - train_size
        X_train, X_test = X[0:train_size], X[train_size:len(X)] 
        y_train, y_test = y[0:train_size], y[train_size:len(y)]
        return X_train, X_test, y_train, y_test
    
    #--------------------------------------------------------------------
    # create method  "dl_train_model_scaled".
    # class:cmqlmlsetup
    # usage: mql data
    # /param  var
    #--------------------------------------------------------------------
    def dl_train_model_scaled(self, lpdf):
        scaler = StandardScaler()
        return scaler.fit_transform(lpdf)

    #--------------------------------------------------------------------
    # create method  "dl_test_model_scaled".
    # class:cmqlmlsetup
    # usage: mql data
    # /param  var
    #--------------------------------------------------------------------
    def dl_test_model_scaled(self, lpdf):
        scaler = StandardScaler()
        return scaler.fit_transform(lpdf)

    #--------------------------------------------------------------------
    # create method  "dl_model_performance"
    # class:cmqlmlsetup
    # usage: mql data
    # /pdl_build_neuro_network
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


#----------------------------------------------------
# Class CMqlWindowGenerator
#+--------------------------------------------------
class CMqlWindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None, **kwargs):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in 
                                         enumerate(label_columns)}
        self.column_indices = {name: i for i, name in 
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
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

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        return inputs, labels

    def plot(self, model=None, plot_col='close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

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
        return plt

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def window_slicer(self, df, window_size, shift_size):

        # Stack three slices, the length of the total window.
        slice1 = df[:window_size]
        slice2 = df[shift_size:shift_size + window_size]
        slice3 = df[2 * shift_size:2 * shift_size + window_size]

        # Ensure all slices are of the same shape
        min_length = min(len(slice1), len(slice2), len(slice3))

        varwin = tf.stack([
            slice1[:min_length],
            slice2[:min_length],
            slice3[:min_length]
        ])

        varwin_inputs, varwin_labels = self.split_window(varwin)
        return varwin ,varwin_inputs, varwin_labels

    import numpy as np

    def Sequential_Input(df, input_sequence):
        """
        Prepare sequential input data for an LSTM model.
        
        Parameters:
        df (pd.DataFrame or pd.Series): The data to prepare, typically a time series.
        input_sequence (int): The number of time steps to include in each input sequence.
        
        Returns:
        tuple: Two numpy arrays, X (input sequences) and y (corresponding labels).
        """
        df_np = df.to_numpy()  # Convert DataFrame or Series to numpy array
        X = []
        y = []
        for i in range(len(df_np) - input_sequence):
            row = df_np[i:i + input_sequence]  # Input sequence
            X.append(row)
            label = df_np[i + input_sequence]  # Corresponding label
            y.append(label)
        return np.array(X), np.array(y)

    