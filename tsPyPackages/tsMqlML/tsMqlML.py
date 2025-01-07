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

    
    #--------------------------------------------------------------------
    # create method  "d__repr__".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    #--------------------------------------------------------------------
    # create method  "plot".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def plot(self, model=None, plot_col='close', max_subplots=3):
        #inputs, labels = self.example
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

    #--------------------------------------------------------------------
    # create method  Sequential_Input".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
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

    #--------------------------------------------------------------------
    # create method  window_slicer".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def window_slicer(self, df, window_size, shift_size):
        print("length of df", len(df), "window_size", window_size, "shift_size", shift_size)
        # Ensure the input is compatible
        if isinstance(df, np.ndarray):
            data = df
        elif hasattr(df, "values"):  # For Pandas DataFrame or Series
            data = df.values
        else:
            raise TypeError("Input must be a Pandas DataFrame, Series, or NumPy array.")
        # Ensure the data length is sufficient
        if 2 * shift_size + window_size > len(data):
            raise ValueError("Data is too small for the given window and shift sizes.")
        # Convert window_size to an integer
        window_size = int(window_size)
        # Create slices using NumPy slicing
        slices = []
        for start_idx in range(0, len(data) - window_size + 1, shift_size):
            end_idx = start_idx + window_size
            slices.append(data[start_idx:end_idx])
        # Ensure all slices are of the same shape
        min_length = min(len(s) for s in slices)
        varwin = tf.stack([s[:min_length] for s in slices])
        return varwin

    #--------------------------------------------------------------------
    # create method  "split_window".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def split_window0(self, data, input_size, output_size, stride):
        """Splits data into windows for time series forecasting.

        Args:
            data: A NumPy array or TensorFlow Dataset containing the time series data.
            input_size: The number of time steps in the input window.
            output_size: The number of time steps in the output window.
            stride: The number of time steps to shift the window.

        Returns:
            A TensorFlow Dataset of input-output pairs.
        """

        # Ensure input_size, output_size, and stride are integers
        input_size = int(input_size)
        output_size = int(output_size)
        stride = int(stride)

        ds = tf.data.Dataset.from_tensor_slices(data)

        def _split_window(window):
            """Internal function to create a single input-output pair."""
            input_data = window[:input_size]
            output_data = window[input_size:input_size + output_size]
            return input_data, output_data

        ds = ds.window(input_size + output_size, shift=stride, drop_remainder=True)
        ds = ds.flat_map(lambda window: window.batch(input_size + output_size))
        ds = ds.map(_split_window)

        return ds


     #--------------------------------------------------------------------
    # create method  "make_dataset".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    def make_dataset0(self, data, batch_size=32, total_window_size=24, shuffle=True, targets=1, output_size=1, stride=1, **kwargs):
        data = np.array(data, dtype=np.float32)

        # Ensure targets is an integer and within the valid range
        if not isinstance(targets, int) or targets <= 0 or targets >= len(data):
            raise ValueError("Targets must be a positive integer less than the length of the data.")

        # Create the dataset
        ds = self.split_window(data, input_size=total_window_size - output_size, output_size=output_size, stride=stride)
        
        # Shuffle and batch the dataset
        if shuffle:
            ds = ds.shuffle(buffer_size=len(data))
        ds = ds.batch(batch_size)

        # Debugging outputs
        print("Data type:", type(data))
        print("Data shape:", data.shape)
        print("Batch size:", batch_size)
        print("Total window size:", total_window_size)
        print("Stride:", stride)
        print("Output size:", output_size)

        return ds


    #--------------------------------------------------------------------
    # create method  tf Datasets".
    # class:cmqlmlsetup
    # usage: mql data
    # /pdlsplit data
    #--------------------------------------------------------------------
    @property
    def train(self):
            return self.make_dataset(self.train_df,batch_size=32,total_window_size=24,shuffle=True,targets=None)

    @property
    def val(self):
            return self.make_dataset(self.val_df, batch_size=32, total_window_size=24, shuffle=True, targets=None)

    @property
    def test(self):
            return self.make_dataset(self.test_df, batch_size=32, total_window_size=24, shuffle=True, targets=None)


    


    def split_window(self, data, input_size, output_size, stride):
        """Splits data into windows for time series forecasting.

        Args:
            data: A NumPy array or TensorFlow Dataset containing the time series data.
            input_size: The number of time steps in the input window.
            output_size: The number of time steps in the output window.
            stride: The number of time steps to shift the window.

        Returns:
            A tuple containing two TensorFlow Datasets: inputs and labels.
        """

        # Ensure input_size, output_size, and stride are integers
        input_size = int(input_size)
        output_size = int(output_size)
        stride = int(stride)

        ds = tf.data.Dataset.from_tensor_slices(data)

        def _split_window(window):
            """Internal function to create a single input-output pair."""
            input_data = window[:input_size]
            output_data = window[input_size:input_size + output_size]
            return input_data, output_data

        ds = ds.window(input_size + output_size, shift=stride, drop_remainder=True)
        ds = ds.flat_map(lambda window: window.batch(input_size + output_size))
        ds = ds.map(_split_window)

        inputs = ds.map(lambda input_data, output_data: input_data)
        labels = ds.map(lambda input_data, output_data: output_data)

        return inputs, labels

    def make_dataset(self, data, batch_size=32, total_window_size=24, shuffle=True, targets=1, output_size=1, stride=1, **kwargs):
        data = np.array(data, dtype=np.float32)

        # Ensure targets is an integer and within the valid range
        if not isinstance(targets, int) or targets <= 0 or targets >= len(data):
            raise ValueError("Targets must be a positive integer less than the length of the data.")

        # Create the dataset
        inputs, labels = self.split_window(data, input_size=total_window_size - output_size, output_size=output_size, stride=stride)

        # Zip the inputs and labels into a single dataset
        ds = tf.data.Dataset.zip((inputs, labels))

        # Shuffle and batch the dataset
        if shuffle:
            ds = ds.shuffle(buffer_size=len(data))
        ds = ds.batch(batch_size)

        # Debugging outputs
        print("Data type:", type(data))
        print("Data shape:", data.shape)
        print("Batch size:", batch_size)
        print("Total window size:", total_window_size)
        print("Stride:", stride)
        print("Output size:", output_size)
        return ds

    # Function to print elements in a dataset
    def print_dataset_elements(self, dataset, num_elements=1):
        shapes = []
        dtypes = []
        for element in dataset.take(num_elements):
            print("Dataset Element:")
            if isinstance(element, tuple):
                for idx, el in enumerate(element):
                    print(f"Element {idx}: shape={el.shape}, dtype={el.dtype}")
                    shapes.append(el.shape)
                    dtypes.append(el.dtype)
            else:
                print(f"Element: shape={element.shape}, dtype={element.dtype}")
                shapes.append(element.shape)
                dtypes.append(element.dtype)
        
        # Return a list of shapes and dtypes
        return shapes, dtypes

    def mergeXyTensor(self, X, y, batch_size=32):
        X = tf.expand_dims(X, axis=0)
        y = tf.expand_dims(y, axis=0)
    
        try:
            df = tf.data.Dataset.from_tensor_slices(
                 (X, y)
            ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            print("Slice DF merge:", df)
        except ValueError as e:
                print(f"Error: {e}")
                print(f"Merge issue X: {X.shape}")
                print(f"Merge issue y: {y.shape}")
    
        return df
    