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

    def make_dataset(self, data, batch_size=32,total_window_size=24,shuffle=True,targets=None):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=targets,
            sequence_length=total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df,batch_size=32,total_window_size=24,shuffle=True,targets=None)

    @property
    def val(self):
        return self.make_dataset(self.val_df, batch_size=32, total_window_size=24, shuffle=True, targets=None)

    @property
    def test(self):
        return self.make_dataset(self.test_df, batch_size=32, total_window_size=24, shuffle=True, targets=None)

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



    def window_slicer(self, df, window_size, shift_size):
        # Ensure the DataFrame is large enough for the required slices
        print("length of df",len(df), "window_size", window_size, "shift_size", shift_size, "2 * shift_size + window_size", 2 * shift_size + window_size)
        if len(df) < 2 * shift_size + window_size:
            raise ValueError("DataFrame is too small for the given window and shift sizes")

        # Stack three slices, the length of the total window.
        slice1 = df.iloc[:window_size]
        slice2 = df.iloc[shift_size:shift_size + window_size]
        slice3 = df.iloc[2 * shift_size:2 * shift_size + window_size]

        # Ensure all slices are of the same shape
        min_length = min(len(slice1), len(slice2), len(slice3))

        varwin = tf.stack([
            slice1[:min_length].values,
            slice2[:min_length].values,
            slice3[:min_length].values
        ])

        return varwin

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

    import tensorflow as tf

    def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
        """
        Creates a windowed dataset for time series data.

        Args:
        - series (array-like): The time series data to process.
        - window_size (int): The size of each window.
        - batch_size (int): The batch size for training.
        - shuffle_buffer (int): The buffer size for shuffling the data.

        Returns:
        - tf.data.Dataset: The processed dataset ready for training.
        """
        # Step 1: Convert the series into a TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices(series)

        # Step 2: Create windows of the specified size
        dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

        # Step 3: Flatten each window into a single dataset
        dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

        # Step 4: Split each window into features (x) and labels (y)
        dataset = dataset.map(lambda window: (window[:-1], window[-1]))

        # Step 5: Shuffle the dataset with the specified buffer size
        dataset = dataset.shuffle(shuffle_buffer)

        # Step 6: Batch the dataset
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return dataset
