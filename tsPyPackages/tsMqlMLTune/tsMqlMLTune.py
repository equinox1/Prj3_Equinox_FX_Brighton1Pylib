import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import keras_tuner as kt
from keras_tuner import HyperParameters
import pathlib
from pathlib import Path, PurePosixPath
import posixpath


class CMdtuner:
    def __init__(self, **kwargs):
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')

        if self.traindataset is not None:
            self.traindataset = self.traindataset.map(self.cast_to_float32)
        if self.valdataset is not None:
            self.valdataset = self.valdataset.map(self.cast_to_float32)
        if self.testdataset is not None:
            self.testdataset = self.testdataset.map(self.cast_to_float32)

        # Model configuration
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)
        self.multi_inputs = kwargs.get('multi_inputs', False)

        # Data shapes
        self.data_input_shape = kwargs.get('data_input_shape')
        self.batch_size = kwargs.get('batch_size', 32)
        self.dropout = kwargs.get('dropout', 0.3)

        # Shape variables
        self.main_input_shape = None
        self.input_shapes = {}

        # Training configurations
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.factor = kwargs.get('factor', 3)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', 'best_model.keras')
        self.basepath = kwargs.get('basepath', 'tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuning')
        self.num_trials = kwargs.get('num_trials', 3)

        # Activation functions
        self.activation1 = kwargs.get('activation1', 'relu')
        self.activation2 = kwargs.get('activation2', 'linear')

        # Output dimensions
        self.output_dim = kwargs.get('output_dim', 1)

        # Ensure the base path exists
        os.makedirs(self.basepath, exist_ok=True)

        # TensorFlow debugging (optional)
        self.enable_debugging(kwargs)

        # Prepare input shapes
        self.prepare_shapes()

        # Validate configurations
        self.validate_config()

        # Initialize the tuner
        self.initialize_tuner()

    @staticmethod
    def cast_to_float32(x, y):
        return tf.cast(x, tf.float32), y

    def enable_debugging(self, kwargs):
        if kwargs.get('tf1', False):
            tf.debugging.set_log_device_placement(True)
        if kwargs.get('tf2', False):
            tf.debugging.enable_check_numerics()

    def prepare_shapes(self):
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")

        self.main_input_shape = self.get_shape(self.data_input_shape)
        if self.cnn_model:
            self.input_shapes['cnn_input'] = self.main_input_shape
        if self.lstm_model:
            self.input_shapes['lstm_input'] = self.main_input_shape
        if self.gru_model:
            self.input_shapes['gru_input'] = self.main_input_shape
        if self.transformer_model:
            self.input_shapes['transformer_input'] = self.main_input_shape

    @staticmethod
    def get_shape(data_shape):
        if len(data_shape) not in [2, 3]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D or 3D.")
        return tuple(data_shape)

    def validate_config(self):
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")

    def initialize_tuner(self):
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=True
        )
        self.tuner.search_space_summary()

    def build_model(self, hp):
        # Define inputs
        inputs = []

        # CNN branch
        if self.cnn_model:
            cnn_input = Input(shape=self.input_shapes['cnn_input'], name='cnn_input')
            inputs.append(cnn_input)
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', 32, 128, step=32),
                kernel_size=hp.Int('cnn_kernel_size', 2, 5),
                activation=hp.Choice('cnn_activation', ['relu', 'tanh'])
            )(cnn_input)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
        else:
            x_cnn = None

        # LSTM branch
        if self.lstm_model:
            lstm_input = Input(shape=self.input_shapes['lstm_input'], name='lstm_input')
            inputs.append(lstm_input)
            x_lstm = LSTM(
                units=hp.Int('lstm_units', 32, 128, step=32),
                activation=hp.Choice('lstm_activation', ['relu', 'tanh'])
            )(lstm_input)
        else:
            x_lstm = None

        # GRU branch
        if self.gru_model:
            gru_input = Input(shape=self.input_shapes['gru_input'], name='gru_input')
            inputs.append(gru_input)
            x_gru = GRU(
                units=hp.Int('gru_units', 32, 128, step=32),
                activation=hp.Choice('gru_activation', ['relu', 'tanh'])
            )(gru_input)
        else:
            x_gru = None

        # Transformer branch
        if self.transformer_model:
            transformer_input = Input(shape=self.input_shapes['transformer_input'], name='transformer_input')
            inputs.append(transformer_input)
            x_transformer = MultiHeadAttention(
                num_heads=hp.Int('num_heads', 2, 8),
                key_dim=hp.Int('key_dim', 32, 128, step=32)
            )(transformer_input, transformer_input)
            x_transformer = LayerNormalization()(x_transformer)
            x_transformer = GlobalAveragePooling1D()(x_transformer)
        else:
            x_transformer = None

        # Combine the outputs of each branch
        branches = [branch for branch in [x_cnn, x_lstm, x_gru, x_transformer] if branch is not None]
        if len(branches) > 1:
            combined = Concatenate()(branches)
        elif len(branches) == 1:
            combined = branches[0]
        else:
            raise ValueError("No branches have been configured. At least one branch (CNN, LSTM, GRU, Transformer) must be enabled.")

        # Add dense layers on top of combined features
        x = Dense(50, activation=self.activation1)(combined)
        x = Dropout(self.dropout)(x)
        output = Dense(self.output_dim, activation=self.activation2)(x)

        # Create the model
        model = Model(inputs=inputs, outputs=output)

        # Compile the model
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        learning_rate = hp.Float('lr', 1e-4, 1e-2, sampling='LOG')
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

        model.compile(
            optimizer=opt,
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model

    def get_callbacks(self):
        return [
            EarlyStopping(monitor=self.objective, patience=3, verbose=1),
            ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=True, verbose=1),
            TensorBoard(log_dir=os.path.join(self.basepath, 'logs'))
        ]

    def run_search(self):
        try:
            self.tuner.search(
                self.traindataset,
                validation_data=self.valdataset,
                callbacks=self.get_callbacks(),
                epochs=self.max_epochs
            )
        except Exception as e:
            print(f"Error during tuning: {e}")

    def export_best_model(self, ftype='tf'):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.basepath, self.project_name, 'best_model')
            if ftype == 'h5':
                best_model.save(export_path + '.h5')  # For HDF5 format
                print(f"Model saved to {export_path}.h5")
            else:
                best_model.save(export_path + '.keras')  # For TensorFlow SavedModel format
                print(f"Model saved to {export_path}.keras")

        except IndexError:
            print("No models found to export.")
        except Exception as e:
            print(f"Error saving the model: {e}")


    def run_prediction(self, test_data, batch_size=None):
        """
        Run predictions using the best model obtained from hyperparameter tuning.

        Parameters:
        - test_data: Dataset or numpy array to predict on.
        - batch_size: Batch size for prediction. Defaults to self.batch_size.

        Returns:
        - Predictions made by the model.
        """
        try:
            # Load the best model
            best_model = self.tuner.get_best_models(num_models=1)[0]

            # Ensure test_data is casted to tf.float32
            if isinstance(test_data, tf.data.Dataset):
                test_data = test_data.map(self.cast_to_float32)

            # Run predictions
            predictions = best_model.predict(test_data, batch_size=batch_size or self.batch_size)
            return predictions
        except IndexError:
            print("No models found. Ensure tuning has been run successfully.")
        except Exception as e:
            print(f"Error during prediction: {e}")
   
