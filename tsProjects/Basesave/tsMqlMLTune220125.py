import tensorflow as tf
from keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import pad_sequences

import keras_tuner as kt
import os
import numpy as np


class CMdtuner:
    def __init__(self, hypermodel_params=None, traindataset=None, valdataset=None, testdataset=None, **kwargs):
        """
        Initialize the CMdtuner class for dynamic model creation and hyperparameter tuning.
        """
        self.hypermodel_params = hypermodel_params or {}
        self.traindataset = traindataset
        self.valdataset = valdataset
        self.testdataset = testdataset

        self._set_defaults(kwargs)
        self._prepare_directories()
        self._prepare_shapes()
        self._initialize_inputs()
        self._initialize_tuner()
        self._start_search()

    def _set_defaults(self, kwargs):
        """Set default values for configuration parameters."""
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)
        self.multi_inputs = kwargs.get('multi_inputs', False)

        self.data_input_shape = kwargs.get('data_input_shape')
        print(f"Data Input Shape: {self.data_input_shape}")
        if self.data_input_shape is None:
            raise ValueError("data_input_shape must be provided.")
        self.batch_size = kwargs.get('batch_size', 32)

        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)

        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.factor = kwargs.get('factor', 3)

        self.activation1 = kwargs.get('activation1', 'relu')
        self.activation2 = kwargs.get('activation2', 'linear')
        self.dropout = kwargs.get('dropout', 0.3)

        self.basepath = kwargs.get('basepath', './tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuner_project')
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', os.path.join(self.basepath, 'best_model.keras'))
        self.overwrite = kwargs.get('overwrite', False)

        self.chk_patience = kwargs.get('chk_patience', 3)
        self.chk_monitor = kwargs.get('chk_monitor', 'val_loss')

    def _prepare_directories(self):
        """Ensure base directories exist."""
        os.makedirs(self.basepath, exist_ok=True)

    def _prepare_shapes(self):
        """Prepare input shapes for each model branch based on provided data."""
        if self.data_input_shape:
            self.timesteps = self.data_input_shape[1] if len(self.data_input_shape) > 1 else None
            self.features = self.data_input_shape[2] if len(self.data_input_shape) > 2 else None
            self.channels = self.data_input_shape[3] if len(self.data_input_shape) > 3 else None

    def _initialize_inputs(self):
        """Initialize input layers for each branch based on configuration."""
        if self.cnn_model:
            self.cnn_inputs = Input(shape=(self.timesteps, self.features), name="cnn_input")
            print(f"CNN Inputs Shape: {self.cnn_inputs.shape}")
        if self.lstm_model:
            self.lstm_inputs = Input(shape=(self.timesteps, self.features), name="lstm_input")
            print(f"LSTM Inputs Shape: {self.lstm_inputs.shape}")
        if self.gru_model:
            self.gru_inputs = Input(shape=(self.timesteps, self.features), name="gru_input")
            print(f"GRU Inputs Shape: {self.gru_inputs.shape}")
        if self.transformer_model:
            self.transformer_inputs = Input(shape=(self.timesteps, self.features), name="transformer_input")
            print(f"Transformer Inputs Shape: {self.transformer_inputs.shape}")

    def _initialize_tuner(self):
        """Initialize the KerasTuner with Hyperband."""
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=self.overwrite
        )

    def preprocess_dataset(self, dataset, max_timesteps=None):
        """Preprocess the dataset for uniform input shapes."""
        inputs, labels = [], []
        for data in dataset:
            feature, label = data
            inputs.append(feature)
            labels.append(label)

        # Determine max_timesteps if not provided
        if max_timesteps is None:
            max_timesteps = max([np.shape(x)[0] for x in inputs])

        # Pad or truncate the inputs
        inputs = pad_sequences(inputs, maxlen=max_timesteps, dtype='float32', padding='post', truncating='post')

        # Process labels
        if isinstance(labels[0], (list, np.ndarray)):
            max_label_len = max([len(label) for label in labels])
            labels = pad_sequences(labels, maxlen=max_label_len, dtype='float32', padding='post', truncating='post')
        else:
            labels = np.array(labels, dtype='float32')

        return inputs, labels

    def _start_search(self):
        """Start hyperparameter tuning."""
        if not self.traindataset or not self.valdataset:
            raise ValueError("Training and validation datasets must be provided.")

        train_inputs, train_labels = self.preprocess_dataset(self.traindataset)
        val_inputs, val_labels = self.preprocess_dataset(self.valdataset)

        print(f"Train Inputs Shape: {np.shape(train_inputs)}")
        print(f"Train Labels Shape: {np.shape(train_labels)}")
        print(f"Validation Inputs Shape: {np.shape(val_inputs)}")
        print(f"Validation Labels Shape: {np.shape(val_labels)}")

        self.tuner.search(
            x=train_inputs,
            y=train_labels,
            validation_data=(val_inputs, val_labels),
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=self._get_callbacks(),
        )

    def build_model(self, hp):
        """Build the model based on hyperparameters."""
        branches = []
        self.cnn_inputs = Input(shape=self.data_input_shape)
        if self.cnn_model:
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', 32, 128, step=32),
                kernel_size=hp.Int('cnn_kernel_size', 2, 5),
                activation='relu'
            )(self.cnn_inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
            branches.append(x_cnn)

        if self.lstm_model:
            x_lstm = LSTM(hp.Int('lstm_units', 32, 128, step=32))(self.lstm_inputs)
            branches.append(x_lstm)

        if self.gru_model:
            x_gru = GRU(hp.Int('gru_units', 32, 128, step=32))(self.gru_inputs)
            branches.append(x_gru)

        if self.transformer_model:
            x_trans = MultiHeadAttention(
                num_heads=hp.Int('num_heads', 2, 4), key_dim=hp.Int('key_dim', 32, 128, step=32)
            )(self.transformer_inputs, self.transformer_inputs)
            x_trans = LayerNormalization()(x_trans)
            x_trans = GlobalAveragePooling1D()(x_trans)
            branches.append(x_trans)

        if len(branches) > 1:
            combined = Concatenate()(branches)
        else:
            combined = branches[0]

        x = Dense(50, activation=self.activation1)(combined)
        x = Dropout(self.dropout)(x)
        output = Dense(1, activation=self.activation2)(x)

        inputs = [
            inp for inp in [
                self.cnn_inputs, self.lstm_inputs, self.gru_inputs, self.transformer_inputs
            ] if inp is not None
        ]

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model

    def _get_callbacks(self):
        """Return callbacks for training."""
        return [
            EarlyStopping(monitor=self.chk_monitor, patience=self.chk_patience, verbose=1),
            ModelCheckpoint(filepath=self.checkpoint_filepath, save_best_only=True, verbose=1),
            TensorBoard(log_dir=os.path.join(self.basepath, 'logs'))
        ]
