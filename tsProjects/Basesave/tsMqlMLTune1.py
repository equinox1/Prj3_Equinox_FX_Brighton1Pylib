import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import os

class CMdtuner:
    def __init__(self, **kwargs):
        # Load datasets
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')

        # Model configuration
        self.cnn_model = kwargs.get('cnn_model', False)
        self.lstm_model = kwargs.get('lstm_model', False)
        self.gru_model = kwargs.get('gru_model', False)
        self.transformer_model = kwargs.get('transformer_model', False)

        # Hyperparameter tuning configuration
        self.objective = kwargs.get('objective', 'val_loss')
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.factor = kwargs.get('factor', 3)
        self.seed = kwargs.get('seed', 42)
        self.step = kwargs.get('step', 5)
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.batch_size = kwargs.get('batch_size', 32)
        self.activation1 = kwargs.get('activation1', 'relu')
        self.activation2 = kwargs.get('activation2', 'linear')

        # Checkpoint configuration
        self.chk_patience = kwargs.get('chk_patience', 3)
        self.basepath = kwargs.get('basepath', './tuning_results')
        self.project_name = kwargs.get('project_name', 'project')
        os.makedirs(self.basepath, exist_ok=True)

        # Input shape configuration
        self.input_shape = kwargs.get('input_shape')  # Explicitly set the input shape
        if not self.input_shape:
            raise ValueError("Input shape must be provided via 'input_shape' in kwargs.")

        # Tuner configuration
        self.tuner = kt.Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            factor=self.factor,
            directory=self.basepath,
            project_name=self.project_name
        )

    def build_model(self, hp):
        # Ensure at least one model branch is selected
        if not any([self.cnn_model, self.lstm_model, self.gru_model, self.transformer_model]):
            raise ValueError("At least one model branch must be enabled (cnn_model, lstm_model, gru_model, or transformer_model).")

        inputs = Input(shape=self.input_shape)
        branches = []

        # CNN Branch
        if self.cnn_model:
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32),
                kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1),
                activation='relu'
            )(inputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
            branches.append(x_cnn)

        # LSTM Branch
        if self.lstm_model:
            x_lstm = LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
            x_lstm = LSTM(hp.Int('lstm_units', min_value=32, max_value=128, step=32))(x_lstm)
            branches.append(x_lstm)

        # GRU Branch
        if self.gru_model:
            x_gru = GRU(hp.Int('gru_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
            x_gru = GRU(hp.Int('gru_units', min_value=32, max_value=128, step=32))(x_gru)
            branches.append(x_gru)

        # Transformer Branch
        if self.transformer_model:
            key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32)
            num_heads = hp.Int('num_heads', min_value=2, max_value=4, step=1)

            x_trans = self.add_positional_encoding(inputs, self.input_shape[0], key_dim)
            x_trans = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_trans, x_trans)
            x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
            x_trans = GlobalAveragePooling1D()(x_trans)
            branches.append(x_trans)

        # Combine branches
        if len(branches) > 1:
            x = Concatenate()(branches)
        else:
            x = branches[0]

        # Fully connected layers
        x = Dense(50, activation=self.activation1)(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation=self.activation2)(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
            loss=MeanSquaredError(),
            metrics=[MeanAbsoluteError()]
        )
        return model

    def add_positional_encoding(self, inputs, seq_len, model_dim):
        positions = tf.range(0, seq_len, dtype=tf.float32)[:, tf.newaxis]
        dims = tf.range(0, model_dim, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000, (2 * (dims // 2)) / tf.cast(model_dim, tf.float32))
        angle_rads = positions * angle_rates
        angle_rads = tf.concat([tf.sin(angle_rads[:, 0::2]), tf.cos(angle_rads[:, 1::2])], axis=-1)
        return inputs + tf.expand_dims(angle_rads, axis=0)

    def get_callbacks(self):
        checkpoint_filepath = os.path.join(self.basepath, 'best_model.keras')
        return [
            EarlyStopping(monitor=self.objective, patience=self.chk_patience, verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, verbose=1)
        ]

    def tune(self):
        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            epochs=self.max_epochs,
            callbacks=self.get_callbacks()
        )
        return self.tuner
