import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout,
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import keras_tuner as kt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CMdtuner:
    def __init__(self, **kwargs):
        """Initialize the tuner with hyperparameters and model configurations."""
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        self.initialize_parameters(kwargs)
        self.setup_tensorflow()
        self.validate_config()
        self.initialize_tuner()

    def initialize_parameters(self, kwargs):
        """Initialize all hyperparameters and model configurations."""
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')
        self.keras_tuner = kwargs.get('keras_tuner', 'hyperband')
        self.model_types = {
            'cnn': kwargs.get('cnn_model',True),
            'lstm': kwargs.get('lstm_model', True),
            'gru': kwargs.get('gru_model', True),
            'transformer': kwargs.get('transformer_model', True)
        }
        self.multi_inputs = kwargs.get('multi_inputs', False)
        self.multi_outputs = kwargs.get('multi_outputs', False)
        self.multi_branches = kwargs.get('multi_branches',True)
        self.data_input_shape = kwargs.get('data_input_shape')
        self.batch_size = kwargs.get('batch_size', 32)
        self.dropout = kwargs.get('dropout', 0.3)
        self.steps_per_execution = kwargs.get('steps_per_execution', 32)
        self.max_epochs = kwargs.get('max_epochs', 10)
        self.min_epochs = kwargs.get('min_epochs', 1)
        self.verbose = kwargs.get('verbose', 1)
        self.checkpoint_filepath = kwargs.get('checkpoint_filepath', 'best_model.keras')
        self.basepath = kwargs.get('basepath', 'tuner_results')
        self.project_name = kwargs.get('project_name', 'cm_tuning')
        self.loss_functions = {
            'mse': tf.keras.losses.MeanSquaredError,
            'binary_crossentropy': tf.keras.losses.BinaryCrossentropy,
            'mae': tf.keras.losses.MeanAbsoluteError,
        }
        self.metrics = {
            'accuracy': tf.keras.metrics.Accuracy,
            'mae': tf.keras.metrics.MeanAbsoluteError,
            'mse': tf.keras.metrics.MeanSquaredError,
        }
        os.makedirs(self.basepath, exist_ok=True)

    def setup_tensorflow(self):
        """Enable TensorFlow optimizations."""
        tf.config.optimizer.set_jit(True)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        self.strategy = tf.distribute.MirroredStrategy()
        logging.info(f"Using {self.strategy.num_replicas_in_sync} devices")

    def validate_config(self):
        """Ensure at least one model type is enabled."""
        if not any(self.model_types.values()):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")

    def initialize_tuner(self,):
        """Set up the Keras Tuner."""
        hp = kt.HyperParameters()
        tuner_classes = {
            'random': kt.RandomSearch,
            'hyperband': kt.Hyperband,
            'bayesian': kt.BayesianOptimization
        }
        if self.keras_tuner not in tuner_classes:
            raise ValueError(f"Unsupported keras_tuner type: {self.keras_tuner}")
        self.tuner = tuner_classes[self.keras_tuner](
            hypermodel=self.build_model,
            objective='val_loss',
            max_epochs=self.max_epochs,
            directory=self.basepath,
            project_name=self.project_name,
            overwrite=True,
        )
        self.tuner.search_space_summary()

    def build_model(self, hp):
        """Construct the model based on the selected configurations."""
        inputs = Input(shape=self.data_input_shape, name='input_layer')
        branches = []
        if self.model_types['cnn']:
            branches.append(self.build_cnn_branch(inputs, hp))
        if self.model_types['lstm']:
            branches.append(self.build_lstm_branch(inputs, hp))
        if self.model_types['gru']:
            branches.append(self.build_gru_branch(inputs, hp))
        if self.model_types['transformer']:
            branches.append(self.build_transformer_branch(inputs, hp))
        merged = Concatenate()(branches) if len(branches) > 1 else branches[0]
        output = Dense(1, activation='sigmoid')(Dropout(self.dropout)(merged))
        model = Model(inputs, output)
        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def build_cnn_branch(self, inputs, hp):
        x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, 32), kernel_size=3, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        return Flatten()(x)

    def build_lstm_branch(self, inputs, hp):
        return LSTM(units=hp.Int('lstm_units', 32, 128, 32), activation='tanh', return_sequences=False)(inputs)

    def build_gru_branch(self, inputs, hp):
        return GRU(units=hp.Int('gru_units', 32, 128, 32), activation='tanh', return_sequences=False)(inputs)

    def build_transformer_branch(self, inputs, hp):
        attn_out = MultiHeadAttention(num_heads=4, key_dim=64)(inputs, inputs)
        attn_out = LayerNormalization()(inputs + attn_out)
        ffn_out = Dense(128, activation='relu')(attn_out)
        return GlobalAveragePooling1D()(ffn_out)

    def run_search(self):
        """Start hyperparameter tuning."""
        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            epochs=self.max_epochs,
            callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
        )

    def export_best_model(self):
        """Save the best model."""
        best_model = self.tuner.get_best_models(num_models=1)[0]
        best_model.save(os.path.join(self.basepath, self.project_name, 'best_model.keras'))
        logging.info("Best model saved.")
