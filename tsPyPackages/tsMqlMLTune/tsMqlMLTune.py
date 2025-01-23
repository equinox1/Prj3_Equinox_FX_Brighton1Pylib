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

class CMdtuner:
    def __init__(self, **kwargs):
        self.hypermodel_params = kwargs.get('hypermodel_params', {})
        self.traindataset = kwargs.get('traindataset')
        self.valdataset = kwargs.get('valdataset')
        self.testdataset = kwargs.get('testdataset')

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
        self.num_trials = kwargs.get('num_trials', 10)
        self.num_models = kwargs.get('num_models', 1)

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

    def enable_debugging(self, kwargs):
        if kwargs.get('tf1', False):
            tf.debugging.set_log_device_placement(True)
        if kwargs.get('tf2', False):
            tf.debugging.enable_check_numerics()

    def prepare_shapes(self):
        if self.data_input_shape:
            self.main_input_shape = self.get_shape(self.data_input_shape)
            if self.cnn_model:
                self.input_shapes['cnn_input'] = self.main_input_shape
            if self.lstm_model:
                self.input_shapes['lstm_input'] = self.main_input_shape
            if self.gru_model:
                self.input_shapes['gru_input'] = self.main_input_shape
            if self.transformer_model:
                self.input_shapes['transformer_input'] = self.main_input_shape

    def get_shape(self, data_shape):
        if not data_shape:
            raise ValueError("Data shape cannot be None. Please provide a valid input shape.")
        if len(data_shape) not in [2, 3, 4]:
            raise ValueError(f"Unsupported input shape: {data_shape}. Must be 2D, 3D, or 4D.")
        return tuple(data_shape)

    def validate_config(self):
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model type (CNN, LSTM, GRU, Transformer) must be enabled.")
        if not self.data_input_shape:
            raise ValueError("Data input shape must be specified.")

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
        branches = []
        inputs = []

        # CNN branch
        if self.cnn_model:
            cnn_input = Input(shape=self.input_shapes['cnn_input'], name='cnn_input')
            x_cnn = Conv1D(
                filters=hp.Int('cnn_filters', 32, 128, step=32),
                kernel_size=hp.Int('cnn_kernel_size', 2, 5),
                activation=hp.Choice('cnn_activation', ['relu', 'tanh'])
            )(cnn_input)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Flatten()(x_cnn)
            branches.append(x_cnn)
            inputs.append(cnn_input)

        # LSTM branch
        if self.lstm_model:
            lstm_input = Input(shape=self.input_shapes['lstm_input'], name='lstm_input')
            x_lstm = LSTM(
                units=hp.Int('lstm_units', 32, 128, step=32),
                return_sequences=True
            )(lstm_input)
            x_lstm = LSTM(
                units=hp.Int('lstm_units', 32, 128, step=32)
            )(x_lstm)
            branches.append(x_lstm)
            inputs.append(lstm_input)

        # GRU branch
        if self.gru_model:
            gru_input = Input(shape=self.input_shapes['gru_input'], name='gru_input')
            x_gru = GRU(
                units=hp.Int('gru_units', 32, 128, step=32),
                return_sequences=True
            )(gru_input)
            x_gru = GRU(
                units=hp.Int('gru_units', 32, 128, step=32)
            )(x_gru)
            branches.append(x_gru)
            inputs.append(gru_input)

        # Transformer branch
        if self.transformer_model:
            transformer_input = Input(shape=self.input_shapes['transformer_input'], name='transformer_input')
            x_trans = MultiHeadAttention(
                num_heads=hp.Int('num_heads', 2, 4),
                key_dim=hp.Int('key_dim', 32, 128, step=32)
            )(transformer_input, transformer_input)
            x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
            x_trans = GlobalAveragePooling1D()(x_trans)
            branches.append(x_trans)
            inputs.append(transformer_input)

        # Combine branches
        if len(branches) > 1:
            x = Concatenate()(branches)
        else:
            x = branches[0]

        # Dense layers
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.dropout)(x)
        output = Dense(1, activation='linear')(x)


        #conactatenate all inputs as is or use only 1 input given 1 data input passed
        if self.multi_inputs:
            inputs = inputs
        else:
            inputs = inputs[0]
        
        model = Model(inputs=inputs, outputs=output)

        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        if optimizer == 'adam':
            opt = Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='LOG'))
        elif optimizer == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='LOG'))
        elif optimizer == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='LOG'))

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

    def export_best_model(self):
        try:
            best_model = self.tuner.get_best_models(num_models=1)[0]
            export_path = os.path.join(self.basepath, 'final_model')
            best_model.save(export_path, save_format='tf')
            print(f"Model saved to {export_path}")
        except Exception as e:
            print(f"Error saving the model: {e}")

    def run_tuner(self):
        # Define a ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        )

        # Define EarlyStopping callback
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1
        )

        # Run the tuner
        self.tuner.search(
            self.traindataset,
            validation_data=self.valdataset,
            epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint_callback, early_stopping_callback]
        )

        # Get the best model
        best_model = self.tuner.get_best_models(self.num_models)[0]
        return best_model


