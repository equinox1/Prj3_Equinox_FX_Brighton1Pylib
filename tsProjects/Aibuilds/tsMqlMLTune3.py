import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from keras_tuner import Hyperband
import numpy as np  # Use this for float types if needed

# Class for running the tuner
class CMdtuner:
    def __init__(self, mv_X_train, mv_y_train, mp_cnn_model, mp_lstm_model, mp_gru_model, mp_transformer_model, 
                 mp_lstm_input_shape, mp_lstm_features, mp_cnn_input_shape, mp_cnn_features, 
                 mp_gru_input_shape, mp_gru_features, mp_transformer_input_shape, mp_transformer_features, 
                 mp_objective, mp_max_epochs, mp_factor, mp_seed, mp_hyperband_iterations, 
                 mp_tune_new_entries, mp_allow_new_entries, mp_max_retries_per_trial, mp_max_consecutive_failed_trials, 
                 mp_validation_split, mp_epochs, mp_batch_size, mp_dropout, mp_oracle, mp_hypermodel, mp_max_model_size, 
                 mp_optimizer, mp_loss, mp_metrics, mp_distribution_strategy, mp_directory, mp_project_name, mp_logger, 
                 mp_tuner_id, mp_overwrite, mp_executions_per_trial):
        
        # Initialize all parameters
        self.X_train = mv_X_train
        self.y_train = mv_y_train
        self.cnn_model = mp_cnn_model
        self.lstm_model = mp_lstm_model
        self.gru_model = mp_gru_model
        self.transformer_model = mp_transformer_model
        self.lstm_shape = mp_lstm_input_shape
        self.lstm_features = mp_lstm_features
        self.cnn_shape = mp_cnn_input_shape
        self.cnn_features = mp_cnn_features
        self.gru_shape = mp_gru_input_shape
        self.gru_features = mp_gru_features
        self.transformer_shape = mp_transformer_input_shape
        self.transformer_features = mp_transformer_features
        self.objective = mp_objective
        self.max_epochs = mp_max_epochs
        self.factor = mp_factor
        self.seed = mp_seed
        self.hyperband_iterations = mp_hyperband_iterations
        self.tune_new_entries = mp_tune_new_entries
        self.allow_new_entries = mp_allow_new_entries
        self.max_retries_per_trial = mp_max_retries_per_trial
        self.max_consecutive_failed_trials = mp_max_consecutive_failed_trials
        self.validation_split = mp_validation_split 
        self.epochs = mp_epochs
        self.batch_size = mp_batch_size
        self.dropout = mp_dropout
        self.oracle = mp_oracle
        self.hypermodel = mp_hypermodel
        self.max_model_size = mp_max_model_size
        self.optimizer = mp_optimizer
        self.loss = mp_loss
        self.metrics = mp_metrics
        self.distribution_strategy = mp_distribution_strategy
        self.directory = mp_directory
        self.project_name = mp_project_name
        self.logger = mp_logger
        self.tuner_id = mp_tuner_id
        self.overwrite = mp_overwrite
        self.executions_per_trial = mp_executions_per_trial

        # Initialize the Keras Tuner
        self.tuner = Hyperband(
            hypermodel=self.build_model,
            objective=self.objective,
            max_epochs=self.max_epochs,
            executions_per_trial=self.executions_per_trial,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=self.overwrite,
            factor=self.factor
        )

    def build_model(self, hp):
        print("Building model with hp:", hp)
        x_cnn = x_lstm = x_gru = x_transformer = None
        cnninputs = lstminputs = gruinputs = transformerinputs = None

        # Ensure at least one model is enabled
        if not (self.cnn_model or self.lstm_model or self.gru_model or self.transformer_model):
            raise ValueError("At least one model (cnn_model, lstm_model, gru_model, transformer_model) must be enabled.")

        # Define input shapes for the models
        if self.cnn_model:
            cnninputs = Input(shape=(self.cnn_shape[1], self.cnn_features))
            print("Set cnninputs shape:", cnninputs.shape)
        if self.lstm_model:
            lstminputs = Input(shape=(self.lstm_shape[1], self.lstm_features))
            print("Set lstminputs shape:", lstminputs.shape)
        if self.gru_model:
            gruinputs = Input(shape=(self.gru_shape[1], self.gru_features))
            print("Set gruinputs shape:", gruinputs.shape)
        if self.transformer_model:
            transformerinputs = Input(shape=(self.transformer_shape[1], self.transformer_features))
            print("Set transformerinputs shape:", transformerinputs.shape)

        # CNN Layers
        if self.cnn_model:
            x_cnn = Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
                          kernel_size=hp.Choice('conv_kernel_size', values=[1, 2, 3]),
                          activation='relu')(cnninputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Dropout(self.dropout)(x_cnn)
            x_cnn = Flatten()(x_cnn)

        # LSTM Layers
        if self.lstm_model:
            x_lstm = LSTM(hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(lstminputs)
            x_lstm = LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32))(x_lstm)
            x_lstm = Dropout(self.dropout)(x_lstm)
            x_lstm = Flatten()(x_lstm)

        # GRU Layers
        if self.gru_model:
            x_gru = GRU(hp.Int('gru_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(gruinputs)
            x_gru = GRU(hp.Int('gru_units_2', min_value=32, max_value=128, step=32))(x_gru)
            x_gru = Dropout(self.dropout)(x_gru)
            x_gru = Flatten()(x_gru)

        # Transformer Layers
        if self.transformer_model:
            x_transformer = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), 
                                               key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(transformerinputs, transformerinputs)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = Dropout(self.dropout)(x_transformer)
            x_transformer = GlobalAveragePooling1D()(x_transformer)

        # Concatenate Layers if multiple models are enabled
        concat_layers = [layer for layer in [x_cnn, x_lstm, x_gru, x_transformer] if layer is not None]
        if len(concat_layers) > 1:
            x = Concatenate()(concat_layers)
        elif len(concat_layers) == 1:
            x = concat_layers[0]
        else:
            raise ValueError("At least one model (cnn_model, lstm_model, gru_model, transformer_model) must be enabled.")

        # Dense layer for final prediction
        x = Dense(1, activation='sigmoid')(x)

        # Create the model
        model = Model(inputs=[input for input in [cnninputs, lstminputs, gruinputs, transformerinputs] if input is not None], outputs=x)

        # Compile the model
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model

    def run_tuner(self):
        self.tuner.search(self.X_train, self.y_train, 
                          validation_split=self.validation_split, 
                          epochs=self.epochs, 
                          batch_size=self.batch_size, 
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
        best_model = self.tuner.get_best_models()[0]
        return best_model
