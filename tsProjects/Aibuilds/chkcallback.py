import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from keras_tuner import Hyperband
import os

class CMdtuner:
    def __init__(self, X_train, y_train, cnn_model, lstm_model, gru_model, transformer_model, shapes_and_features, objective, max_epochs, factor, seed, hyperband_iterations, tune_new_entries, allow_new_entries, max_retries_per_trial, max_consecutive_failed_trials, validation_split, epochs, batch_size, dropout, oracle, hypermodel, max_model_size, optimizer, loss, metrics, distribution_strategy, directory, project_name, logger, tuner_id, overwrite, executions_per_trial):
        self.X_train = X_train
        self.y_train = y_train
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.transformer_model = transformer_model
        self.shapes_and_features = shapes_and_features
        self.objective = objective
        self.max_epochs = max_epochs
        self.factor = factor
        self.seed = seed
        self.hyperband_iterations = hyperband_iterations
        self.tune_new_entries = tune_new_entries
        self.allow_new_entries = allow_new_entries
        self.max_retries_per_trial = max_retries_per_trial
        self.max_consecutive_failed_trials = max_consecutive_failed_trials
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.oracle = oracle
        self.hypermodel = hypermodel
        self.max_model_size = max_model_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.distribution_strategy = distribution_strategy
        self.directory = directory
        self.project_name = project_name
        self.logger = logger
        self.tuner_id = tuner_id
        self.overwrite = overwrite
        self.executions_per_trial = executions_per_trial

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
        inputs = []
        layers = []

        # Define input shapes for the models
        for model_type, (input_shape, features) in self.shapes_and_features.items():
            if model_type == 'cnn' and self.cnn_model:
                input_tensor = Input(shape=(input_shape[1], features))
                inputs.append(input_tensor)
                # CNN Layers
                x = Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
                           kernel_size=hp.Choice('conv_kernel_size', values=[1, 2, 3]),
                           activation='relu')(input_tensor)
                x = MaxPooling1D(pool_size=2)(x)
                x = Dropout(self.dropout)(x)
                x = Flatten()(x)
                layers.append(x)
            elif model_type == 'lstm' and self.lstm_model:
                input_tensor = Input(shape=(input_shape[1], features))
                inputs.append(input_tensor)
                # LSTM Layers
                x = LSTM(hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(input_tensor)
                x = LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32))(x)
                x = Dropout(self.dropout)(x)
                x = Flatten()(x)
                layers.append(x)
            elif model_type == 'gru' and self.gru_model:
                input_tensor = Input(shape=(input_shape[1], features))
                inputs.append(input_tensor)
                # GRU Layers
                x = GRU(hp.Int('gru_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(input_tensor)
                x = GRU(hp.Int('gru_units_2', min_value=32, max_value=128, step=32))(x)
                x = Dropout(self.dropout)(x)
                x = Flatten()(x)
                layers.append(x)
            elif model_type == 'transformer' and self.transformer_model:
                input_tensor = Input(shape=(input_shape[1], features))
                inputs.append(input_tensor)
                # Transformer Layers
                x = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1),
                                       key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(input_tensor, input_tensor)
                x = LayerNormalization(epsilon=1e-6)(x)
                x = Dropout(self.dropout)(x)
                x = GlobalAveragePooling1D()(x)
                layers.append(x)

        # Concatenate Layers if multiple models are enabled
        if len(layers) > 1:
            x = Concatenate()(layers)
        elif len(layers) == 1:
            x = layers[0]
        else:
            raise ValueError("At least one model (cnn_model, lstm_model, gru_model, transformer_model) must be enabled.")

        # Dense layer for final prediction
        x = Dense(1, activation='sigmoid')(x)
        # Compile the model with learning rate tuning
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG', default=1e-3)
        model = Model(inputs=inputs, outputs=x)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss=BinaryCrossentropy(), metrics=['accuracy', AUC()])

        return model

    def _get_checkpoint_fname(self, trial_id):
        return os.path.join(self.directory, self.project_name, f'trial_{trial_id}_checkpoint')

    def run_tuner(self):
        checkpoint_callback = ModelCheckpoint(filepath=self._get_checkpoint_fname('{trial_id}'), save_weights_only=True, save_best_only=True, monitor='val_loss')
        
        self.tuner.search(self.X_train, self.y_train,
                          validation_split=self.validation_split,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3), checkpoint_callback])

        best_hp = self.tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
        model = self.build_model(best_hp)
        model.load_weights(self._get_checkpoint_fname(best_hp['trial_id']))

        return model
