import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from keras_tuner import Hyperband
import os  # Import the os module

class CMdtuner:
    def __init__(self, lv_X_train, lv_y_train, lp_cnn_model, lp_lstm_model, lp_gru_model, lp_transformer_model,
                 lp_shapes_and_features, lp_objective, lp_max_epochs, lp_factor, lp_seed, lp_hyperband_iterations,
                 lp_tune_new_entries, lp_allow_new_entries, lp_max_retries_per_trial, lp_max_consecutive_failed_trials,
                 lp_validation_split, lp_epochs, lp_batch_size, lp_dropout, lp_oracle, lp_hypermodel, lp_max_model_size, lp_optimizer,
                 lp_loss, lp_metrics, lp_distribution_strategy, lp_directory, lp_project_name, lp_logger, lp_tuner_id,
                 lp_overwrite, lp_executions_per_trial):
        # Set the input data
        self.X_train = lv_X_train
        self.y_train = lv_y_train
        self.cnn_model = lp_cnn_model
        self.lstm_model = lp_lstm_model
        self.gru_model = lp_gru_model
        self.transformer_model = lp_transformer_model
        self.shapes_and_features = lp_shapes_and_features
        self.objective = lp_objective
        self.max_epochs = lp_max_epochs
        self.factor = lp_factor
        self.seed = lp_seed
        self.hyperband_iterations = lp_hyperband_iterations
        self.tune_new_entries = lp_tune_new_entries 
        self.allow_new_entries = lp_allow_new_entries
        self.max_retries_per_trial = lp_max_retries_per_trial   
        self.max_consecutive_failed_trials = lp_max_consecutive_failed_trials
        self.validation_split = lp_validation_split 
        self.epochs = lp_epochs
        self.batch_size = lp_batch_size
        self.dropout = lp_dropout   
        self.oracle = lp_oracle
        self.hypermodel = lp_hypermodel
        self.max_model_size = lp_max_model_size
        self.optimizer = lp_optimizer
        self.loss = lp_loss
        self.metrics = lp_metrics
        self.distribution_strategy = lp_distribution_strategy
        self.directory = lp_directory
        self.project_name = lp_project_name
        self.logger = lp_logger
        self.tuner_id = lp_tuner_id
        self.overwrite = lp_overwrite
        self.executions_per_trial = lp_executions_per_trial

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

    def run_tuner(self):
        # Define a ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.directory, 'best_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )
        
        # Run the hyperparameter tuning
        self.tuner.search(self.X_train, self.y_train,
                          validation_split=self.validation_split,
                          epochs=self.epochs,
                          batch_size=self.batch_size,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=3), checkpoint_callback])
        
        # Get the best model
        best_model = self.tuner.get_best_models()[0]
        return best_model