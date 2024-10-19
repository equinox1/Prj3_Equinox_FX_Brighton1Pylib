import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC
import keras_tuner as kt
from keras_tuner import Hyperband
from ml_dtypes import float8_e5m2

# Class for running the tuner
class CMdtuner:
    def __init__(self, X_train, y_train, cnn_model, lstm_model, gru_model, transformer_model, lstm_shape, lstm_features, cnn_shape, cnn_features, gru_shape, gru_features, transformer_shape, transformer_features, objective, max_trials, executions_per_trial, directory, project_name, validation_split, epochs, batch_size, factor, channels, dropout):
        self.X_train = X_train
        self.y_train = y_train
        self.cnn_model = cnn_model
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.transformer_model = transformer_model
        self.lstm_shape = lstm_shape
        self.lstm_features = lstm_features
        self.cnn_shape = cnn_shape
        self.cnn_features = cnn_features
        self.gru_shape = gru_shape
        self.gru_features = gru_features
        self.transformer_shape = transformer_shape
        self.transformer_features = transformer_features
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory
        self.project_name = project_name
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.factor = factor
        self.channels = channels
        self.dropout = dropout
        
        # Initialize the Keras Tuner
        self.tuner = Hyperband(
            self.build_model,
            objective=self.objective,
            max_epochs=self.epochs,
            executions_per_trial=self.executions_per_trial,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=True,
            factor=self.factor
        )

    def build_model(self, hp):
        print("Building model with hp:", hp)

        # Initialize all to None
        x_cnn = x_lstm = x_gru = x_transformer = None
        cnninputs = lstminputs = gruinputs = transformerinputs = None

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
                          kernel_size=hp.Int('conv_kernel_size', min_value=3, max_value=7, step=2),
                          activation='relu')(cnninputs)
            x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
            x_cnn = Dropout(0.2)(x_cnn)
            x_cnn = Flatten()(x_cnn)

        # LSTM Layers
        if self.lstm_model: 
            x_lstm = LSTM(hp.Int('lstm_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(lstminputs)
            x_lstm = LSTM(hp.Int('lstm_units_2', min_value=32, max_value=128, step=32))(x_lstm)
            x_lstm = Dropout(0.2)(x_lstm)
            x_lstm = Flatten()(x_lstm)

        # GRU Layers
        if self.gru_model:
            x_gru = GRU(hp.Int('gru_units_1', min_value=32, max_value=128, step=32), return_sequences=True)(gruinputs)
            x_gru = GRU(hp.Int('gru_units_2', min_value=32, max_value=128, step=32))(x_gru)
            x_gru = Dropout(0.2)(x_gru)
            x_gru = Flatten()(x_gru)

        # Transformer Layers
        if self.transformer_model:
            x_transformer = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(transformerinputs, transformerinputs)
            x_transformer = LayerNormalization(epsilon=1e-6)(x_transformer)
            x_transformer = Dropout(0.2)(x_transformer)
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
        model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy', AUC()])

        return model

    def run_tuner(self):
        # Prepare the input data based on the enabled models
        # Assuming self.X_train has two dimensions (rows, columns)
        #inputs.append(self.X_train[:, :self.cnn_shape[1]])


        inputs = []
        if self.cnn_model:
            inputs.append(self.X_train[:, :self.cnn_shape[1], :self.cnn_features])
        if self.lstm_model:
            inputs.append(self.X_train[:, :self.lstm_shape[1], :self.lstm_features])
        if self.gru_model:
            inputs.append(self.X_train[:, :self.gru_shape[1], :self.gru_features])
        if self.transformer_model:
            inputs.append(self.X_train[:, :self.transformer_shape[1], :self.transformer_features])
    
        print("Inputs shape:", [input_data.shape for input_data in inputs])

        # Ensure inputs are correctly shaped
        inputs = [input_data for input_data in inputs if input_data.shape[1] > 0 and input_data.shape[2] > 0]

        # Run the tuner search
        self.tuner.search(inputs, self.y_train, validation_split=self.validation_split, epochs=self.epochs, batch_size=self.batch_size)
    
        # Get the best model
        best_model = self.tuner.get_best_models()[0]
        return best_model