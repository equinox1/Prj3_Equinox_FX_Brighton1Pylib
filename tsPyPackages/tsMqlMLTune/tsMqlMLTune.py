import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, 
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from keras_tuner import HyperModel
import numpy as np

# Transformer Block definition (used in transformer model)
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(0.1)
        self.dropout2 = Dropout(0.1)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define a HyperModel class for Keras Tuner
class HybridEnsembleHyperModel(kt.HyperModel):
    def __init__(self, input_shape, lstm_shape, cnn_shape, gru_shape, transformer_shape, objective, max_trials, executions_per_trial, directory, project_name, validation_split, epochs, batch_size, arraysize, channels):
        super(HybridEnsembleHyperModel, self).__init__()
        self.input_shape = input_shape
        self.lstm_shape = lstm_shape  # (timesteps, features)
        self.cnn_shape = cnn_shape    # (height, width, channels)
        self.gru_shape = gru_shape    # (timesteps, features)
        self.transformer_shape = transformer_shape  # (sequence_length, d_model)
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory
        self.project_name = project_name
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.arraysize = arraysize
        self.channels = channels

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        lstm_input = Input(shape=self.lstm_shape)
        cnn_input = Input(shape=self.cnn_shape)
        gru_input = Input(shape=self.gru_shape)
        transformer_input = Input(shape=self.transformer_shape)

        print("The build input shape required by the lstm model is:", lstm_input.shape)
        print("The build input shape required by the cnn model is:", cnn_input.shape)
        print("The build input shape required by the gru model is:", gru_input.shape)
        print("The build input shape required by the transformer model is:", transformer_input.shape)
        print("The build Incoming shape is:", inputs.shape)

        # Define the base models
        
        #LSTM
        batch_size=self.batch_size
        timesteps=self.X_train.shape[0]
        features=self.X_train.shape[1]
        lstm_input_layer = Input(shape=(batch_size, timesteps, features))
        self.lstm_shape=lstm_input_layer

        #CNN
        height=self.X_train.shape[0]
        width=self.X_train.shape[1]
        channels=self.channels
        cnn_input_layer = Input(shape=(height, width, channels))
        self.cnn_shape=cnn_input_layer
        
        #GRU
        timesteps=self.X_train.shape[0]
        features=self.X_train.shape[1]
        gru_input_layer = Input(shape=(timesteps, features))
        self.gru_shape=gru_input_layer
        
        #Transformer
        batch_size=self.batch_size
        sequence_length=self.X_train.shape[0]
        d_model=self.X_train.shape[1]
        transformer_input_layer = Input(shape=(sequence_length, d_model))
        self.transformer_layer = transformer_input_layer

        #Required input data
        print("tuner: input X_train shape:", self.X_train.shape)
        print("tuner: input X_train shape:length:",  len(self.X_train.shape))
        
        print("The input shape required by the lstm model is:", self.lstm_shape)
        print("The input shape required by the cnn model is:", self.cnn_shape)
        print("The input shape required by the gru model is:", self.gru_shape)
        print("The input shape required by the transformer model is:", self.transformer_shape)
        print("The Incoming shape is:", self.X_train.shape)
        

        
        # Base Models Definition
        # LSTM Model
        def build_lstm():
            x = LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)(lstm_input)
            x = Flatten()(x)
            x = Dense(hp.Int('lstm_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # 1D CNN Model
        def build_cnn():
            x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, step=32), 
                       kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
                       padding='same', activation='relu')(cnn_input)
            x = MaxPooling1D(pool_size=2, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # GRU Model
        def build_gru():
            x = GRU(hp.Int('gru_units', 32, 128, step=32), return_sequences=True)(gru_input)
            x = Flatten()(x)
            x = Dense(hp.Int('gru_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # Transformer Model
        def build_transformer():
            embed_dim = hp.Int('transformer_embed_dim', 32, 128, step=32)
            x = Dense(embed_dim, activation='relu')(transformer_input)  # Project input to the required embedding dimension
            x = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=hp.Int('transformer_num_heads', 2, 8, step=2),
                ff_dim=hp.Int('transformer_ff_dim', 32, 128, step=32)
            )(x)
            x = GlobalAveragePooling1D()(x)
            x = Flatten()(x)
            x = Dense(hp.Int('transformer_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # Build the base models
        lstm_output = build_lstm()
        cnn_output = build_cnn()
        gru_output = build_gru()
        transformer_output = build_transformer()

         # Combine all the outputs
        combined = Concatenate()([lstm_output, cnn_output, gru_output, transformer_output])

        # Dense layers for final prediction
        x = Dense(hp.Int('ensemble_dense_units', 64, 256, step=64), activation='relu')(combined)
        x = Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        output = Dense(1, activation='sigmoid')(x)

        # Define the model
        model = Model(inputs=[lstm_input, cnn_input, gru_input, transformer_input], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

# Class for running the tuner
class CMdtuner:
    def __init__(self, X_train_input_shape, X_train, y_train, lstm_shape, cnn_shape, gru_shape, transformer_shape, objective, max_trials, executions_per_trial, directory, project_name, validation_split, epochs, batch_size, arraysize, channels):
        self.input_shape = X_train_input_shape
        self.X_train = X_train
        self.y_train = y_train
        self.lstm_shape = lstm_shape
        self.cnn_shape = cnn_shape
        self.gru_shape = gru_shape
        self.transformer_shape = transformer_shape
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory
        self.project_name = project_name
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.arraysize = arraysize
        self.channels = channels
                
    def run_tuner(self):
        # Define the tuner
        tuner = kt.RandomSearch(hypermodel=HybridEnsembleHyperModel( 
                input_shape=self.input_shape,
                lstm_shape=self.lstm_shape,
                cnn_shape=self.cnn_shape,
                gru_shape=self.gru_shape,
                transformer_shape=self.transformer_shape,
                objective=self.objective,
                max_trials=self.max_trials,
                executions_per_trial=self.executions_per_trial,
                directory=self.directory,
                 project_name=self.project_name,
                validation_split=self.validation_split,
                epochs=self.epochs,
                batch_size=self.batch_size,
                arraysize=self.arraysize,
                channels=self.channels
            ),
        objective=self.objective,
        max_trials=self.max_trials,
        executions_per_trial=self.executions_per_trial,
        directory=self.directory,
        project_name=self.project_name
        )


        # Reshape input data
        print("tuner: input X_train shape:", self.X_train.shape)
        print("tuner: input X_train shape:length:",  len(self.X_train.shape))
        
        print("The input shape required by the lstm model is:", self.lstm_shape)
        print("The input shape required by the cnn model is:", self.cnn_shape)
        print("The input shape required by the gru model is:", self.gru_shape)
        print("The input shape required by the transformer model is:", self.transformer_shape)
        print("The Incoming shape is:", self.X_train.shape)
        
        # Run tuner search
        tuner.search(self.X_train, self.y_train, 
                     epochs=self.epochs, 
                     batch_size=self.batch_size,
                     validation_split=self.validation_split)

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"Best Hyperparameters: {best_hp.values}")
        return best_model