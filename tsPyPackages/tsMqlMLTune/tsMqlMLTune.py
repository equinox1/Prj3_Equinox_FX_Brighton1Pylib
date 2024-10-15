import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, 
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
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
    def __init__(self, input_shape):
        if not isinstance(input_shape, tuple):
            raise ValueError(f"Input shape must be a tuple, got {type(input_shape)}")
        self.input_shape = input_shape

    def build(self, hp):
        ### Base Models ###       
        # LSTM Model
        def build_lstm(hp):
            inputs = Input(shape=self.input_shape)
            x = LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)(inputs)
            x = Flatten()(x)
            x = Dense(hp.Int('lstm_dense_units', 32, 128, step=32), activation='relu')(x)
            return Model(inputs, x)

        # 1D CNN Model
        def build_cnn(hp):
            inputs = Input(shape=self.input_shape)
            x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, step=32), 
                       kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
                       padding='same', activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return Model(inputs, x)

        # GRU Model
        def build_gru(hp):
            inputs = Input(shape=self.input_shape)
            x = GRU(hp.Int('gru_units', 32, 128, step=32), return_sequences=True)(inputs)
            x = Flatten()(x)
            x = Dense(hp.Int('gru_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return Model(inputs, x)

        # Transformer Model
        def build_transformer(hp):
            inputs = Input(shape=self.input_shape)
            x = TransformerBlock(
                embed_dim=hp.Int('transformer_embed_dim', 32, 128, step=32),
                num_heads=hp.Int('transformer_num_heads', 2, 8, step=2),
                ff_dim=hp.Int('transformer_ff_dim', 32, 128, step=32)
            )(inputs)
            x = GlobalAveragePooling1D()(x)
            x = Flatten()(x)
            x = Dense(hp.Int('transformer_dense_units', 32, 128, step=32), activation='relu')(x)
            return Model(inputs, x)

        ### Hybrid Ensemble Model ###
        lstm_model = build_lstm(hp)
        cnn_model = build_cnn(hp)
        gru_model = build_gru(hp)
        transformer_model = build_transformer(hp)

        combined = Concatenate()([
            lstm_model.output, cnn_model.output, 
            gru_model.output, transformer_model.output
        ])

        x = Dense(hp.Int('ensemble_dense_units', 64, 256, step=64), activation='relu')(combined)
        x = Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=[lstm_model.input, cnn_model.input, gru_model.input, transformer_model.input], outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

# Class for running the tuner
class CMdtuner:
    def __init__(self, input_shape, X_train, y_train, max_trials, executions_per_trial, 
                 directory, project_name, validation_split, epochs, batch_size, arraysize):    
        self.input_shape = input_shape
        self.X_train = X_train
        self.y_train = y_train
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.directory = directory
        self.project_name = project_name
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.arraysize = arraysize

    def run_tuner(self):
        tuner = kt.RandomSearch(
            HybridEnsembleHyperModel(input_shape=self.input_shape), 
            objective='val_accuracy',  
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,  
            directory=self.directory,
            project_name=self.project_name
        )

        # Reshape input data
        new_shape = (self.X_train.shape[0], self.arraysize, 1)
        if np.prod(new_shape) == self.X_train.size:
            self.X_train = self.X_train.reshape(new_shape)
        else:
            raise ValueError(f"Cannot reshape array to {new_shape}")

        # Search for best hyperparameters
        tuner.search(self.X_train, self.y_train, validation_split=self.validation_split, 
                     epochs=self.epochs, batch_size=self.batch_size)

        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"Best hyperparameters: {best_hp.values}")
        
        # Return the best model
        return tuner.get_best_models(num_models=1)[0]
