import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, 
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

class MyLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)

    def call(self, x):
        return tf.nn.relu(x)  # Assuming a ReLU activation function for demonstration

# Transformer Block definition
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

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

# Define a HyperModel class for Keras Tuner
class HybridEnsembleHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super(HybridEnsembleHyperModel, self).__init__()
        self.input_shape = input_shape

    def build(self, hp):
        inputs = Input(shape=self.input_shape)

        # Reshape input to ensure compatibility with all models
        reshaped_inputs = tf.expand_dims(inputs, axis=-1) if len(self.input_shape) == 2 else inputs

        ### Base Models ###       
        # LSTM Model
        def build_lstm(hp, inputs):
            x = LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)(inputs)
            x = Flatten()(x)
            x = Dense(hp.Int('lstm_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # 1D CNN Model
        def build_cnn(hp, inputs):
            x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, step=32), 
                       kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
                       padding='same', activation='relu')(inputs)
            x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # GRU Model
        def build_gru(hp, inputs):
            x = GRU(hp.Int('gru_units', 32, 128, step=32), return_sequences=True)(inputs)
            x = Flatten()(x)
            x = Dense(hp.Int('gru_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # Transformer Model
        def build_transformer(hp, inputs):
            embed_dim = hp.Int('transformer_embed_dim', 32, 128, step=32)
            reshaped_transformer_inputs = Reshape((-1, inputs.shape[-1]))(inputs)
            x = Dense(embed_dim, activation='relu')(reshaped_transformer_inputs)
            x = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=hp.Int('transformer_num_heads', 2, 8, step=2),
                ff_dim=hp.Int('transformer_ff_dim', 32, 128, step=32)
            )(x)
            x = GlobalAveragePooling1D()(x)
            x = Dense(hp.Int('transformer_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        ### Hybrid Ensemble Model ###
        lstm_output = build_lstm(hp, reshaped_inputs)
        cnn_output = build_cnn(hp, reshaped_inputs)
        gru_output = build_gru(hp, reshaped_inputs)
        transformer_output = build_transformer(hp, reshaped_inputs)

        combined = Concatenate()([lstm_output, cnn_output, gru_output, transformer_output])

        x = Dense(hp.Int('ensemble_dense_units', 64, 256, step=64), activation='relu')(combined)
        x = Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1))(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

# Class for running the tuner
class CMdtuner:
    def __init__(self, input_shape, X_train, y_train, objective, max_trials, executions_per_trial, 
                 directory, project_name, validation_data, epochs, batch_size):
        self.input_shape = input_shape
        self.X_train = X_train
        self.y_train = y_train
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.objective = objective
        self.directory = directory
        self.project_name = project_name
        self.validation_data = validation_data
        self.epochs = epochs
        self.batch_size = batch_size

    def run_tuner(self):
        tuner = kt.RandomSearch(
            HybridEnsembleHyperModel(input_shape=self.input_shape), 
            objective=self.objective,  
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,  
            directory=self.directory,
            project_name=self.project_name
        )
        
        # Search for best hyperparameters
        tuner.search(self.X_train, self.y_train, validation_data=self.validation_data, 
                     epochs=self.epochs, batch_size=self.batch_size)

        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Return the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        return best_model