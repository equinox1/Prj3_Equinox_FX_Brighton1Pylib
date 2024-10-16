import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, 
    Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, TimeDistributed
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

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

# Define a HyperModel class for Keras Tuner
class HybridEnsembleHyperModel(kt.HyperModel):
    def __init__(self, input_shape):
        super(HybridEnsembleHyperModel, self).__init__()
        if not isinstance(input_shape, tuple):
            raise ValueError(f"Input shape must be a tuple, got {type(input_shape)}")
        self.input_shape = input_shape

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        lstm_shape = self.lstm_shape
        cnn_shape = self.cnn_shape
        gru_shape = self.gru_shape
        transformer_shape = self.transformer_shape

        # Ensure inputs have the correct shape for TimeDistributed layer
        #reshaped_inputs = tf.expand_dims(inputs, axis=-1) if len(self.input_shape) == 2 else inputs

        ### Base Models ###       
        # LSTM Model
        def build_lstm(hp):
            x = LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)(lstm_shape)
            x = Flatten()(x)
            x = Dense(hp.Int('lstm_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # 1D CNN Model
        def build_cnn(hp):
            x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, step=32), 
                       kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
                       padding='same', activation='relu')(cnn_shape)
            x = MaxPooling1D(pool_size=2, strides=1, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # GRU Model
        def build_gru(hp):
            x = GRU(hp.Int('gru_units', 32, 128, step=32), return_sequences=True)(gru_shape)
            x = Flatten()(x)
            x = Dense(hp.Int('gru_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # Transformer Model
        def build_transformer(hp):
            embed_dim = hp.Int('transformer_embed_dim', 32, 128, step=32)
            reshaped_transformer_inputs = tf.reshape(transformer_shape)
            x = Dense(embed_dim, activation='relu')(reshaped_transformer_inputs)  # Project input to the required embedding dimension
            x = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=hp.Int('transformer_num_heads', 2, 8, step=2),
                ff_dim=hp.Int('transformer_ff_dim', 32, 128, step=32)
            )(x)
            x = GlobalAveragePooling1D()(x)
            x = Flatten()(x)
            x = Dense(hp.Int('transformer_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        ### Hybrid Ensemble Model ###
        lstm_output = build_lstm(hp)
        cnn_output = build_cnn(hp)
        gru_output = build_gru(hp)
        transformer_output = build_transformer(hp)

        combined = Concatenate()([
            lstm_output, cnn_output, 
            gru_output, transformer_output
        ])

        inputs = Concatenate()([lstm_shape, cnn_shape, gru_shape, transformer_shape])

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
                 directory, project_name, validation_data, epochs, batch_size, arraysize, lstm_shape,
                 cnn_shape, gru_shape, transformer_shape,channels):    
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
        self.arraysize = arraysize

        self.lstm_shape = None
        self.cnn_shape = None
        self.gru_shape = None
        self.transformer_shape = None
        self.channels = channels

        print("init input_shape:", self.input_shape, " param:", input_shape)
        print("init X_train:", self.X_train, " param:", X_train)
        print("init y_train:", self.y_train, " param:", y_train)
        print("init max_trials:", self.max_trials, " param:", max_trials)
        print("init executions_per_trial:", self.executions_per_trial, " param:", executions_per_trial)
        print("init objective:", self.objective, " param:", objective)
        print("init directory:", self.directory, " param:", directory)
        print("init project_name:", self.project_name, " param:", project_name)
        print("init validation_data:", self.validation_data, " param:", validation_data)
        print("init epochs:", self.epochs, " param:", epochs)
        print("init batch_size:", self.batch_size, " param:", batch_size)
        print("init arraysize:", self.arraysize, " param:", arraysize)
        

    def run_tuner(self):
        tuner = kt.RandomSearch(
            HybridEnsembleHyperModel(input_shape=self.input_shape), 
            objective=self.objective,  
            max_trials=self.max_trials,
            executions_per_trial=self.executions_per_trial,  
            directory=self.directory,
            project_name=self.project_name
        )

               
        
        """
        LSTMs ((Long Short-Term Memory)) are commonly used for sequential data, like time series or text.
        ====================================================================
        Input shape: (batch_size, timesteps, features)
        batch_size: Number of samples processed before the model often left unspecified in the input layer).
        timesteps: The number of time steps or sequence length.
        features: The number of features at each timestep (dimensionality of the input at each step).
        input_layer = keras.Input(shape=(timesteps, features))
        example:
        ========
        input_layer = keras.Input(shape=(timesteps, features))
        lstm_layer = keras.layers.LSTM(64)(input_layer)
        
        CNN (Convolutional Neural Network)
        ===================================
        CNNs are commonly used for image data but can be applied to other grid-like data,
        such as audio spectrograms or 1D signals.

        Input shape (for images): (height, width, channels)
        height and width: Dimensions of the image.
        channels: Number of color channels (e.g., 3 for RGB, 1 for grayscale).

        example:
        ========
        input_layer = keras.Input(shape=(height, width, channels))
        cnn_layer = keras.layers.Conv2D(32, (3, 3))(input_layer)

        GRU (Gated Recurrent Unit)
        ==========================
        GRUs are similar to LSTMs but have a simpler structure.
        Like LSTM, GRU is also used for sequential data.
        Input shape: Same as LSTM, i.e., (batch_size, timesteps, features).

         Transformer
         ==========================
        Transformers can handle sequential data, including text, and are more flexible regarding their input shape.

        Input shape (for sequence data): (batch_size, sequence_length, d_model)
        sequence_length: Length of the sequence (number of tokens or steps).
        d_model: Dimensionality of the embedding space (i.e., the number of features per token/step).

        input_layer = keras.Input(shape=(sequence_length, d_model))
        transformer_layer = keras.layers.TransformerBlock(embedding_dim=d_model, num_heads=8)(input_layer)
        
        Summary:
        LSTM/GRU: (timesteps, features)
        CNN (2D for images): (height, width, channels)
        CNN (1D for sequences): (timesteps, features)
        Transformer: (sequence_length, d_model)

        You can always leave batch_size undefined when defining the input shape since Keras will 
        infer it automatically during training.

        """
        #LSTM
        batch_size=self.batch_size
        timesteps=self.X_train.shape[0]
        features=self.X_train.shape[1]
        lstm_input_layer = keras.Input(shape=(batch_size, timesteps, features))
        self.lstm_shape=lstm_input_layer

        #CNN
        height=self.X_train.shape[0]
        width=self.X_train.shape[1]
        channels=self.channels
        cnn_input_layer = keras.Input(shape=(height, width, channels))
        self.cnn_shape=cnn_input_layer
        
        #GRU
        timesteps=self.X_train.shape[0]
        features=self.X_train.shape[1]
        gru_input_layer = keras.Input(shape=(timesteps, features))
        self.gru_shape=gru_input_layer
        
        #Transformer
        batch_size=self.batch_size
        sequence_length=self.X_train.shape[0]
        d_model=X_train.shape[1]
        transformer_input_layer = keras.Input(shape=(sequence_length, d_model))
        self.transformer_layer = transformer_input_layer

        # Reshape input data
        print("tuner: input X_train shape:", self.X_train.shape)
        print("tuner: input X_train shape:length:",  len(self.X_train.shape))
        
        print("The input shape required by the lstm model is:", self.lstm_shape)
        print("The input shape required by the cnn model is:", self.cnn_shape)
        print("The input shape required by the gru model is:", self.gru_shape)
        print("The input shape required by the transformer model is:", self.transformer_shape)
        print("The Incoming shape is:", self.X_train.shape)
        

        # Search for best hyperparameters
        tuner.search(self.X_train, self.y_train, validation_data=self.validation_data, 
                     epochs=self.epochs, batch_size=self.batch_size)

        # Get the best hyperparameters
        best_hp = tuner.get_best_hyperparameters(num_trials=self.max_trials)
        if not best_hp:
            raise ValueError("No hyperparameters found. Ensure that the tuner has completed trials.")

        best_hp = best_hp[0]
        print(f"Best hyperparameters: {best_hp.values}")

        # Return the best model
        best_models = tuner.get_best_models(num_models=1)
        if not best_models:
            raise ValueError("No models found. Ensure that the tuner has completed trials.")

        return best_models[0]