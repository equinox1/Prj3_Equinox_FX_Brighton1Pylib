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
        super(kt.HyperModel, self).__init__()
        input_shape,
        lstm_shape,  # (timesteps, features)
        cnn_shape,    # (height, width, channels)
        gru_shape,    # (timesteps, features)
        transformer_shape,  # (sequence_length, d_model)
        objective,
        max_trials,
        executions_per_trial,
        directory,
        project_name,
        validation_split,
        epochs,
        batch_size,
        arraysize,
        channels

    def build(self, hp):
        inputs = Input(shape=self.input_shape)
        lstm_shape = Input(shape=self.lstm_shape)
        cnn_shape = Input(shape=self.cnn_shape)
        gru_shape = Input(shape=self.gru_shape)
        transformer_shape = Input(shape=self.transformer_shape)

        print("The build input shape required by the lstm model is:", lstm_shape)
        print("The build input shape required by the cnn model is:", cnn_shape)
        print("The build input shape required by the gru model is:", gru_shape)
        print("The build input shape required by the transformer model is:", transformer_shape)
        print("The build Incoming shape is:", inputs)

        # Base Models Definition
        # LSTM Model
        def build_lstm():
            x = LSTM(hp.Int('lstm_units', 32, 128, step=32), return_sequences=True)(lstm_shape)
            x = Flatten()(x)
            x = Dense(hp.Int('lstm_dense_units', 32, 128, step=32), activation='relu')(x)
            return x

        # 1D CNN Model
        def build_cnn():
            x = Conv1D(filters=hp.Int('cnn_filters', 32, 128, step=32), 
                       kernel_size=hp.Choice('cnn_kernel_size', [3, 5]), 
                       padding='same', activation='relu')(cnn_shape)
            x = MaxPooling1D(pool_size=2, padding='same')(x)
            x = Flatten()(x)
            x = Dense(hp.Int('cnn_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # GRU Model
        def build_gru():
            x = GRU(hp.Int('gru_units', 32, 128, step=32), return_sequences=True)(gru_shape)
            x = Flatten()(x)
            x = Dense(hp.Int('gru_dense_units', 32, 128, step=32), activation='sigmoid')(x)
            return x

        # Transformer Model
        def build_transformer():
            embed_dim = hp.Int('transformer_embed_dim', 32, 128, step=32)
            x = Dense(embed_dim, activation='relu')(transformer_shape)  # Project input to the required embedding dimension
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
        model = Model(inputs=[lstm_shape, cnn_shape, gru_shape, transformer_shape], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

# Class for running the tuner
class CMdtuner:
    def __init__(self, X_train_input_shape,X_train, y_train, lstm_shape, cnn_shape, gru_shape, transformer_shape,objective, max_trials, executions_per_trial, directory, project_name, tuner_params):
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
        self.tuner_params = tuner_params
            
    def run_tuner(self):
        tuner = kt.RandomSearch(
            HybridEnsembleHyperModel(
                input_shape=self.input_shape,
                lstm_shape=self.lstm_shape,  # (timesteps, features)
                cnn_shape=self.cnn_shape,    # (height, width, channels)
                gru_shape=self.gru_shape,    # (timesteps, features)
                transformer_shape=self.transformer_shape,  # (sequence_length, d_model)
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
            **self.tuner_params
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
                     epochs=self.tuner_params['epochs'], 
                     batch_size=self.tuner_params['batch_size'])

        # Get the best model
        best_model = tuner.get_best_models(num_models=1)[0]
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        print(f"Best Hyperparameters: {best_hp.values}")
        return best_model