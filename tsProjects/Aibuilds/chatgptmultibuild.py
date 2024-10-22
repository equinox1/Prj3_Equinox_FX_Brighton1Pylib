import tensorflow as tf
from tensorflow.keras import layers, models
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

# Define model building function
def build_model(hp):
    inputs = Input(shape=(60, 1))

    # CNN Layer
    x = Conv1D(filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
               kernel_size=hp.Choice('conv_kernel_size', values=[3, 5, 7]),
               activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)

    # LSTM or GRU layer (Choice between LSTM and GRU)
    for i in range(hp.Int('rnn_layers', 1, 3)):  # Add multiple RNN layers if needed
        rnn_type = hp.Choice('rnn_type', ['LSTM', 'GRU'])
        if rnn_type == 'LSTM':
            x = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
                     return_sequences=True if i < hp.Int('rnn_layers', 1, 3) - 1 else False)(x)
        else:
            x = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32),
                    return_sequences=True if i < hp.Int('rnn_layers', 1, 3) - 1 else False)(x)
        x = Dropout(rate=hp.Float('dropout_rnn_rate', min_value=0.1, max_value=0.5, step=0.1))(x)

    # Transformer Layer (Optional Transformer-based attention mechanism)
    if hp.Boolean('use_transformer'):
        x = LayerNormalization(epsilon=1e-6)(x)
        attn_output = layers.MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=2),
                                                key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(x, x)
        # Add residual connection and layer normalization
        x = layers.Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Check if the sequence dimension is still present
        if len(x.shape) == 3:  # If x is 3D, we can apply GlobalAveragePooling1D
            x = layers.GlobalAveragePooling1D()(x)
        else:  # If x is 2D, apply Flatten to make it compatible with Dense layers
            x = Flatten()(x)
    else:
        # Use Flatten when not using Transformer
        x = Flatten()(x)

    # Dense output layer for regression
    # Softmax activation with the correct axis
    output = layers.Softmax(axis=-1)(output_tensor)


    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error',
        metrics=['mae'])

    return model

# Create a tuner with RandomSearch
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=20,  # You can increase this for more thorough search
    executions_per_trial=2,  # Average over multiple runs
    directory='forex_tuning',
    project_name='cnn_lstm_gru_transformer_forex')

# Sample synthetic data (time series of 60 steps, 1 feature)
import numpy as np
X_train = np.random.rand(1000, 60, 1)
y_train = np.random.rand(1000, 1)
X_val = np.random.rand(200, 60, 1)
y_val = np.random.rand(200, 1)

# Search for the best model
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
