import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import keras_tuner as kt
import os

def build_model(hp):
    inputs = Input(shape=(60, 1))  # input shape as (None, 60, 1)
    
    # CNN branch
    x_cnn = Conv1D(filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32), 
                   kernel_size=hp.Int('cnn_kernel_size', min_value=2, max_value=5, step=1), 
                   activation='relu')(inputs)
    x_cnn = MaxPooling1D(pool_size=2)(x_cnn)
    x_cnn = Flatten()(x_cnn)
    
    # LSTM branch
    x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
    x_lstm = LSTM(units=hp.Int('lstm_units', min_value=32, max_value=128, step=32))(x_lstm)
    
    # GRU branch
    x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32), return_sequences=True)(inputs)
    x_gru = GRU(units=hp.Int('gru_units', min_value=32, max_value=128, step=32))(x_gru)
    
    # Transformer branch
    x_trans = MultiHeadAttention(num_heads=hp.Int('num_heads', min_value=2, max_value=4, step=1), 
                                 key_dim=hp.Int('key_dim', min_value=32, max_value=128, step=32))(inputs, inputs)
    x_trans = LayerNormalization(epsilon=1e-6)(x_trans)
    x_trans = GlobalAveragePooling1D()(x_trans)
    
    # Concatenate the outputs of the branches
    combined = Concatenate()([x_cnn, x_lstm, x_gru, x_trans])
    x = Dense(50, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss=MeanSquaredError(), 
                  metrics=[MeanAbsoluteError()])
    
    return model
