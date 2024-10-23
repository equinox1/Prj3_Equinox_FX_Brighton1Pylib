import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, GRU, Dropout, Concatenate, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
import keras_tuner as kt

import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


# Assuming your data is in a CSV file
file = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\Mql5Data\EURUSD_tickdata1.csv"

data = pd.read_csv(file)
print("Data Loaded: ", data.head(5))
data = data.rename(columns={'Date': 'time', 'Timestamp': 'time_msc', 'Bid Price': 'bid', 'Ask Price': 'ask', 'Last Price': 'close', 'Volume': 'volume'})
data = data[['time', 'bid', 'ask', 'close', 'time_msc', 'volume']]
print(tabulate(data.head(10), showindex=False, headers=data.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))

# Split into features and target
data['target'] = data['close'].shift(-60)
data.dropna(inplace=True)
X = data[['close']].values
y = data['target'].values

# Normalize your data if needed
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trim extra elements
X_scaled = X_scaled[:1048500]  # Trim to nearest number divisible by 60

# Reshape after trimming
X_train = X_scaled.reshape(-1, 60, 1)
# Reshape your data to match the model's input
#X_train = X_scaled.reshape(-1, 60, 1)  # Adjust dimensions as required
y_train = y[:len(X_train)]  # Ensure this matches your target variable's dimensions

# Split data into training and validation sets
split_index = int(0.8 * len(X_train))
X_val = X_train[split_index:]
y_val = y_train[split_index:]
X_train = X_train[:split_index]
y_train = y_train[:split_index]

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
    print("Model Parameters: inputs: ",inputs,"outputs: ",output)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='LOG')),
                  loss=MeanSquaredError(), 
                  metrics=[MeanAbsoluteError()])
    
    return model

tuner = kt.Hyperband(build_model,
                     objective='val_mean_absolute_error',
                     max_epochs=2,
                     factor=3,
                     directory='c:\\keras_tuner_dir',
                     project_name='forex_price_forecasting')

tuner.search_space_summary()

# Assuming you have your data loaded in X_train, y_train, X_val, and y_val
tuner.search(X_train, y_train,
             epochs=1,
             validation_data=(X_val, y_val),
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

best_model = tuner.get_best_models(num_models=1)[0]
best_params = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model.summary()