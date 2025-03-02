import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

# Function to create sequences for time series forecasting
def create_sequences(data, target_col, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_col])  # Predicting Close Price
    return np.array(X), np.array(y)

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
    df = df[['Open', 'High', 'Low', 'Close']].values  # Extract OHLC columns
    return df

# Define model builder function
def model_builder(hp):
    model = keras.Sequential()
    model.add(layers.LSTM(hp.Int('units', min_value=32, max_value=256, step=32),
                           return_sequences=True, input_shape=(window_size, 4)))
    model.add(layers.LSTM(hp.Int('units', min_value=32, max_value=256, step=32)))
    model.add(layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mse', metrics=['mae'])
    return model

# Load data
data = load_data('ohlc_data.csv')  # Update with your data file
window_size = 1440  # 24 hours * 60 minutes
X, y = create_sequences(data, target_col=3, window_size=window_size)

# Split data
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Hyperparameter tuning
tuner = kt.RandomSearch(model_builder,
                         objective='val_loss',
                         max_trials=10,
                         executions_per_trial=2,
                         directory='tuner_dir',
                         project_name='ohlc_forecast')

tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32)

# Get best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)

# Train best model
best_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=32)

# Forecast next closing price
predicted_close = best_model.predict(X_test[-1].reshape(1, window_size, 4)) #Corrected reshape
print(f"Predicted close price: {predicted_close[0, 0]}")