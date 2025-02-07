import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load OHLC data (assumes CSV with columns: time, open, high, low, close, volume)
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    return df

# Preprocess data
def preprocess_data(df, window_size=1440, future_gap=1440):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
    
    X, y = [], []
    for i in range(len(df_scaled) - window_size - future_gap):
        X.append(df_scaled[i:i + window_size])
        y.append(df_scaled[i + window_size + future_gap - 1, 3])  # Predict future close price
    
    return np.array(X), np.array(y), scaler

# Define LSTM model
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Load and process data
file_path = 'forex_data.csv'  # Update with actual file path
df = load_data(file_path)
X, y, scaler = preprocess_data(df)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train model
model = build_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('forex_lstm_model.h5')

# Predict function
def predict_future(model, data, scaler):
    prediction = model.predict(data)
    prediction = scaler.inverse_transform(
        np.hstack((np.zeros((len(prediction), 4)), prediction.reshape(-1, 1)))
    )[:, -1]  # Extract the close price
    return prediction

# Example prediction
predicted_prices = predict_future(model, X_test[-10:], scaler)
print(predicted_prices)
