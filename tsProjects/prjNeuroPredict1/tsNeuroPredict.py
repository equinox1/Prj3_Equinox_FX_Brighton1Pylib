# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                                  www.equinox.com |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "www.equinox.com"
# property version   "1.01"
# Test mode to pass through Litmus test data
mp_test = True

# +-------------------------------------------------------------------
# Import standard Python packages
# +-------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keyring as kr
from tabulate import tabulate

# +-------------------------------------------------------------------
# Import MetaTrader 5 (MT5) and other necessary packages
# +-------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# Import custom modules for MT5 and AI-related functionality
from tsMqlConnect import CMqlinitdemo
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup
from tsMqlMLTune import CMdtuner

# +-------------------------------------------------------------------
# Import TensorFlow for machine learning
# +-------------------------------------------------------------------
import tensorflow as tf
tf.compat.v1.reset_default_graph()  # Ensure compatibility with TensorFlow v1 functions

# Check GPU availability and configure memory growth if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
            )
    except RuntimeError as e:
        print(e)

# +-------------------------------------------------------------------
# Start MetaTrader 5 (MQL) terminal login
# +-------------------------------------------------------------------
# Fetch credentials from keyring
cred = kr.get_credential("xercesdemo", "")
username = kr.get_password("xercesdemo", "username")
password = kr.get_password("xercesdemo", "password")
username = cred.username
password = cred.password
# Check if the credentials are fetched successfully
if not username or not password:
    raise ValueError("Username or password not found in keyring")

# Ensure username is a valid integer
try:
    MPLOGIN = int(username)
except ValueError:
    raise ValueError("Username is not a valid integer")

# Display the fetched credentials
print("username:", username)
print("password:", password)

# Parameters for connecting to MT5 terminal
MPPATH = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\mql5\brokers\icmarkets\terminal64.exe"
MPPASS = str(password)
MPSERVER = r"ICMarketsSC-Demo"
MPTIMEOUT = 60000
MPPORTABLE = True

# Initialize and login to the MT5 terminal
c1 = CMqlinitdemo(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
login_success = c1.run_mql_login(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
if not login_success:
    raise ConnectionError("Failed to login to MT5 terminal")

# +-------------------------------------------------------------------
# Import data from MQL
# +-------------------------------------------------------------------
# Set parameters for data extraction
mp_symbol_primary = "EURUSD"
mp_symbol_secondary = "GBPUSD"
mp_year = 2024
mp_month = 1
mp_day = 1
mp_timezone = 'etc/UTC'
mp_rows = 100000
mp_rowcount = 100
mp_command = mt5.COPY_TICKS_ALL
mp_dfName = "df_rates"
mv_manual = True
mp_path = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\Mql5Data"

lpfileid = "tickdata1"
mp_filename = f"{mp_symbol_primary}_{lpfileid}.csv"

# Set up dataset
d1 = CMqldatasetup()
mv_utc_from = d1.set_mql_timezone(mp_year, mp_month, mp_day, mp_timezone)
print(f"Timezone Set to: {mv_utc_from}")
print(f"mp_path Set to: {mp_path}")
print(f"mp_filename Set to: {mp_filename}")

# Load tick data from MQL
mv_ticks1 = d1.run_load_from_mql(mv_manual, mp_dfName, mv_utc_from, mp_symbol_primary, mp_rows, mp_rowcount, mp_command, mp_path, mp_filename)
if not isinstance(mv_ticks1, pd.DataFrame) or mv_ticks1.empty:
    raise ValueError("Failed to load tick data from MQL")

# +-------------------------------------------------------------------
# Prepare and process the data
# +-------------------------------------------------------------------
mv_ticks2 = mv_ticks1.copy()  # Copy the tick data for further processing

# Shift the data by a specified time interval (e.g., 60 seconds)
mp_seconds = 60
mp_unit = 's'

mv_ticks3 = d1.run_shift_data1(mv_ticks2, mp_seconds, mp_unit)  # Ensure the method name is correct
if not isinstance(mv_ticks3, pd.DataFrame) or mv_ticks3.empty:
    raise ValueError("Failed to shift tick data")

# Display the first few rows of the data for verification
print(tabulate(mv_ticks1.head(10), showindex=False, headers=mv_ticks1.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks2.head(10), showindex=False, headers=mv_ticks2.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks3.head(10), showindex=False, headers=mv_ticks3.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))

print("1:Start Shapes of DataFrames:")
print(f"mv_ticks1: {mv_ticks1.shape}")
print(f"mv_ticks2: {mv_ticks2.shape}")
print(f"mv_ticks3: {mv_ticks3.shape}")
print("1:End Shapes of DataFrames:")
# Check for the presence of a 'target' column
if 'target' in mv_ticks3.columns:
    print(f"mv_ticks3: target column found with length {len(mv_ticks3['target'])}")
else:
    print("mv_ticks3: 'target' column not found")

# +-------------------------------------------------------------------
# Split the data into training and test sets
# +-------------------------------------------------------------------
m1 = CMqlmlsetup()
mp_test_size = 0.2
mp_shuffle = False

mv_X_train, mv_X_test, mv_y_train, mv_y_test = m1.dl_split_data_sets(mv_ticks3, mp_test_size, mp_shuffle)
if mv_X_train.empty or mv_X_test.empty or mv_y_train.empty or mv_y_test.empty:
    raise ValueError("Failed to split data into training and test sets")

# Scale the training and test data
mv_X_train.head(5)

print("2:Start Shapes of split data DataFrames:")
print(f"mv_X_train: {mv_X_train.shape}")
print(f"mv_X_test: {mv_X_test.shape}")
print(f"mv_y_train: {mv_y_train.shape}")
print(f"mv_y_test: {mv_y_test.shape}")
print("2:End Shapes of split data DataFrames:")

mv_X_train_scaled = pd.DataFrame(m1.dl_train_model_scaled(mv_X_train))
mv_X_test_scaled = pd.DataFrame(m1.dl_test_model_scaled(mv_X_test))

print("3:Start Shapes of scaled split data DataFrames:")
print(f"mv_X_train_scaled shape:", mv_X_train_scaled.shape)
print(f"mv_X_train_scaled head:" ,mv_X_train_scaled.head(len(mv_X_train_scaled)))

print(f"mv_X_test_scaled:", mv_X_test_scaled.shape)
print(f"mv_X_test_scaled head:", mv_X_test_scaled.head(len(mv_X_test_scaled)))
print("3:End Shapes of scaled split data DataFrames:")

# Ensure the scaled data shapes match the original labels
mp_X_train_input_shape = mv_X_train_scaled.shape
mp_X_test_input_shape = mv_X_test_scaled.shape


# +-------------------------------------------------------------------
# Hyperparameter tuning and model setup
# +-------------------------------------------------------------------
# Define parameters for the model tuning process
mp_epochs = 1
mp_batch_size = 16
mp_objective = str('val_accuracy')
print("mp_objective:", mp_objective)
mp_max_trials = 10
mp_executions_per_trial = 1
mp_validation_split = 0.2
mp_arraysize = 1
mp_lstm_shape = None
mp_cnn_shape = None
mp_gru_shape = None
mp_transformer_shape = None
mp_channels=1

mp_modeldatapath = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\PythonLib\tsModelData"
mp_directory = f"{mp_modeldatapath}\\tshybrid_ensemble_tuning_prod"
mp_project_name = f"{mp_modeldatapath}\\tshybrid_ensemble_model_prod"

# Switch directories for testing if in test mode
if mp_test:
    mp_directory = f"{mp_modeldatapath}\\tshybrid_ensemble_tuning_test"
    mp_project_name = f"{mp_modeldatapath}\\tshybrid_ensemble_model_test"

# Run the tuner to find the best model configuration
print("Running tuner with mp_X_train_input_scaled input shape:",mv_X_train_scaled.shape)
print("Running tuner with mp_X_train_input_scaled scaled data: Rows:", mv_X_train_scaled.shape[0], "Columns:", mv_X_train_scaled.shape[1])

# Run the tuner to find the best model configuration
mt = CMdtuner(mp_X_train_input_shape,
               mv_X_train_scaled,
               mv_y_train,
               mp_objective,
               mp_max_trials,
               mp_executions_per_trial, 
               mp_directory, 
               mp_project_name,
               mp_validation_split, 
               mp_epochs,
               mp_batch_size,
               mp_arraysize,
               mp_lstm_shape,
               mp_cnn_shape,
               mp_gru_shape,
               mp_transformer_shape,
               mp_channels)

# Run the tuner to find the best model configuration
best_model = mt.run_tuner()
if best_model is None:
    raise ValueError("Failed to find the best model configuration")
"""
# Display the best model's summary
best_model.summary()

# +-------------------------------------------------------------------
# Train and evaluate the model
# +-------------------------------------------------------------------
# Train the model using the scaled data
mv_X_train_list = [mv_X_train_scaled] * 4  # Create 4 identical tensors for the ensemble
mv_model = best_model.fit(mv_X_train_list, mv_y_train, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)
if mv_model is None:
    raise ValueError("Failed to train the model")

# Reshape the test data to match the model input requirements
mv_X_test_list = [mv_X_test_scaled] * 4

# Make predictions using the trained model
predictions = pd.DataFrame(best_model.predict(mv_X_test_list))
if predictions.empty:
    raise ValueError("Failed to make predictions")
print("Predictions:", predictions.head(5))

# Evaluate model performance (accuracy, precision, recall, etc.)
accuracy, precision, recall, f1 = m1.model_performance(best_model, mv_X_test_list, mv_y_test)
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
"""