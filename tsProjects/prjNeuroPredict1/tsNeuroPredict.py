# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                                  www.equinox.com |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "www.equinox.com"
# property version   "1.01"
# Test mode to pass through Litmus test data
mp_test = False

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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
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
print("Tensorflow Version", tf.__version__)
# Check GPU availability and configure memory growth if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# +-------------------------------------------------------------------
# Start MetaTrader 5 (MQL) terminal login
# +-------------------------------------------------------------------
# Fetch credentials from keyring
cred = kr.get_credential("xercesdemo", "")
if cred is None:
    raise ValueError("Credentials not found in keyring")

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
mp_rows = 10000
mp_rowcount = 10000
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

mv_X_ticks3, mv_y_ticks3 = d1.run_shift_data2(mv_ticks2, mp_seconds, mp_unit)  # Ensure the method name is correct
if mv_X_ticks3 is None or mv_y_ticks3 is None:
    raise ValueError("Shifted data is invalid. Check the `run_shift_data2` method.")

# Check if the data is a pandas DataFrame or a NumPy array
df = mv_X_ticks3
# Check if df is a pandas DataFrame
if isinstance(df, pd.DataFrame):
    print("df is a pandas DataFrame")
# Check if df is a NumPy array
elif isinstance(df, np.ndarray):
    df = pd.DataFrame(df[:, :, 0])  # Assuming the first element is required
    mv_X_ticks3 = pd.DataFrame(df)
else:
    print("df is neither a pandas DataFrame nor a NumPy array")

mv_X_ticks3 = pd.DataFrame(mv_X_ticks3)
mv_y_ticks3 = pd.DataFrame(mv_y_ticks3)

# Display the first few rows of the data for verification
print(tabulate(mv_ticks1.head(10), showindex=False, headers=mv_ticks1.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_ticks2.head(10), showindex=False, headers=mv_ticks2.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_X_ticks3.head(10), showindex=False, headers=mv_X_ticks3.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print(tabulate(mv_y_ticks3.head(10), showindex=False, headers=mv_y_ticks3.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))


print("1:Start Shapes of DataFrames:")
print(f"mv_ticks1: {mv_ticks1.shape}")
print(f"mv_ticks2: {mv_ticks2.shape}")
print(f"mv_ticks3: {mv_X_ticks3.shape}")
print(f"mv_ticks3: {mv_y_ticks3.shape}")
print("1:End Shapes of DataFrames:")


# +-------------------------------------------------------------------
# Split the data into training and test sets
# +-------------------------------------------------------------------
m1 = CMqlmlsetup()
mp_train_split = 0.8
mp_test_split = 0.2
mp_shuffle = False

mv_X_train, mv_X_test, mv_y_train, mv_y_test = m1.dl_split_data_sets(mv_X_ticks3, mv_y_ticks3, mp_train_split, mp_test_split, mp_shuffle)
if mv_X_train.empty or mv_X_test.empty or mv_y_train.empty or mv_y_test.empty:
    raise ValueError("Failed to split data into training and test sets")


print("2:Start Shapes of split data DataFrames:")
print(f"mv_X_train shape: {mv_X_train.shape}")
print(f"mv_X_test shape: {mv_X_test.shape}")
print(f"mv_y_train shape: {mv_y_train.shape}")
print(f"mv_y_test shape: {mv_y_test.shape}")
print("2:End Shapes of split data DataFrames:")


# define the input shape for the model
mp_X_train_input_shape = mv_X_train.shape
mp_X_test_input_shape = mv_X_test.shape
mp_y_train_input_shape = mv_y_train.shape
mp_y_test_input_shape = mv_y_test.shape

# +-------------------------------------------------------------------
# Hyperparameter tuning and model setup
# +-------------------------------------------------------------------
# Define parameters for the model tuning process
# Data sources
mv_X_train = mv_X_train
#Shaping the data
#=====================
mv_X_train  = mv_X_train[:1048500]  # Trim to nearest number divisible by 60
mv_X_train_numpy = mv_X_train.values
# Reshape the array (-1, 60, 1)
reshaped_array = mv_X_train_numpy.reshape(-1, 60, 1)
# If you want to convert it back to a DataFrame:
reshaped_df = pd.DataFrame(reshaped_array.reshape(-1, reshaped_array.shape[1]))
mv_X_train = reshaped_df
# End of reshaping
#=====================

#=======================
mv_y_train = mv_y_train
# Select Model 
mp_cnn_model = True
mp_lstm_model = True   
mp_gru_model = True
mp_transformer_model = True
mp_run_single_input_model = True
mp_run_single_input_submodels = False      
# define inputshapes
mp_lstm_input_shape = mp_X_train_input_shape
mp_cnn_input_shape = mp_X_train_input_shape
mp_gru_input_shape = mp_X_train_input_shape
mp_transformer_input_shape = mp_X_train_input_shape
mp_lstm_features = 1
mp_cnn_features = 1
mp_gru_features = 1
mp_transformer_features = 1
# Hypermodel parameters
mp_Hypermodel = 'HyperModel'
mp_objective = 'val_loss'
mp_max_epochs = 1
mp_factor = 3
mp_seed = 42
mp_hyperband_iterations = 1
mp_tune_new_entries = False
mp_allow_new_entries = False
mp_max_retries_per_trial = 1
mp_max_consecutive_failed_trials = 1
# base tuner parameters
mp_validation_split = 0.2
mp_epochs = 1
mp_batch_size = 16   
mp_dropout = 0.2
mp_oracle = None
mp_hypermodel = None
mp_max_model_size = 1
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
mp_metrics = ['mean_squared_error']
mp_distribution_strategy = None


mp_logger = None
mp_tuner_id = None
mp_overwrite = False
mp_executions_per_trial = 1
mp_chk_fullmodel = True

# Checkpoint parameters
mp_chk_verbosity = 1    # 0, 1mp_chk_mode = 'min' # 'min' or 'max'
mp_chk_mode = 'min' # 'min' or 'max'
mp_chk_monitor = 'val_loss' # 'val_loss' or 'val_mean_squared_error'
mp_chk_sav_freq = 'epoch' # 'epoch' or 'batch'

mp_modeldatapath = r"c:\users\shepa\onedrive\8.0 projects\8.3 projectmodelsequinox\equinrun\PythonLib\tsModelData"
mp_directory = f"tshybrid_ensemble_tuning_prod"
mp_project_name = "prjEquinox1_prod"
mp_checkpoint_filepath = os.path.join(mp_modeldatapath,mp_directory, mp_project_name)
print("mp_checkpoint_filepath:", mp_checkpoint_filepath)
# Switch directories for testing if in test mode
if mp_test:
    mp_directory = f"{mp_modeldatapath}\\tshybrid_ensemble_tuning_test"
    mp_project_name = f"\\tshybrid_ensemble_model_test"

# Run the tuner to find the best model configuration
print("Running tuner with mp_X_train_input_scaled input shape:", mv_X_train.shape)
print("Running tuner with mp_X_train_input_scaled scaled data: Rows:", mv_X_train.shape[0], "Columns:", mv_X_train.shape[1])


shapes_and_features = {
    'cnn': (mp_cnn_input_shape, mp_cnn_features),
    'lstm': (mp_lstm_input_shape, mp_lstm_features),
    'gru': (mp_gru_input_shape, mp_gru_features),
    'transformer': (mp_transformer_input_shape, mp_transformer_features),
}

# Create an instance of the tuner class
print("Creating an instance of the tuner class")

mt = CMdtuner(mv_X_train,
              mv_y_train,
              mv_X_test,
              mv_y_test,
              mp_cnn_model,
              mp_lstm_model,
              mp_gru_model,
              mp_transformer_model,
              mp_run_single_input_model,
              mp_run_single_input_submodels,
              shapes_and_features, 
              mp_objective,
              mp_max_epochs,
              mp_factor,
              mp_seed,
              mp_hyperband_iterations,
              mp_tune_new_entries,
              mp_allow_new_entries,
              mp_max_retries_per_trial,
              mp_max_consecutive_failed_trials,
              mp_validation_split, 
              mp_epochs,
              mp_batch_size,    
              mp_dropout,
              mp_oracle,
              mp_hypermodel,
              mp_max_model_size,
              mp_optimizer,
              mp_loss,
              mp_metrics,
              mp_distribution_strategy,
              mp_directory, 
              mp_project_name,
              mp_logger,
              mp_tuner_id,
              mp_overwrite,
              mp_executions_per_trial,
              mp_chk_fullmodel,
              mp_chk_verbosity,
              mp_chk_mode,
              mp_chk_monitor,
              mp_chk_sav_freq,
              mp_checkpoint_filepath,
              mp_modeldatapath
        )
      
# Run the tuner to find the best model configuration
print("Running tuner")
best_model = mt.tuner.get_best_models()
best_params = mt.tuner.get_best_hyperparameters(num_trials=1)[0]
best_model[0].summary()
 
# Display the best model's summary

"""
# +-------------------------------------------------------------------
# Train and evaluate the model
# +-------------------------------------------------------------------
# Train the model using the scaled data
# Fit the scaler to the training data
# Assuming mv_y_train is a DataFrame
mv_y_train_reshaped = mv_y_train.values.reshape(-1, 1)  # Convert to NumPy array and reshape
scaler.fit(mv_y_train_reshaped)  # Use the reshaped array for fitting
mv_model = best_model.fit(mv_X_train, mv_y_train_reshaped, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)

# +-------------------------------------------------------------------
# Predict the test data using the trained model
# +-------------------------------------------------------------------

# Making predictions
predicted_fx_price = best_model.predict(mv_X_test)
predicted_fx_price = scaler.inverse_transform(predicted_fx_price)  # No need to reshape

# Inverse transform to get actual prices
mv_y_test_reshaped = mv_y_test.values.reshape(-1, 1)  # Convert to NumPy array and reshape
real_fx_price = scaler.inverse_transform(mv_y_test_reshaped)

# Visualizing the results
plt.plot(real_fx_price, color='red', label='Real FX Price')
plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
plt.title('FX Price Prediction')
plt.xlabel('Time')
plt.ylabel('FX Price')
plt.legend()
# Save the plot to a file


plt.savefig(mp_directory + '\\' + 'plot.png')
#plt.show()dir

# Evaluate model performance (accuracy, precision, recall, etc.)
# Uncomment and adjust model evaluation metrics for regression

#mse = m1.model_performance(best_model, mv_X_test, mv_y_test)
#print(f"Mean Squared Error: {mse}")
"""