# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"

####################################################################
# PARAMS
####################################################################
# Login
BROKER1 = "xerces_icm"
BROKER2 = "xerces_meta"
MPBROKPATH1 = r"Brokers/ICMarkets/terminal64.exe"
MPBROKPATH2 = r"Brokers/Metaquotes/terminal64.exe"
SERVER1 = "ICMarketsSC-Demo"
SERVER2 = "MetaQuotes-Demo"
# Parameters for connecting to MT5 terminal
MBROKER = BROKER2
MKBROKPATH=MPBROKPATH2
MSERVER = SERVER2

# Parameters for connecting to MT5 terminal
MPBASEPATH = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"
MPBROKPATH1 = MKBROKPATH
MPPATH = MPBASEPATH + MPBROKPATH
MPPASS = str(password)
MPSERVER = MSERVER
MPTIMEOUT = 60000
MPPORTABLE = True
#Test mode
mp_test = False
# Shift the data by a specified time interval (e.g., 60 seconds)
SECONDS = 1
MINUTES = 60
HOURS = 60 * MINUTES
DAYS = 24 * HOURS
WEEKS = 7 * DAYS
YEARS = 365 * DAYS
mp_datatype = 'M1' # Data type: M1, M5, M15, M30, H1, H4, D1, W1, MN1

mp_seconds = MINUTES # Shift the data by 60 second interval
mp_unit = 's' # Shift the data by 60 seconds

# Set parameters for data extraction
mp_symbol_primary = "EURUSD"
mp_symbol_secondary = "GBPUSD"
mp_year = 2024
mp_month = 1
mp_day = 1
mp_timezone = 'etc/UTC'
mp_rows = 100000
mp_rowcount = 100000

mp_dfName = "df_rates"
mv_manual = True
mp_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"

lpfileid = "tickdata1"
mp_filename = f"{mp_symbol_primary}_{lpfileid}.csv"

# Set parameters for the model
mp_param_steps = 1
mp_param_max_epochs=10
mp_param_min_epochs=1
mp_param_epochs = 200
mp_param_chk_patience = 3

mp_multiactivate=True  
####################################################################

# +-------------------------------------------------------------------
# Import standard Python packages
# +-------------------------------------------------------------------
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keyring as kr
from tabulate import tabulate
from datetime import datetime, date
# +-------------------------------------------------------------------
# Import MetaTrader 5 (MT5) and other necessary packages
# +-------------------------------------------------------------------
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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
import onnx
import tf2onnx
import onnxruntime
import onnxruntime.backend as backend
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer

tf.compat.v1.reset_default_graph()  # Ensure compatibility with TensorFlow v1 functions
print("Tensorflow Version", tf.__version__)

from tensorflow.keras.layers import Input
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
# Fetch credentials from keyring for metaquotes and xerces_meta


cred = kr.get_credential(MBROKER, "")
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


# Initialize and login to the MT5 terminal
c1 = CMqlinitdemo(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
login_success = c1.run_mql_login(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE)
if not login_success:
    raise ConnectionError("Failed to login to MT5 terminal")

terminal_info = mt5.terminal_info()
print(terminal_info)
file_path=terminal_info.data_path +r"/MQL5/Files/"
print(f"MQL file_path:" ,file_path)

#data_path to save model
mp_data_path=file_path
print(f"data_path to save onnx model: ",mp_data_path)
# +-------------------------------------------------------------------
# Import data from MQL
# +-------------------------------------------------------------------

# Set up dataset
d1 = CMqldatasetup()
mp_command = mt5.COPY_TICKS_ALL
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
#
# Select Model 
mp_cnn_model = True
mp_lstm_model = True   
mp_gru_model = True
mp_transformer_model = True
mp_run_single_input_model = True
mp_run_single_input_submodels = False # not implemented yet     

# define inputshapes
mp_single_input_shape = mp_X_train_input_shape[1],
mp_lstm_input_shape = mp_X_train_input_shape[1]
mp_cnn_input_shape = mp_X_train_input_shape[1]
mp_gru_input_shape = mp_X_train_input_shape[1]
mp_transformer_input_shape = mp_X_train_input_shape[1]

# define features
mp_null=None
mp_single_features = 1
mp_lstm_features = 1
mp_cnn_features = 1
mp_gru_features = 1
mp_transformer_features = 1

# Hypermodel parameters
mp_activation1= 'relu'     
mp_activation2 = 'linear'
mp_activation3 = 'softmax'
mp_activation4 = 'sigmoid'     
mp_Hypermodel = 'HyperModel'
mp_objective = 'val_loss'
mp_max_epochs = mp_param_max_epochs 
mp_factor = 10
mp_seed = 42
mp_hyperband_iterations = 1
mp_tune_new_entries = False
mp_allow_new_entries = False
mp_max_retries_per_trial = 1
mp_max_consecutive_failed_trials = 1
# base tuner parameters
mp_validation_split = 0.2
mp_epochs = mp_param_epochs 
mp_batch_size = 16   
mp_dropout = 0.2
mp_oracle = None
mp_hypermodel = None
mp_max_model_size = 1
mp_optimizer = 'adam'
mp_loss = 'mean_squared_error'
mp_metrics = ['mean_squared_error']
mp_distribution_strategy = None
mp_directory = None
mp_logger = None
mp_tuner_id = None
mp_overwrite = True
mp_executions_per_trial = 1
mp_chk_fullmodel = True


# Checkpoint parameters
mp_chk_verbosity = 1    # 0, 1mp_chk_mode = 'min' # 'min' or 'max'
mp_chk_mode = 'min' # 'min' or 'max'
mp_chk_monitor = 'val_loss' # 'val_loss' or 'val_mean_squared_error'
mp_chk_sav_freq = 'epoch' # 'epoch' or 'batch'
mp_chk_patience = mp_param_chk_patience

mp_modeldatapath = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsModelData"
mp_directory = f"tshybrid_ensemble_tuning_prod"
mp_project_name = "prjEquinox1_prod"
mp_today=date.today().strftime('%Y-%m-%d %H:%M:%S')
mp_random = np.random.randint(0, 1000)
print("mp_random:", mp_random)
print("mp_today:", mp_today)
mp_baseuniq=str(1) # str(mp_random)
mp_basepath = os.path.join(mp_modeldatapath, mp_directory,mp_baseuniq)
mp_checkpoint_filepath = os.path.join(mp_modeldatapath,mp_directory,mp_basepath, mp_project_name)
print("mp_checkpoint_filepath:", mp_checkpoint_filepath)
# Switch directories for testing if in test mode
if mp_test:
    mp_directory = f"tshybrid_ensemble_tuning_test"
    mp_project_name = "prjEquinox1_test"

# Run the tuner to find the best model configuration
print("Running tuner1 with mp_X_train_input_scaled input shape:", mv_X_train.shape)
print("Running tuner2 with mp_X_train_input_scaled scaled data: Rows:", mv_X_train.shape[0], "Columns:", mv_X_train.shape[1])
print("Running tuner3 with mp_X_train_input_scaled input shape:", mv_X_train.shape)
mp_inputs = Input(shape=(mv_X_train.shape[1],1) ) 
print("Running tuner4 with mp_X_train_input_scaled input shape:", mp_inputs)

# Create an instance of the tuner class
print("Creating an instance of the tuner class")
mt = CMdtuner(
    X_train=mv_X_train,
    y_train=mv_y_train,
    X_test=mv_X_test,
    y_test=mv_y_test,
    inputs=mp_inputs,
    cnn_model=mp_cnn_model,
    lstm_model=mp_lstm_model,
    gru_model=mp_gru_model,
    transformer_model=mp_transformer_model,
    run_single_input_model=mp_run_single_input_model,
    run_single_input_submodels=mp_run_single_input_submodels,
    objective=mp_objective,
    max_epochs=mp_param_max_epochs,
    min_epochs=mp_param_min_epochs,
    factor=mp_factor,
    seed=mp_seed,
    hyperband_iterations=mp_hyperband_iterations,
    tune_new_entries=mp_tune_new_entries,
    allow_new_entries=mp_allow_new_entries,
    max_retries_per_trial=mp_max_retries_per_trial,
    max_consecutive_failed_trials=mp_max_consecutive_failed_trials,
    validation_split=mp_validation_split,
    epochs=mp_epochs,
    batch_size=mp_batch_size,
    dropout=mp_dropout,
    oracle=mp_oracle,
    activation1=mp_activation1,
    activation2=mp_activation2,
    activation3=mp_activation3,
    activation4=mp_activation4,
    hypermodel=mp_hypermodel,
    max_model_size=mp_max_model_size,
    optimizer=mp_optimizer,
    loss=mp_loss,
    metrics=mp_metrics,
    distribution_strategy=mp_distribution_strategy,
    directory=mp_directory,
    basepath=mp_basepath,
    project_name=mp_project_name,
    logger=mp_logger,
    tuner_id=mp_tuner_id,
    overwrite=mp_overwrite,
    executions_per_trial=mp_executions_per_trial,
    chk_fullmodel=mp_chk_fullmodel,
    chk_verbosity=mp_chk_verbosity,
    chk_mode=mp_chk_mode,
    chk_monitor=mp_chk_monitor,
    chk_sav_freq=mp_chk_sav_freq,
    chk_patience=mp_chk_patience,
    checkpoint_filepath=mp_checkpoint_filepath,
    modeldatapath=mp_modeldatapath,
    step=mp_param_steps,
    multiactivate=mp_multiactivate
)


      
# Run the tuner to find the best model configuration
print("Running Main call to tuner")
best_model = mt.tuner.get_best_models()
best_params = mt.tuner.get_best_hyperparameters(num_trials=1)[0]
best_model[0].summary()

# +-------------------------------------------------------------------
# Scale the data
# +-------------------------------------------------------------------
scaler = StandardScaler()
mv_X_train = scaler.fit_transform(mv_X_train)
mv_X_test = scaler.transform(mv_X_test)

# +-------------------------------------------------------------------
# Train and evaluate the model
# +-------------------------------------------------------------------

best_model[0].fit(mv_X_train, mv_y_train, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)
best_model[0].evaluate(mv_X_test, mv_y_test)

# Assuming mv_X_train is your training data
scaler = StandardScaler()
scaler.fit(mv_X_train)  # Fit the scaler on your training data

# +-------------------------------------------------------------------
# Predict the test data using the trained model
# +-------------------------------------------------------------------
# Assuming mv_X_train had 60 features, and the target is one of them (let's say the last one)
scaler = StandardScaler()
scaler.fit(mv_X_train)  # Fit scaler on training data (number of seconds features)
predicted_fx_price = best_model[0].predict(mv_X_test)

# If predicted_fx_price is only 1D, reshape to 2D
predicted_fx_price = predicted_fx_price.reshape(-1, 1)

# Inverse transform only the target column
predicted_fx_price = scaler.inverse_transform(
    np.hstack([np.zeros((predicted_fx_price.shape[0], mv_X_train.shape[1] - 1)), predicted_fx_price])
)[:, -1]  # Extract only the target column after inverse transform


# Assuming mv_y_train is the target variable (FX prices) used during training
target_scaler = StandardScaler()

# Fit the scaler on the training target values (not the features)
target_scaler.fit(mv_y_train.values.reshape(-1, 1))

# Now inverse transform mv_y_test using the target-specific scaler
mv_y_test_reshaped = mv_y_test.values.reshape(-1, 1)  # Reshape to match the scaler's input shape
real_fx_price = target_scaler.inverse_transform(mv_y_test_reshaped)  # Inverse transform to get actual prices

#print(real_fx_price)

# Evaluation and visualization
#Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
# The lower the MSE, the better the model.

#Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values. 
# Like MSE, lower values indicate better model performance.

#R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
# dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a 
# perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the target
# variable. Negative values indicate poor model performance.

mse, mae, r2 = mean_squared_error(real_fx_price, predicted_fx_price), mean_absolute_error(real_fx_price, predicted_fx_price), r2_score(real_fx_price, predicted_fx_price)
print(f"Mean Squared Error: {mse}, Mean Absolute Error: {mae}, R² Score: {r2}")

plt.plot(real_fx_price, color='red', label='Real FX Price')
plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
plt.title('FX Price Prediction')
plt.xlabel('Time')
plt.ylabel('FX Price')
plt.legend()
plt.savefig(mp_basepath + '//' + 'plot.png')


# +-------------------------------------------------------------------
# Save model to ONNX
# +-------------------------------------------------------------------
# Save the model to ONNX format

mp_output_path = mp_data_path + "model_" + mp_symbol_primary + "_" + mp_datatype + "_" + str(mp_seconds) + ".onnx"
print(f"output_path: ",mp_output_path)

# Assuming your model has a single input

# Convert the model
print("mp_inputs: ",mp_inputs)
spec = mp_inputs.shape
spec = (tf.TensorSpec(spec, tf.float32, name="input"),)
print("spec: ",spec)
# Convert the model to ONNX format
opver = 17
onnx_model = tf2onnx.convert.from_keras(best_model[0], input_signature=spec, output_path= mp_output_path, opset=opver)
print("ONNX Runtime version:", ort.__version__)
print(f"model saved to ", mp_output_path)

from onnx import checker 
checker.check_model(best_model[0])
# finish
mt5.shutdown()
plt.show()