# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +-------------------------------------------------------------------
# Import standard Python packages
# +-------------------------------------------------------------------
import os
import pathlib
from pathlib import Path, PurePosixPath
import posixpath
import sys
import time
import json
import keyring as kr
from tabulate import tabulate
from datetime import datetime, date
import pytz
import matplotlib.pyplot as plt
import seaborn as sns
import logging
# Import MetaTrader 5 (MT5) and other necessary packages
import MetaTrader5 as mt5
# import python ML packages
import numpy as np
from sklearn.preprocessing import MinMaxScaler ,StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Import dataclasses for data manipulation
import pandas as pd
from dataclasses import dataclass
# Import Temporian
# import temporian as tp # under dev windows only
# Import equinox functionality
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup, CMqlWindowGenerator
from tsMqlMLTune import CMdtuner
from tsMqlReference import CMqlTimeConfig
# Import TensorFlow for machine learning
import tensorflow as tf
import onnx
import tf2onnx
import onnxruntime
import onnxruntime.backend as backend
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import warnings
# set values for libs
warnings.filterwarnings("ignore")
scaler = StandardScaler()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
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

#################################################################
# Start of the main script
#################################################################
#Variable input
loadtensor = True
loadtemporian = False
broker = "METAQUOTES" # "ICM" or "METAQUOTES"
mp_test = False
show_dtype = False
mp_dfName1 = "df_rates1"
mp_dfName2 = "df_rates2"
mv_loadapiticks = True
mv_loadapirates = True
mv_loadfileticks = True
mv_loadfilerates = True
mv_usedata = 'loadfileticks' # 'loadapiticks' or 'loadapirates'or loadfileticks or loadfilerates

mp_rows = 1000
mp_rowcount = 10000
MPDATAFILE1 =  "tickdata1.csv"
MPDATAFILE2 =  "ratesdata1.csv"

#Set time constants
config = CMqlTimeConfig()
constants = config.get_constants()
print(constants)
# Set the parameters for data import
mp_history_size = 5 # Number of years of data to fetch

mp_symbol_primary = constants['SYMBOLS'][0]
print("1:mp_symbol_primary: ", mp_symbol_primary)

mp_symbol_secondary = constants['SYMBOLS'][1]
print("2:mp_symbol_secondary: ", mp_symbol_secondary)

mp_shiftvalue = constants['DATATYPE'][1]
print("1:mp_shiftvalue: ", mp_shiftvalue)

mp_unit = constants['UNIT'][0]
print("1:mp_unit: ", mp_unit)

mp_seconds = constants['TIMEVALUE']['SECONDS']
print("1:mp_seconds: ", mp_seconds)

mp_year = datetime.now().year
print("1:mp_year: ", mp_year)

mp_day = datetime.now().day
print("1:mp_day: ", mp_day)

mp_month = datetime.now().month
print("1:mp_month: ", mp_month)

mp_timezone = constants['TIMEZONES'][0]
print("1:mp_timezone: ", mp_timezone)

mp_timeframe = constants['TIMEFRAME'][5]
print("1:mp_timeframe: ", mp_timeframe)
print("1:mp_timeframe: ",mp_timeframe)

# Set parameters for the Tensorflow model
mp_param_steps = 1
mp_param_max_epochs=10
mp_param_min_epochs=1
mp_param_epochs = 200
mp_param_chk_patience = 3
mp_multiactivate=True  
# +-------------------------------------------------------------------
# Start MetaTrader 5 (MQL) terminal login
# +-------------------------------------------------------------------
c0 = CMqlBrokerConfig(broker, mp_symbol_primary, MPDATAFILE1, MPDATAFILE2)
broker_config = c0.set_mql_broker()

BROKER = broker_config['BROKER']
MPPATH = broker_config['MPPATH']
MPBASEPATH = broker_config['MPBASEPATH']
MPDATAPATH = broker_config['MPDATAPATH']
MPFILEVALUE1 = broker_config['MPFILEVALUE1']
MPFILEVALUE2 = broker_config['MPFILEVALUE2']
MKFILES = broker_config['MKFILES']
MPSERVER = broker_config['MPSERVER']
MPTIMEOUT = broker_config['MPTIMEOUT']
MPPORTABLE = broker_config['MPPORTABLE']
MPENV = broker_config['MPENV']
print(f"Broker: {BROKER}")

# Fetch credentials from keyring for metaquotes and xerces_meta
cred = kr.get_credential(BROKER, "")
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
MPPASS = str(password)
print(f"Login: {MPLOGIN}")

# Initialize and login to the MT5 terminal
c1 = CMqlinit(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV)
login_success = c1.run_mql_login(MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV)
if not login_success:
    raise ConnectionError("Failed to login to MT5 terminal")

terminal_info = mt5.terminal_info()
print(terminal_info)

# Set the data path for MQL files
MQLFILES=r"/MQL5/Files/"
mql_file_path=os.path.join(MPBASEPATH,MKFILES)

print(f"MQL file_path:" ,mql_file_path)

#data_path to save models
mp_data_path=mql_file_path
print(f"data_path to save onnx model: ",mp_data_path)
# +-------------------------------------------------------------------
# Import data from MQL
# +-------------------------------------------------------------------

# Set up dataset
d1 = CMqldatasetup()

mp_command = mt5.COPY_TICKS_ALL

mv_utc_from = d1.set_mql_timezone(mp_year-mp_history_size, mp_month, mp_day, mp_timezone)
mv_utc_to = d1.set_mql_timezone(mp_year, mp_month, mp_day, mp_timezone)

print("mv_utc_from set to : ",mv_utc_from, "mv_utc_to set to : ",mv_utc_to)
mp_path=MPDATAPATH
mp_filename1=MPFILEVALUE1
mp_filename2=MPFILEVALUE2
print(f"Year: {mp_year}, Month: {mp_month}, Day: {mp_day}")
print(f"Timezone Set to: {mv_utc_from}")
print(f"mp_path Set to: {MPDATAPATH}")
print(f"mp_filename1 Set to: {MPFILEVALUE1}")
print(f"mp_filename2 Set to: {MPFILEVALUE2}")

# Load tick data from MQL
if loadtensor:
    mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = d1.run_load_from_mql(mv_loadapiticks, mv_loadapirates, mv_loadfileticks, mv_loadfilerates, mp_dfName1, mp_dfName2, mv_utc_from, mp_symbol_primary, mp_rows, mp_rowcount, mp_command, mp_path, mp_filename1, mp_filename2, mp_timeframe)
    mv_tdata1apiticks=d1.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False,mp_dropna=False,mp_merge=False,mp_convert=False)
    mv_tdata1apirates=d1.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False,  filter_dtmi=False, filter_dtmf=False,mp_dropna=False,mp_merge=False,mp_convert=False)
    mv_tdata1loadticks=d1.wrangle_time(mv_tdata1loadticks, mp_unit,mp_filesrc= "ticks2", filter_int=False, filter_flt=False, filter_obj=False,  filter_dtmi=False, filter_dtmf=False,mp_dropna=False,mp_merge=True,mp_convert=True)
    mv_tdata1loadrates=d1.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False,  filter_dtmi=False, filter_dtmf=False,mp_dropna=False,mp_merge=False,mp_convert=False)
elif loadtemporian:
    mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = d1.temporian_load_from_mql(mv_loadapiticks, mv_loadapirates, mv_loadfileticks, mv_loadfilerates, mp_dfName1, mp_dfName2, mv_utc_from, mp_symbol_primary, mp_rows, mp_rowcount, mp_command, mp_path, mp_filename1, mp_filename2, mp_timeframe)

if show_dtype:
    print("1: dtypes of the dataframes")
    print(mv_tdata1apiticks.dtypes)  # Check the data types of the columns
    print("2: dtypes of the dataframes")
    print(mv_tdata1apirates.dtypes)  # Check the data types of the columns
    print("3: dtypes of the dataframes")
    print(mv_tdata1loadticks.dtypes)  # Check the data types of the columns
    print("4: dtypes of the dataframes")
    print(mv_tdata1loadrates.dtypes)  # Check the data types of the columns

mv_tdata1apiticks = d1.create_target(
    df=mv_tdata1apiticks,
    lookahead_seconds=mp_seconds,
    bid_column='T1_Bid_Price',
    ask_column='T1_Ask_Price',
    column_in='T1_Bid_Price',
    column_out1='close',
    column_out2='target',
    run_mode=1
)


mv_tdata1apirates = d1.create_target(
    df=mv_tdata1apirates,
    lookahead_seconds=mp_seconds,
    bid_column='R1_Bid_Price',
    ask_column='R1_Ask_Price',
    column_in='R1_Close',
    column_out1='close',
    column_out2='target',
    run_mode=2
)

mv_tdata1loadticks = d1.create_target(
    df=mv_tdata1loadticks,
    lookahead_seconds=mp_seconds,
    bid_column='T2_Bid_Price',
    ask_column='T2_Ask_Price',
    column_in='T2_Bid_Price',
    column_out1='close',
    column_out2='target',
    run_mode=3
)

mv_tdata1loadrates = d1.create_target(
    df=mv_tdata1loadrates,
    lookahead_seconds=mp_seconds,
    bid_column='R2_Bid_Price',
    ask_column='R2_Ask_Price',
    column_in='R2_Close',
    column_out1='close',
    column_out2='target',
    run_mode=4
)


# Display the first few rows of the data for verification
hrows=10
print("1:Start First few rows of the API Tick data:Count",len(mv_tdata1apiticks))
print(tabulate(mv_tdata1apiticks.head(hrows), showindex=False, headers=mv_tdata1apiticks.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print("2:Start First few rows of the API Rates data:Count",len(mv_tdata1apirates))
print(tabulate(mv_tdata1apirates.head(hrows), showindex=False, headers=mv_tdata1apirates.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print("3: Start First few rows of the FILE Tick data:Count",len(mv_tdata1loadticks))
print(tabulate(mv_tdata1loadticks.head(hrows), showindex=False, headers=mv_tdata1loadticks.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))
print("4: Start First few rows of the FILE Rates data:Count",len(mv_tdata1loadrates))
print(tabulate(mv_tdata1loadrates.head(hrows), showindex=False, headers=mv_tdata1loadrates.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))


# +-------------------------------------------------------------------
# Prepare and process the data
# +-------------------------------------------------------------------
mv_X_tdata2a = mv_tdata1apiticks.copy()  # Copy the data for further processing
mv_y_tdata2a = mv_tdata1apiticks.copy()  # Copy the data for further processing

mv_X_tdata2b = mv_tdata1apirates.copy()  # Copy the data for further processing
mv_y_tdata2b = mv_tdata1apirates.copy()  # Copy the data for further processing

mv_X_tdata2c = mv_tdata1loadticks.copy()  # Copy the data for further processing
mv_y_tdata2c = mv_tdata1loadticks.copy()  # Copy the data for further processing

mv_X_tdata2d = mv_tdata1loadrates.copy()  # Copy the data for further processing
mv_y_tdata2d = mv_tdata1loadrates.copy()  # Copy the data for further processing

# Check the switch of which file to use
if mv_usedata == 'loadapiticks':
    print("Using API Tick data")
    mv_X_tdata2 = mv_X_tdata2a
    mv_y_tdata2 = mv_y_tdata2a

elif mv_usedata == 'loadapirates':
    print("Using API Rates data")
    mv_X_tdata2 = mv_X_tdata2b
    mv_y_tdata2 = mv_y_tdata2b

elif mv_usedata == 'loadfileticks':
    print("Using File Tick data")
    mv_X_tdata2 = mv_X_tdata2c
    mv_y_tdata2 = mv_y_tdata2c

elif mv_usedata == 'loadfilerates':
    print("Using File Rates data")
    mv_X_tdata2 = mv_X_tdata2d
    mv_y_tdata2 = mv_y_tdata2d

# Ensure mv_X_tdata2 and mv_y_tdata2 are set correctly
if mv_X_tdata2 is None or mv_y_tdata2 is None:
    raise ValueError("Invalid mv_usedata value. Data not loaded properly.")



# +-------------------------------------------------------------------
# Split the data into training and test sets
# +-------------------------------------------------------------------
m1 = CMqlmlsetup()
mp_train_split = 0.7
mp_validation_split = 0.2
mp_test_split = 0.1
mp_gap=1-mp_test_split
mp_shuffle = False

# Split the data into training and test sets
n = len(mv_X_tdata2)
train_df = mv_X_tdata2[0:int(n*mp_train_split)]
val_df = mv_X_tdata2[int(n*mp_train_split):int(n*mp_gap)]
test_df = mv_X_tdata2[int(n*mp_gap):]


print("train",train_df.head(5))
print("val",val_df.head(5))
print("test",test_df.head(5))

train_df = pd.DataFrame(train_df)
val_df = pd.DataFrame(val_df)
test_df = pd.DataFrame(test_df)
# +-------------------------------------------------------------------
# End Split the data into training and test sets
# +-------------------------------------------------------------------

# +-------------------------------------------------------------------
# Normalize the data
# +-------------------------------------------------------------------
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std
# +-------------------------------------------------------------------
# End Normalize the data
# +-------------------------------------------------------------------

# +-------------------------------------------------------------------
# Establish Windows for the data
# +-------------------------------------------------------------------

mp_past_inputwidth_timewindow = 24  # 24 hours of history data INPUT WIDTH
mp_future_offsetwidth_timewindow = 24 # one LABEL=1  prediction 24 hours in the future Offset 24 hours
print("mp_past_inputwidth_timewindow:",mp_past_inputwidth_timewindow, "mp_future_offsetwidth_timewindow:",mp_future_offsetwidth_timewindow)

# Ensure mp_feature_columns is defined and is a list
# 24 hours of history data to forecast 24 hours in the future for 1 label 
mp_feature_columns = ['close']  # Example column names for feature independent variables
mp_label_columns = ['target']   # Example column names for label dependent variables
mp_num_features = len(mp_feature_columns) # Number of features
mp_num_labels = len(mp_label_columns) # Number of labels

print("Window Paramas: input_width:",mp_past_inputwidth_timewindow, "label_width:",mp_num_labels, "shift:",mp_future_offsetwidth_timewindow, "label_columns:",mp_feature_columns)
w1 = CMqlWindowGenerator(
    input_width=mp_past_inputwidth_timewindow,
    label_width=mp_num_labels,
    shift=mp_future_offsetwidth_timewindow,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    label_columns=mp_feature_columns
)

print(w1)
print("w1.total_window_size: ",w1.total_window_size)
# +-------------------------------------------------------------------
# End  Establish Windows for the data
# +-------------------------------------------------------------------


# +-------------------------------------------------------------------
# Split the data into windows split into inputs and labels
# +-------------------------------------------------------------------

w1win ,w1win_inputs,w1win_labels = w1.window_slicer( train_df, window_size=w1.total_window_size, shift_size=100)

print('train All shapes are: (batch, time, features)')
print(f'1:Window shape: {w1win.shape}')
print(f'2:Inputs shape: {w1win_inputs.shape}')
print(f'3:Labels shape: {w1win_labels.shape}')

w2win ,w2win_inputs,w2win_labels = w1.window_slicer( val_df, window_size=w1.total_window_size, shift_size=100)
print('val All shapes are: (batch, time, features)')
print(f'1:Window shape: {w2win.shape}')
print(f'2:Inputs shape: {w2win_inputs.shape}')
print(f'3:Labels shape: {w2win_labels.shape}')

w3win ,w3win_inputs,w3win_labels = w1.window_slicer( test_df, window_size=w1.total_window_size, shift_size=100)
print('test All shapes are: (batch, time, features)')
print(f'1:Window shape: {w3win.shape}')
print(f'2:Inputs shape: {w3win_inputs.shape}')
print(f'3:Labels shape: {w3win_labels.shape}')


# +-------------------------------------------------------------------
# End Split the data into windows split into inputs and labels
# +-------------------------------------------------------------------

# +-------------------------------------------------------------------
# TF datasets
# +-------------------------------------------------------------------
train = w1.train
val = w1.val
test = w1.test
for w1win_inputs, w1win_labels in train.take(1):
  print(f'1:Inputs shape (batch, time, features): {w1win_inputs.shape}')
  print(f'2:Labels shape (batch, time, features): {w1win_labels.shape}')

w1.w1win = w1win_inputs, w1win_labels
w1.plot(plot_col='close', model=None, max_subplots=3)


# +-------------------------------------------------------------------
# Hyperparameter tuning and model setup
# +-------------------------------------------------------------------
# Define parameters for the model tuning process
#
# Select Model 
mp_cnn_model = True
mp_lstm_model = True
plot = w1.plot(plot_col='close', model=None, max_subplots=3)
mp_gru_model = True
mp_transformer_model = True
mp_run_single_input_model = True
mp_run_single_input_submodels = False # not implemented yet     

# define inputshapes
print("train.shape[1]:",w1win.shape[1])
mp_single_input_shape = w1win.shape[1]
mp_lstm_input_shape = w1win.shape[1]
mp_cnn_input_shape = w1win.shape[1]
mp_gru_input_shape = w1win.shape[1]
mp_transformer_input_shape = w1win.shape[1]

# define features
mp_null = None
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
mp_max_retries_per_trial = 5
mp_max_consecutive_failed_trials = 6
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

mp_checkpoint_filepath = posixpath.join(mp_modeldatapath, mp_directory, mp_project_name)
print("mp_checkpoint_filepath:", mp_checkpoint_filepath)
# Switch directories for testing if in test mode
if mp_test:
    mp_directory = f"tshybrid_ensemble_tuning_test"
    mp_project_name = "prjEquinox1_test"

# Run the tuner to find the best model configuration
print("Running tuner1 with mp_X_train_input_scaled input shape:", w1win.shape)
print("Running tuner2 with mp_X_train_input_scaled scaled data: Rows:", w1win.shape[0], "Columns:", w1win.shape[1])
print("Running tuner3 with mp_X_train_input_scaled input shape:", w1win.shape)
mp_inputs = Input(shape=(w1win.shape[1],1) ) 
print("Running tuner4 with mp_X_train_input_scaled input shape:", mp_inputs)



# Create an instance of the tuner class
print("Creating an instance of the tuner class")
mt = CMdtuner(
    X_train=train,
    X_val=val,
    X_test=test,
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
    multiactivate=True,
    tf1=False,
    tf2=False,
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
mv_X_train = scaler.fit_transform(train)
mv_X_val = scaler.transform(val)
mv_X_test = scaler.transform(test)

# +-------------------------------------------------------------------
# Train and evaluate the model
# +-------------------------------------------------------------------

best_model[0].fit(mv_X_train, mv_X_val,mv_X_test, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_batch_size)
best_model[0].evaluate(mv_X_val, mv_X_test)


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
plt.show()
print("Plot Model saved to ",mp_basepath + '//' + 'plot.png')


# +-------------------------------------------------------------------
# Save model to ONNX
# +-------------------------------------------------------------------
# Save the model to ONNX format

mp_output_path = mp_data_path + "model_" + mp_symbol_primary + "_" + mp_datatype + "_" + str(mp_seconds) + ".onnx"
print(f"output_path: ",mp_output_path)
onnx_model, _ = tf2onnx.convert.from_keras(best_model[0], opset=self.batch_size)
onnx.save_model(onnx_model, mp_output_path)
print(f"model saved to ",mp_output_path)

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
