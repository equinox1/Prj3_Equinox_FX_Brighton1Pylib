# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
# import os
import os
import pathlib
from pathlib import Path, PurePosixPath
import posixpath
import sys
import time
import json
import keyring as kr
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
from sklearn.model_selection import train_test_split
# Import dataclasses for data manipulation
import pandas as pd
from dataclasses import dataclass
# Import TensorFlow for machine learning
import tensorflow as tf
import onnx
import tf2onnx
import onnxruntime
import onnxruntime.backend as backend
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import warnings
from numpy import concatenate
# Import equinox functionality
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup, CMqlWindowGenerator
from tsMqlMLTune import CMdtuner
from tsMqlReference import CMqlTimeConfig
from tsMqlSetup import tsMqlSetup
from tsMqlReference import CMqlTimeConfig

s1 = tsMqlSetup(loglevel='INFO', warn='ignore')

tm = CMqlTimeConfig(basedatatime='SECONDS', loadeddatatime='MINUTES')
MINUTES = int(tm.get_timevalue('MINUTES'))
HOURS = int(tm.get_timevalue('HOURS'))
DAYS = int(tm.get_timevalue('DAYS'))

TIMEZONE = tm.TIME_CONSTANTS['TIMEZONES'][0]
TIMEFRAME = tm.TIME_CONSTANTS['TIMEFRAME']['H4']

#current date and time
CURRENTYEAR = datetime.now().year
CURRENTYEAR = datetime.now().year
CURRENTDAYS = datetime.now().day
CURRENTMONTH = datetime.now().month
print("CURRENTYEAR:",CURRENTYEAR, "CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH)

#MQL constants
broker = "METAQUOTES" # "ICM" or "METAQUOTES"
tm = CMqlTimeConfig()

mp_symbol_primary = tm.TIME_CONSTANTS['SYMBOLS'][0]
mp_symbol_secondary = tm.TIME_CONSTANTS['SYMBOLS'][1]
mp_shiftvalue = tm.TIME_CONSTANTS['DATATYPE']['MINUTES']
mp_unit = tm.TIME_CONSTANTS['UNIT'][1] 
print("mp_symbol_primary:",mp_symbol_primary, "mp_symbol_secondary:",mp_symbol_secondary, "mp_shiftvalue:",mp_shiftvalue, "mp_unit:",mp_unit)
MPDATAFILE1 =  "tickdata1.csv"
MPDATAFILE2 =  "ratesdata1.csv"

c0 = CMqlBrokerConfig(broker, mp_symbol_primary, MPDATAFILE1, MPDATAFILE2)
broker_config = c0.set_mql_broker()
BROKER = broker_config['BROKER']
MPPATH = broker_config['MPPATH']
MPBASEPATH = broker_config['MPBASEPATH']
MPDATAPATH = broker_config['MPDATAPATH']
MPFILEVALUE1 = broker_config['MPFILEVALUE1']
MPFILEVALUE2 = broker_config['MPFILEVALUE2']
MKFILES = broker_config['MKFILES']
print(f"Broker: {BROKER}")
print(f"Path: {MPPATH}")
print(f"Data Path: {MPDATAPATH}")
print(f"File 1: {MPFILEVALUE1}")
print(f"File 2: {MPFILEVALUE2}")
print(f"Files Path: {MKFILES}")

# +-------------------------------------------------------------------
# STEP:Start MetaTrader 5 (MQL) terminal login
# +-------------------------------------------------------------------
# Retrieve and validate credentials
cred = kr.get_credential(broker_config["BROKER"], "")
if not cred:
    raise ValueError("Credentials not found in keyring")
try:
    MPLOGIN = int(cred.username)
    MPPASS = str(cred.password)
except ValueError:
    raise ValueError("Invalid credentials format")

print(f"Logging in as: {MPLOGIN}")
# Initialize MT5 terminal and login
c1 = CMqlinit(
    MPPATH=broker_config["MPPATH"],
    MPLOGIN=MPLOGIN,
    MPPASS=MPPASS,
    MPSERVER=broker_config["MPSERVER"],
    MPTIMEOUT=broker_config["MPTIMEOUT"],
    MPPORTABLE=broker_config["MPPORTABLE"],
    MPENV=broker_config["MPENV"]
)
if not c1.run_mql_login():
    raise ConnectionError("Failed to login to MT5 terminal")
print("Terminal Info:", mt5.terminal_info())

# +-------------------------------------------------------------------
# STEP: Configuration settings
# +-------------------------------------------------------------------
feature_scaler = MinMaxScaler()

#data load states
mp_data_rownumber = False
mp_data_show_dtype = False
mp_data_loadapiticks = True
mp_data_loadapirates = True
mp_data_loadfileticks = True
mp_data_loadfilerates = True

#ml states
mp_ml_shuffle = False
#ml Keras states
mp_ml_cnn_model = True
mp_ml_lstm_model = True
mp_ml_gru_model = True
mp_ml_transformer_model = True
#model parameters
mp_ml_run_single_input_model = True
mp_ml_run_single_input_submodels = False # not implemented yet    

#Machine Learning (ML) variables
mp_ml_loadtensor = True
mp_ml_loadtemporian = False
mp_ml_multi_inputs = False
mp_ml_tensor_shape = False
mp_ml_multiactivate=True

#Features and target and label definitions
#Time-series datasets usually have a sequence of values over time. For supervised learning:
#X (features): A sequence of input time steps.
#y (labels): The target value(s) for the corresponding time step(s).
#For time windows: #Choose a fixed window size (e.g., window_size = 24)Slide the window across the dataset to create samples.
#Input sequence (X): A series of consecutive time steps as input features.
#Target (y): The value(s) you want to predict, often at a future time step.
#For example, in a univariate time series (e.g., predicting stock prices), X is a sequence of past prices,
# and y is the price at the next time step.




mp_ml_input_keyfeat = {'Close'} # the feature to predict
mp_ml_output_target = {'Target_Label'} # the feature to predict
mp_ml_input_keyfeat_scaled = {feat + '_Scaled' for feat in mp_ml_input_keyfeat}  # the feature to predict
mp_ml_output_target_scaled = {targ + '_Shifted' for targ in mp_ml_output_target}  # the target shifted to predict
mp_ml_target = mp_ml_output_target_scaled
mp_ml_label = mp_ml_target

mp_ml_label_count=len(mp_ml_label)
mp_ml_windowmodel = '24_24_1' # '24_24_1' or '6_1_1'
mp_ml_batch_size = 32
mp_ml_cfg_period1 = 24
mp_ml_cfg_period2= 6
mp_ml_cfg_period = mp_ml_cfg_period1
#Splitting the data
mp_ml_train_split = 0.7
mp_ml_validation_split = 0.2
mp_ml_test_split = 0.1
#Best Models
mp_ml_mp_ml_num_models = 1
mp_ml_num_trials = 1

# Set parameters for the Tensorflow keras model
mp_ml_tf_param_steps = 1
mp_ml_tf_param_max_epochs=10
mp_ml_tf_param_min_epochs=1
mp_ml_tf_param_epochs = 200
mp_ml_tf_param_chk_patience = 3
mp_ml_tf_shiftin=1
mp_ml_tf_ma_windowin=14 # 14 DAYS typical indicator window
mp_ml_cfg_period1=24 # 24 HOURS
mp_ml_cfg_period2=6 # 6 HOURS
mp_ml_cfg_period=1 # 1 HOURS

# Set the shape of the data
mp_ml_input_shape=3 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_cnn_shape=2 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_lstm_shape=2 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_gru_shape=2 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_transformer_shape = 2 # mp_data_tab_rows, batches, timesteps, features


# setting dictionary for the model
common_ml_params = {
    "lookahead_periods": mp_ml_cfg_period,
    "ma_window": mp_ml_tf_ma_windowin,
    "hl_avg_col": "HLAvg",
    "ma_col": "SMA",
    "returns_col": "LogReturns",
    "shift_in": mp_ml_tf_shiftin,
    "rownumber": mp_data_rownumber,
}

other_ml_params = {
    "returns_col_scaled": "LogReturns_scaled",
}
mp_ml_return_col_scaled = other_ml_params["returns_col_scaled"]

#Data variables
mp_data_data_target = 1 # 1: just the target, 2: target and features, 3: target, features and time, 4: full dataset
mv_data_dfname1 = "df_rates1"
mv_data_dfname2 = "df_rates2"
mp_data_rows = 1000 # number of mp_data_tab_rows to fetch
mp_data_rowcount = 10000 # number of mp_data_tab_rows to fetch
mp_data_history_size = 5 # Number of years of data to fetch
mp_data_cfg_usedata = 'loadfilerates' # 'loadapiticks' or 'loadapirates'or loadfileticks or loadfilerates
mp_data_command_ticks = mt5.COPY_TICKS_ALL
mp_data_command_rates = None
mp_data_timeframe = TIMEFRAME
mp_data_tab_rows = 10
mp_data_tab_width=30
print("TIMEFRAME:",TIMEFRAME, "TIMEZONE:",TIMEZONE,"MT5 TIMEFRAME:",mp_data_timeframe)
# +-------------------------------------------------------------------
# STEP: Data Preparation and Loading
# +-------------------------------------------------------------------
# Set up dataset
d1 = CMqldatasetup(lp_features=mp_ml_input_keyfeat, lp_target=mp_ml_target, lp_label=mp_ml_label, lp_label_count=mp_ml_label_count)
print("CURRENTYEAR:",CURRENTYEAR, "CURRENTYEAR-mp_data_history_size",CURRENTYEAR-mp_data_history_size,"CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH,"TIMEZONE:",TIMEZONE)
mv_data_utc_from = d1.set_mql_timezone(CURRENTYEAR-mp_data_history_size, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
mv_data_utc_to = d1.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
print("UTC From:",mv_data_utc_from)
print("UTC To:",mv_data_utc_to)

# Load tick data from MQL and FILE
mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = d1.run_load_from_mql(mp_data_loadapiticks, mp_data_loadapirates, mp_data_loadfileticks, mp_data_loadfilerates, mv_data_dfname1, mv_data_dfname2, mv_data_utc_from, mp_symbol_primary, mp_data_rows, mp_data_rowcount, mp_data_command_ticks,mp_data_command_rates, MPDATAPATH, MPFILEVALUE1, MPFILEVALUE2, TIMEFRAME)

#wrangle the data merging and transforming time to numeric
if len(mv_tdata1apiticks) > 0:  
    mv_tdata1apiticks = d1.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
if len(mv_tdata1apirates) > 0:
    mv_tdata1apirates = d1.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
if len(mv_tdata1loadticks) > 0:
    mv_tdata1loadticks = d1.wrangle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
if len(mv_tdata1loadrates) > 0:
    mv_tdata1loadrates = d1.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
           
# Create targets
mv_tdata1apiticks = d1.create_target_wrapper(
    df=mv_tdata1apiticks,
    bid_column="T1_Bid_Price",
    ask_column="T1_Ask_Price",
    column_in="T1_Bid_Price",
    column_out1=list(mp_ml_input_keyfeat)[0],
    column_out2=list(mp_ml_target)[0],
    open_column="R1_Open",
    high_column="R1_High",
    low_column="R1_Low",
    close_column="R1_Close",
    run_mode=1,
    **common_ml_params
)

mv_tdata1apirates = d1.create_target_wrapper(
    df=mv_tdata1apirates,
    bid_column="R1_Bid_Price",
    ask_column="R1_Ask_Price",
    column_in="R1_Close",
    column_out1=list(mp_ml_input_keyfeat)[0],
    column_out2=list(mp_ml_target)[0],
    open_column="R1_Open",
    high_column="R1_High",
    low_column="R1_Low",
    close_column="R1_Close",
    run_mode=2,
    **common_ml_params
)

mv_tdata1loadticks = d1.create_target_wrapper(
    df=mv_tdata1loadticks,
    bid_column="T2_Bid_Price",
    ask_column="T2_Ask_Price",
    column_in="T2_Bid_Price",
    column_out1=list(mp_ml_input_keyfeat)[0],
    column_out2=list(mp_ml_target)[0],
    open_column="R2_Open",
    high_column="R2_High",
    low_column="R2_Low",
    close_column="R2_Close",
    run_mode=3,
    **common_ml_params
)

mv_tdata1loadrates = d1.create_target_wrapper(
    df=mv_tdata1loadrates,
    bid_column="R2_Bid_Price",
    ask_column="R2_Ask_Price",
    column_in="R2_Close",
    column_out1=list(mp_ml_input_keyfeat)[0],
    column_out2=list(mp_ml_target)[0],
    open_column="R2_Open",
    high_column="R2_High",
    low_column="R2_Low",
    close_column="R2_Close",
    run_mode=4,
    **common_ml_params
)

# Display the data
d1.run_mql_print(mv_tdata1apiticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1apirates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1loadticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1loadrates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")

# copy the data for config selection
data_sources = [mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates]
data_copies = [data.copy() for data in data_sources]
mv_tdata2a, mv_tdata2b, mv_tdata2c, mv_tdata2d = data_copies
# Define a mapping of configuration values to data variables
data_mapping = {
    'loadapiticks': mv_tdata2a,
    'loadapirates': mv_tdata2b,
    'loadfileticks': mv_tdata2c,
    'loadfilerates': mv_tdata2d
}

# Check the switch of which file to use
if mp_data_cfg_usedata in data_mapping:
    print(f"Using {mp_data_cfg_usedata.replace('load', '').replace('api', 'API ').replace('file', 'File ').replace('ticks', 'Tick data').replace('rates', 'Rates data')}")
    mv_tdata2 = data_mapping[mp_data_cfg_usedata]
else:
    print("Invalid data configuration")
    mv_tdata2 = None

# print shapes of the data
print("SHAPE: mv_tdata2 shape:", mv_tdata2.shape)


# +-------------------------------------------------------------------
# STEP: Normalize the data
# +-------------------------------------------------------------------
# Chosed featur columns such as Close will be normalised to Close_Scaled
mp_ml_input_keyfeat_list = list(mp_ml_input_keyfeat)
print("Normalise mp_ml_input_keyfeat_list",mp_ml_input_keyfeat_list)
mv_tdata2_scaled = feature_scaler.fit_transform(mv_tdata2[mp_ml_input_keyfeat_list].values)

# convert to pd
mv_tdata2_scaled = pd.DataFrame(mv_tdata2_scaled, columns=list(mp_ml_input_keyfeat_scaled))

print("print Normalise")
d1.run_mql_print(mv_tdata2_scaled,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
print("End of Normalise print")

# Move the target column to the end of the DataFrame
last_col = list(mp_ml_target)[0]
# Assuming mv_tdata2 and mv_train_scaled are pandas DataFrames add the single column scales the base train data
print("Move the target column to the end of the DataFrame")
#d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
mv_tdata2_combined = d1.move_col_to_end(pd.concat([mv_tdata2, mv_tdata2_scaled], axis=1), last_col)
mv_tdata2 = mv_tdata2_combined

d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
print("mv_tdata2.shape",mv_tdata2.shape)


# +-------------------------------------------------------------------
# STEP: remove datetime dtype to numeric from the data
# +-------------------------------------------------------------------

if len(mv_tdata2) > 0:
    mv_tdata2 = d1.wrangle_time(mv_tdata2, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=True, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=False)

d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
# +-------------------------------------------------------------------
# STEP: add The time index to the data
# +-------------------------------------------------------------------
# Check the first column
first_column = mv_tdata2.columns[0]
print("PRE INDEX: Count: ",len(mv_tdata2))

# Ensure no missing values, duplicates, or type issues
mv_tdata2[first_column] = mv_tdata2[first_column].fillna('Unknown')  # Handle NaNs
mv_tdata2[first_column] = mv_tdata2[first_column].astype(str)  # Uniform type
mv_tdata2[first_column] = mv_tdata2[first_column].str.strip()  # Remove whitespaces

# Set the first column as index
mv_tdata2.set_index(first_column, inplace=True)
mv_tdata2=mv_tdata2.dropna()
print("POST INDEX: Count: ",len(mv_tdata2))


# +-------------------------------------------------------------------
# STEP: set the dataset to just the features and the target and sort by time
# +-------------------------------------------------------------------

if mp_data_data_target == 1:
    mv_tdata2 = mv_tdata2[[list(mp_ml_input_keyfeat_scaled)[0]]]
    d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
  
if mp_data_data_target == 2:
    mv_tdata2 = mv_tdata2[[list(mp_ml_input_keyfeat_scaled)[0], list(mp_ml_target)[0]]]
    d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
 
if mp_data_data_target == 3:
    mv_tdata2 = mv_tdata2[[mv_tdata2.columns[0]] + [list(mp_ml_input_keyfeat_scaled)[0], list(mp_ml_target)[0]]]
   # Ensure the data is sorted by time
    mv_tdata2 = mv_tdata2.sort_index()
    d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
  
if mp_data_data_target == 4:
    # Ensure the data is sorted by time use full dataset
    mv_tdata2 = mvtrain.sort_index()
    d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")


# +-------------------------------------------------------------------
# STEP: Generate X and y from the Time Series
# +-------------------------------------------------------------------
m1 = CMqlmlsetup() # Create an instance of the class
# 1: 24 HOURS/24 HOURS prediction window
print("1: MINUTES data entries per time frame: MINUTES:", MINUTES, "HOURS:", MINUTES * 60, "DAYS:", MINUTES * 60 * 24)
timeval = MINUTES * 60 # hours
pasttimeperiods = 24
futuretimeperiods = 24
predtimeperiods = 1

print("timeval:",timeval, "pasttimeperiods:",pasttimeperiods, "futuretimeperiods:",futuretimeperiods, "predtimeperiods:",predtimeperiods)
past_width = pasttimeperiods * timeval
future_width = futuretimeperiods * timeval
pred_width = predtimeperiods * timeval
print("past_width:",past_width, "future_width:",future_width, "pred_width:",pred_width)

#  Create the input features (X) and target values (y)
print("list(mp_ml_input_keyfeat_scaled)",list(mp_ml_input_keyfeat_scaled))
mv_tdata2_X,mv_tdata2_y=m1.create_Xy_time_windows(mv_tdata2, past_width, future_width)
print("mv_tdata2_X.shape",mv_tdata2_X.shape, "mv_tdata2_y.shape",mv_tdata2_y.shape)

# +-------------------------------------------------------------------
# STEP: Split the data into training and test sets Fixed Partitioning
# +-------------------------------------------------------------------
# Batch size alignmentfit the number of rows as whole number divisible by the batch size to avoid float errors   
batch_size = mp_ml_batch_size

precount = len(mv_tdata2_X)
mv_tdata2_X = mv_tdata2_X[mv_tdata2_X.shape[0] % batch_size:]
print("Batch size alignment: mv_tdata2_X shape:", mv_tdata2_X.shape,"Precount:",precount,"Postcount:",len(mv_tdata2_X))

precount = len(mv_tdata2_y)
mv_tdata2_y = mv_tdata2_y[mv_tdata2_y.shape[0] % batch_size:]
print("Batch size alignment: mv_tdata2_y shape:", mv_tdata2_y.shape,"Precount:",precount,"Postcount:",len(mv_tdata2_y))


# Use the first 70% of the data for training, the next 20% for validation, and the last 10% for testing
# Use the Sklearn train_test_split function to split the data # Split into training + validation and test sets
# shuffle=False ensures the temporal order of data is preserved.
X_train_val, X_test, y_train_val, y_test = train_test_split(mv_tdata2_X, mv_tdata2_y, test_size=mp_ml_test_split, shuffle=False)

# Further split training and validation sets shuffle=False ensures the temporal order of data is preserved.
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=mp_ml_test_split, shuffle=False)

print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)

"""


# +-------------------------------------------------------------------
# STEP: Window Generator and Data Slicing
# +-------------------------------------------------------------------

X_win_i24_o24_l1 = CMqlWindowGenerator(
    input_width=past_width,
    shift=future_width,
    label_width=pred_width,
    train_df=X_train,
    val_df=X_val,
    test_df=X_test,
    label_columns=mp_ml_label,
    batch_size=mp_ml_batch_size
)

print("X_win_i24_o24_l1:", X_win_i24_o24_l1.total_window_size)

y_win_i24_o24_l1 = CMqlWindowGenerator(
    input_width=past_width,
    shift=future_width,
    label_width=pred_width,
    train_df=y_train,
    val_df=y_val,
    test_df=y_test,
    label_columns=mp_ml_label,
    batch_size=mp_ml_batch_size
)

# 2: 6 HOURS/1 HOURS prediction window
print("2: MINUTES data entries per time frame: MINUTES:", MINUTES, "HOURS:", MINUTES * 60, "DAYS:", MINUTES * 60 * 24)
timeval = MINUTES * 60 # hours
pasttimeperiods = 6
futuretimeperiods = 1
predtimeperiods = 1
past_width = pasttimeperiods * timeval
future_width = futuretimeperiods * timeval
pred_width = predtimeperiods * timeval

X_win_i6_o1_l1 = CMqlWindowGenerator(
    input_width=past_width,
    shift=future_width,
    label_width=pred_width,
    train_df=X_train,
    val_df=X_val,
    test_df=X_test,
    label_columns=mp_ml_label,
    batch_size=mp_ml_batch_size
)
print("X_win_i6_o1_l1:", X_win_i6_o1_l1.total_window_size)

y_win_i6_o1_l1 = CMqlWindowGenerator(
    input_width=past_width,
    shift=future_width,
    label_width=pred_width,
    train_df=y_train,
    val_df=y_val,
    test_df=y_test,
    label_columns=mp_ml_label,
    batch_size=mp_ml_batch_size
)
print("y_win_i6_o1_l1:", y_win_i6_o1_l1.total_window_size)

# +-------------------------------------------------------------------
# STEP: Data Slice and Split within windows
# +-------------------------------------------------------------------
# 1: Slice the data into sample sets # X 24 x 24 x 1
shift_size = 100
train_slice_win_i24_o24_l1 = win_i24_o24_l1.slice_window(mv_tdata2, win_i24_o24_l1.total_window_size, shift_size)

# 2: Slice the data into sample sets # X 6 x 1 x 1
train_slice_win_i6_o1_l1 = win_i6_o1_l1.slice_window(mv_tdata2, win_i6_o1_l1.total_window_size, shift_size)


# +-------------------------------------------------------------------
# STEP: Split the data into windows split into inputs and labels
# +-------------------------------------------------------------------
#1: 24 x 24 x1 split window into features and labels
inputs_train_split_win_i24_o24_l1, labels_train_split_win_i24_o24_l1 = win_i24_o24_l1.split_window(train_slice_win_i24_o24_l1)

print('All shapes are: (batch, time, features)')
print(f'inputs (Features: ind var X) shape: {inputs_train_split_win_i24_o24_l1.shape}')
print(f'Labels (Labels dep var y) shape: {labels_train_split_win_i24_o24_l1.shape}')
print("total_window_size=win_i24_o24_l1.total_window_size", win_i24_o24_l1.total_window_size)

#2: 24 x 24 x1 split window into features and labels
inputs_train_split_win_i6_o1_l1, labels_train_split_win_i6_o1_l1 = win_i6_o1_l1.split_window(train_slice_win_i6_o1_l1)

print('All shapes are: (batch, time, features)')
print(f'inputs (Features: ind var X) shape: {inputs_train_split_win_i6_o1_l1.shape}')
print(f'Labels (Labels dep var y) shape: {labels_train_split_win_i6_o1_l1.shape}')
print("total_window_size=win_i6_o1_l1.total_window_size", win_i6_o1_l1.total_window_size)

# +-------------------------------------------------------------------
# STEP: Create TF datasets
# +-------------------------------------------------------------------

# 24 x 24 x 1 Create TF datasets
#train_ds_win_i24_o24_l1, val_ds_win_i24_o24_l1, test_ds_win_i24_o24_l1 = win_i24_o24_l1.make_dataset(win_i24_o24_l1)

win_i24_o24_l1.train 
win_i24_o24_l1.val
win_i24_o24_l1.test

# 6 x 1 x 1 Create TF datasets
#train_ds_win_i6_o1_l1, val_ds_win_i6_o1_l1, test_ds_win_i6_o1_l1 = win_i6_o1_l1.make_dataset(win_i6_o1_l1)
win_i6_o1_l1.train
win_i6_o1_l1.val
win_i6_o1_l1.test

# +-------------------------------------------------------------------
# STEP: Window config selection
# +-------------------------------------------------------------------
#set the window model switch
if mp_ml_windowmodel == '24_24_1':
    window = win_i24_o24_l1
    features_shape=inputs_train_split_win_i24_o24_l1.shape
    labels_shape=labels_train_split_win_i24_o24_l1.shape
    train_dataset = win_i24_o24_l1.train
    val_dataset = win_i24_o24_l1.val
    test_dataset = win_i24_o24_l1.test
    
elif mp_ml_windowmodel == '6_1_1':
    window = win_i6_o1_l1
    features_shape=inputs_train_split_win_i6_o1_l1.shape
    train_dataset = win_i6_o1_l1.train
    val_dataset = win_i6_o1_l1.val
    test_dataset = win_i6_o1_l1.test
    
print("mp_ml_windowmodel:", mp_ml_windowmodel,"window:", window)

# +-------------------------------------------------------------------
# STEP: add tensor values for model input
# +-------------------------------------------------------------------

train_shape, val_shape, test_shape = None, None, None

for dataset, name in zip([train_dataset, val_dataset, test_dataset], ['train', 'val', 'test']):
    for spec in dataset.element_spec:
        if name == 'train':
            train_shape = spec.shape
        elif name == 'val':
            val_shape = spec.shape
        elif name == 'test':
            test_shape = spec.shape

# Final summary of shapes
print(f"Train shape: {train_shape}")
print(f"Val shape: {val_shape}")
print(f"Test shape: {test_shape}")
input_shape = train_shape
print("TrainSum:","shape:", train_shape,"Rows:",len(train_dataset),"window.winsize:",window.total_window_size,"window.input_width:",window.input_width,"window.shift:",window.shift,"window.label_width:",window.label_width,"features_shape:",features_shape,"labels_shape:",labels_shape)
print("ValSum:","shape:", val_shape,"Rows:",len(val_dataset),"window.windsize:",window.total_window_size,"window.input_width:",window.input_width,"window.shift:",window.shift,"window.label_width:",window.label_width,"features_shape:",features_shape,"labels_shape:",labels_shape)
print("TestSum:","shape:", test_shape,"Rows:",len(test_dataset),"window.winsize:",window.total_window_size,"window.input_width:",window.input_width,"window.shift:",window.shift,"window.label_width:",window.label_width,"features_shape:",features_shape,"labels_shape:",labels_shape)

#found shape=(None, 1440, 11) 11 features is the dataset columns 1440 is the 24 hrs in minutes

#Example: if you have 30 images of 50x50 pixels in RGB (3 channels), the shape of your input data is (30,50,50,3). 
#Then your input layer tensor, must have this shape (see details in the "shapes in keras" section).
#Earlier, I gave an example of 30 images, 50x50 pixels and 3 channels, having an input shape of (30,50,50,3)
#Since the input shape is the only one you need to define, Keras will demand it in the first layer.
#But in this definition, Keras ignores the first dimension, which is the batch size. Your model should be able 
#to deal with any batch size, so you define only the other dimensions:
#input_shape = (50,50,3) regardless of how many images I have, each image has this shape   
#Optionally, or when it's required by certain kinds of models, you can pass the shape containing the batch 
#size via batch_input_shape=(30,50,50,3) or batch_shape=(30,50,50,3). This limits your training possibilities
#to this unique batch size, so it should be used only when really required.

# +-------------------------------------------------------------------
# STEP: Tune best model Hyperparameter tuning and model setup
# +-------------------------------------------------------------------
# Hyperparameter configuration
def get_hypermodel_params():
    today_date = date.today().strftime('%Y-%m-%d %H:%M:%S')
    random_seed = np.random.randint(0, 1000)
    base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsModelData"
    project_name = "prjEquinox1_prod"
    subdir = os.path.join(base_path, 'tshybrid_ensemble_tuning_prod', str(1))

    return {
        'activation1': 'relu',
        'activation2': 'linear',
        'activation3': 'softmax',
        'activation4': 'sigmoid',
        'objective': 'val_loss',
        'max_epochs': mp_ml_tf_param_max_epochs,
        'factor': 10,
        'seed': 42,
        'hyperband_iterations': 1,
        'tune_new_entries': False,
        'allow_new_entries': False,
        'max_retries_per_trial': 5,
        'max_consecutive_failed_trials': 6,
        'validation_split': 0.2,
        'epochs': mp_ml_tf_param_epochs,
        'batch_size': mp_ml_batch_size,
        'dropout': 0.2,
        'optimizer': 'adam',
        'loss': 'mean_squared_error',
        'metrics': ['mean_squared_error'],
        'directory': os.path.join(base_path, 'tshybrid_ensemble_tuning_prod'),
        'logger': None,
        'tuner_id': None,
        'overwrite': True,
        'executions_per_trial': 1,
        'chk_fullmodel': True,
        'chk_verbosity': 1,
        'chk_mode': 'min',
        'chk_monitor': 'val_loss',
        'chk_sav_freq': 'epoch',
        'chk_patience': mp_ml_tf_param_chk_patience,
        'modeldatapath': base_path,
        'project_name': project_name,
        'today': today_date,
        'random': random_seed,
        'baseuniq': str(1),
        'basepath': subdir,
        'checkpoint_filepath': posixpath.join(base_path, 'tshybrid_ensemble_tuning_prod', project_name),
    }

# Print configuration details for logging
def log_config(hypermodel_params):
    print(f"mp_ml_random: {hypermodel_params['random']}")
    print(f"mp_ml_today: {hypermodel_params['today']}")
    print(f"mp_checkpoint_filepath: {hypermodel_params['checkpoint_filepath']}")

# Initialize the tuner class
def initialize_tuner(hypermodel_params):
    try:
        print("Creating an instance of the tuner class")
        mt = CMdtuner(
            # tf datasets
            traindataset=train_dataset,
            valdataset=val_dataset,
            testdataset=test_dataset,

            # Input data shape
            trainshape=train_shape,
            valshape=val_shape,
            testshape=test_shape,

            # Models
            cnn_model=mp_ml_cnn_model,
            lstm_model=mp_ml_lstm_model,
            gru_model=mp_ml_gru_model,
            transformer_model=mp_ml_transformer_model,

            # Model inputs
            tensorshape=mp_ml_tensor_shape,
            inputs=input_shape,
            shape=input_shape,
            cnn_shape=mp_ml_input_cnn_shape,
            lstm_shape=mp_ml_input_lstm_shape,
            gru_shape=mp_ml_input_gru_shape,
            transformer_shape=mp_ml_input_transformer_shape,

            multi_inputs=mp_ml_multi_inputs,

            # Model parameters
            run_single_input_model=mp_ml_run_single_input_model,
            run_single_input_submodels=mp_ml_run_single_input_submodels,
            objective=hypermodel_params['objective'],
            max_epochs=hypermodel_params['max_epochs'],
            min_epochs=mp_ml_tf_param_min_epochs,
            factor=hypermodel_params['factor'],
            seed=hypermodel_params['seed'],
            hyperband_iterations=hypermodel_params['hyperband_iterations'],
            tune_new_entries=hypermodel_params['tune_new_entries'],
            allow_new_entries=hypermodel_params['allow_new_entries'],
            max_retries_per_trial=hypermodel_params['max_retries_per_trial'],
            max_consecutive_failed_trials=hypermodel_params['max_consecutive_failed_trials'],
            validation_split=hypermodel_params['validation_split'],
            epochs=hypermodel_params['epochs'],
            batch_size=hypermodel_params['batch_size'],
            dropout=hypermodel_params['dropout'],
            optimizer=hypermodel_params['optimizer'],
            loss=hypermodel_params['loss'],
            metrics=hypermodel_params['metrics'],
            directory=hypermodel_params['directory'],
            basepath=hypermodel_params['basepath'],
            project_name=hypermodel_params['project_name'],
            logger=hypermodel_params['logger'],
            tuner_id=hypermodel_params['tuner_id'],
            overwrite=hypermodel_params['overwrite'],
            executions_per_trial=hypermodel_params['executions_per_trial'],
            chk_fullmodel=hypermodel_params['chk_fullmodel'],
            chk_verbosity=hypermodel_params['chk_verbosity'],
            chk_mode=hypermodel_params['chk_mode'],
            chk_monitor=hypermodel_params['chk_monitor'],
            chk_sav_freq=hypermodel_params['chk_sav_freq'],
            chk_patience=hypermodel_params['chk_patience'],
            checkpoint_filepath=hypermodel_params['checkpoint_filepath'],
            modeldatapath=hypermodel_params['modeldatapath'],
            step=mp_ml_tf_param_steps,
            multiactivate=True,
            tf1=False,
            tf2=False,
        )
        print("Tuner initialized successfully.")
        return mt
    except Exception as e:
        print(f"Error initializing the tuner: {e}")
        raise


# +-------------------------------------------------------------------
# STEP:Run the Tuner to find the best model configuration
# +-------------------------------------------------------------------
# Run the tuner to find the best model configuration

# Load hyperparameters
hypermodel_params = get_hypermodel_params()

# Log the configuration
log_config(hypermodel_params)

# Initialize tuner
mt = initialize_tuner(hypermodel_params)


#--------------------------------------------------
print("Running Main call to tuner")
best_model = mt.tuner.get_best_models(mp_ml_num_models=1)
best_params = mt.tuner.get_best_hyperparameters(mp_ml_num_trials=1)[0]
best_model[0].summary()

# +-------------------------------------------------------------------
# STEP: Train and evaluate the best model
# +-------------------------------------------------------------------
best_model[0].fit(train_dataset, validation_split=mp_ml_validation_split, epochs=mp_ml_tf_param_epochs, batch_size=mp_ml_batch_size)
best_model[0].evaluate(val_dataset, test_dataset)

# +-------------------------------------------------------------------
# STEP: Predictions
# +-------------------------------------------------------------------
# Fit scaler on training dataset features
scaler.fit(train_dataset)  # Fit on training features
predicted_fx_price = best_model[0].predict(test_dataset)  # Predict the test data

# Reshape predictions to 2D if needed
if predicted_fx_price.ndim == 1:
    predicted_fx_price = predicted_fx_price.reshape(-1, 1)

# Inverse transform predictions using the original scaler
predicted_fx_price = scaler.inverse_transform(
    np.hstack([
        np.zeros((predicted_fx_price.shape[0], train_dataset.shape[1] - 1)),  # Pad zeros for other columns
        predicted_fx_price
    ])
)[:, -1]  # Extract only the target column

# Fit target scaler on actual target values from training data
target_scaler = StandardScaler()
target_scaler.fit(train_dataset[:, -1].reshape(-1, 1))  # Assuming last column is the target

# Inverse transform real FX prices (if needed for comparison)
real_fx_price = target_scaler.inverse_transform(test_dataset[:, -1].reshape(-1, 1))


# +-------------------------------------------------------------------
# STEP: Performance Check
# +-------------------------------------------------------------------
# Evaluation and visualization
# Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values. 
# The lower the MSE, the better the model.

# Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values. 
# Like MSE, lower values indicate better model performance.

# R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in the
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
plt.savefig(mp_basepath + '/' + 'plot.png')
plt.show()
print("Plot Model saved to ", mp_basepath + '/' + 'plot.png')

# +-------------------------------------------------------------------
# STEP: Save model to ONNX
# +-------------------------------------------------------------------

# Save the model to ONNX format
mp_output_path = mp_data_path + "model_" + mp_symbol_primary + "_" + mp_datatype + "_" + str(mp_seconds) + ".onnx"
print(f"output_path: ",mp_output_path)
onnx_model, _ = tf2onnx.convert.from_keras(best_model[0], opset=self.batch_size)
onnx.save_model(onnx_model, mp_output_path)
print(f"model saved to ",mp_output_path)

# Assuming your model has a single input  Convert the model
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
"""