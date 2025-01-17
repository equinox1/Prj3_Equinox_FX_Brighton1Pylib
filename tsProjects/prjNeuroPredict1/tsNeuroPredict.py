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
MINS = int(tm.get_timevalue('MINUTES'))
HOUR = int(tm.get_timevalue('HOURS'))
DAY = int(tm.get_timevalue('DAYS'))

TIMEZONE = tm.TIME_CONSTANTS['TIMEZONES'][0]
TIMEFRAME = tm.TIME_CONSTANTS['TIMEFRAME']['H4']

#current date and time
CURRENTYEAR = datetime.now().year
CURRENTYEAR = datetime.now().year
CURRENTDAY = datetime.now().day
CURRENTMONTH = datetime.now().month
print("CURRENTYEAR:",CURRENTYEAR, "CURRENTDAY:",CURRENTDAY, "CURRENTMONTH:",CURRENTMONTH)

#MQL constants
broker = "METAQUOTES" # "ICM" or "METAQUOTES"
tm = CMqlTimeConfig()

mp_symbol_primary = tm.TIME_CONSTANTS['SYMBOLS'][0]
mp_symbol_secondary = tm.TIME_CONSTANTS['SYMBOLS'][1]
mp_shiftvalue = tm.TIME_CONSTANTS['DATATYPE']['MINUTE']
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
# Start MetaTrader 5 (MQL) terminal login
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
# End of Login to MetaTrader 5 (MQL) terminal
# +-------------------------------------------------------------------
# +-------------------------------------------------------------------
# start of configuration variables
# +-------------------------------------------------------------------
mp_ml_loadtensor = True
mp_ml_loadtemporian = False
mp_ml_multi_inputs = False
mp_ml_tensor_shape = False
mp_ml_multiactivate=True

#data load states
mp_data_rownumber = True
mp_data_show_dtype = False
mp_data_loadapiticks = True
mp_data_loadapirates = True
mp_data_loadfileticks = True
mp_data_loadfilerates = True

mp_env_test = False
# +-------------------------------------------------------------------
# start of set variables
# +-------------------------------------------------------------------
#Machine Learning (ML) variables
mp_ml_target = {'target'}
mp_ml_label = {'close'} # the feature to predict
mp_ml_input_keyfeat = {'close'} # the feature to predict
mp_ml_input_keyfeat_scaled = {feat + '_scaled' for feat in mp_ml_input_keyfeat}  # the feature to predict
mp_ml_windowmodel = '6_1_1' # '24_24_1' or '6_1_1'
mp_ml_batch_size = 32
mp_ml_cfg_period24 = 24
mp_ml_cfg_period6 = 6


# Set parameters for the Tensorflow keras model
mp_ml_tf_param_steps = 1
mp_ml_tf_param_max_epochs=10
mp_ml_tf_param_min_epochs=1
mp_ml_tf_param_epochs = 200
mp_ml_tf_param_chk_patience = 3
mp_ml_tf_shiftin=1
mp_ml_tf_ma_windowin=14 # 14 days typical indicator window
mp_ml_cfg_period=24 # 24 hours
mp_ml_cfg_period=6 # 6 hours
mp_ml_cfg_period=1 # 1 hour

# Set the shape of the data
mp_ml_input_shape=4 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_cnn_shape=4 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_lstm_shape=6 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_gru_shape=7 # mp_data_tab_rows, batches, timesteps, features
mp_ml_input_transformer_shape = 8 # mp_data_tab_rows, batches, timesteps, features

#Data variables
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
# +-------------------------------------------------------------------
# End of start of set variables
# +-------------------------------------------------------------------
print("TIMEFRAME:",TIMEFRAME, "TIMEZONE:",TIMEZONE,"MT5 TIMEFRAME:",mp_data_timeframe)
# +-------------------------------------------------------------------
# Step 1.1 Data Preparation : Load the data: Use pandas to load and preprocess the data.
# +-------------------------------------------------------------------

# Set up dataset
d1 = CMqldatasetup()
print("CURRENTYEAR:",CURRENTYEAR, "CURRENTYEAR-mp_data_history_size",CURRENTYEAR-mp_data_history_size,"CURRENTDAY:",CURRENTDAY, "CURRENTMONTH:",CURRENTMONTH,"TIMEZONE:",TIMEZONE)
mv_data_utc_from = d1.set_mql_timezone(CURRENTYEAR-mp_data_history_size, CURRENTMONTH, CURRENTDAY, TIMEZONE)
mv_data_utc_to = d1.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAY, TIMEZONE)
print("UTC From:",mv_data_utc_from)
print("UTC To:",mv_data_utc_to)
# Load tick data from MQL
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


                
common_params = {
    "lookahead_periods": mp_ml_cfg_period,
    "ma_window": mp_ml_tf_ma_windowin,
    "hl_avg_col": "HLAvg",
    "ma_col": "SMA",
    "returns_col": "LogReturns",
    "shift_in": mp_ml_tf_shiftin,
    "rownumber": mp_data_rownumber,
}

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
    **common_params
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
    **common_params
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
    **common_params
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
    **common_params
)

# +-------------------------------------------------------------------
# Display the data
# +------------------------------------------------------------------
d1.run_mql_print(mv_tdata1apiticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1apirates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1loadticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
d1.run_mql_print(mv_tdata1loadrates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
# +-------------------------------------------------------------------
# Prepare and process the data
# +-------------------------------------------------------------------
mv_tdata2a = mv_tdata1apiticks.copy()  # Copy the data for further processing
mv_tdata2b = mv_tdata1apirates.copy()  # Copy the data for further processing
mv_tdata2c = mv_tdata1loadticks.copy()  # Copy the data for further processing
mv_tdata2d = mv_tdata1loadrates.copy()  # Copy the data for further processing

# Check the switch of which file to use
if mp_data_cfg_usedata == 'loadapiticks':
    print("Using API Tick data")
    mv_tdata2 = mv_tdata2a
   
elif mp_data_cfg_usedata == 'loadapirates':
    print("Using API Rates data")
    mv_tdata2 = mv_tdata2b

elif mp_data_cfg_usedata == 'loadfileticks':
    print("Using File Tick data")
    mv_tdata2 = mv_tdata2c

elif mp_data_cfg_usedata == 'loadfilerates':
    print("Using File Rates data")
    mv_tdata2 = mv_tdata2d

"""
# print shapes of X and y
print("SHAPE: mv_tdata2 shape:", mv_tdata2.shape)

# +-------------------------------------------------------------------
# Split the data into training and test sets FIXED Partitioning
#--------------------------------------------------------------------
# If the time series contains seasonality, we want to ensure that each period contains a whole number
# of seasons # (i.e. at least one year if it has annual seasonality).
# We then train our model on the training period and evaluate it on the validation period. 
# After tuning the model's hyperparameters to get the desired performance, youcan retrain it on the training 
# and validation data, and then test on the unseen test data.
# Another way to train and test forecasting models is by starting with a short training period and gradually
# increasing it over time. This is referred to as roll-forwardpartitioning.
#
# +-------------------------------------------------------------------
# Batch size alignment   
# +-------------------------------------------------------------------
batch_size = mp_ml_batch_size
mv_tdata2 = mv_tdata2[mv_tdata2.shape[0] % batch_size:]

print("Batch size alignment: mv_tdata2 shape:", mv_tdata2.shape)

# batch size constraint to reduce the complexity of the arithmetic required when working with the LSTM model.

# +-------------------------------------------------------------------
# Split the data into training and test sets through sequences
# +-------------------------------------------------------------------
#Note that val_size, test_size and window_size are also all multiples of batch_size
total_size = len(mv_tdata2)
batched_total_size = total_size - total_size % batch_size   # Ensure total_size is a multiple of batch_size
print("total_size:",total_size, "batched_total_size:",batched_total_size)
mp_train_split = 0.7
mp_validation_split = 0.2
mp_env_test_split = 0.1

train_size = int(batched_total_size * mp_train_split)
val_size = int(batched_total_size * mp_validation_split)
test_size = batched_total_size - train_size - val_size
print("train_size:",train_size, "val_size:",val_size, "test_size:",test_size)

# init object
m1 = CMqlmlsetup()
mp_shuffle = False
shiftin=1
shiftout=1
# Split the data into training and test sets

mv_train = mv_tdata2[0:int(train_size)] #features close
mv_val = mv_tdata2[int(train_size):int(train_size) + int(val_size)]
mv_test = mv_tdata2[int(train_size) + int(val_size):]

print("len(mv_train) ",len(mv_train),"len(mv_val) " ,len(mv_val), "len(mv_test)", len(mv_test))

# +-------------------------------------------------------------------
# End Split the data into training and test sets
# +-------------------------------------------------------------------

# +-------------------------------------------------------------------
# Normalize the data
# +-------------------------------------------------------------------
# Normalize the data via MaxMinScaler
#scaler = MinMaxScaler()
return_col = 'LogReturns'
return_col_scaled = 'LogReturns scaled'

# Initialize scalers for features and targets
feature_scaler = MinMaxScaler()

# Scale the Close column
  
# Scaling features (X)
mv_train_scaled = feature_scaler.fit_transform(mv_train[[mp_ml_input_keyfeat]].values)
mv_val_scaled = feature_scaler.transform(mv_val[[mp_ml_input_keyfeat]].values)
mv_test_scaled = feature_scaler.transform(mv_test[[mp_ml_input_keyfeat]].values)

# convert to pd
mv_train_scaled = pd.DataFrame(mv_train_scaled, columns=[mp_ml_input_keyfeat_scaled], index=mv_train.index)
mv_val_scaled = pd.DataFrame(mv_val_scaled, columns=[mp_ml_input_keyfeat_scaled], index=mv_val.index)
mv_test_scaled = pd.DataFrame(mv_test_scaled, columns=[mp_ml_input_keyfeat_scaled], index=mv_test.index)

# Move the target column to the end of the DataFrame
last_col = 'target'
# Assuming mv_train and mv_train_scaled are pandas DataFrames
mv_train_combined = d1.move_col_to_end(pd.concat([mv_train, mv_train_scaled], axis=1), last_col)
mv_val_combined = d1.move_col_to_end(pd.concat([mv_val, mv_val_scaled], axis=1), last_col)
mv_test_combined = d1.move_col_to_end(pd.concat([mv_test, mv_test_scaled], axis=1), last_col)

mv_train = mv_train_combined
mv_val = mv_val_combined
mv_test = mv_test_combined

d1.run_mql_print(mv_train,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
print("mv_train.shape",mv_train.shape)

# +-------------------------------------------------------------------
# End Normalize the data
# +-------------------------------------------------------------------
# +-------------------------------------------------------------------
# Data windowing
# +-------------------------------------------------------------------
#The models will make a set of predictions based on a window of consecutive smples from the data.
#The main features of the input windows are:

#The mp_data_tab_width (number of time steps) of the input and label windows.
#The time offset between them.
#Which features are used as inputs, labels, or both.
#This script builds a variety of models (including Linear, DNN, CNN and RNN models), and uses them for both:

#Single-output, and multi-output predictions.
#Single-time-step and multi-time-step predictions.
#This section focuses on implementing the data windowing so that it can be reused for all of those models.

#A window size, also known as “look-back period” is the amount of past smples, in our case minutes, 
# that you want to take into consideration at a point of time to predict the next smple. Think of it
# as the relevant immediate past smples that you want to rely on to decide if the financial instrument
# will go up or down.
# +-------------------------------------------------------------------
# Establish Windows for the data
# +-------------------------------------------------------------------
# +-------------------------------------------------------------------
# 1:24 hour/24 hour/1 hour prediction window
# +-------------------------------------------------------------------
#timeframes
print("1:Minutes data entries per time frame: MINUTE:",MINUTE, "HOUR:",MINUTE * 60, "DAYs:",MINUTE * 60 * 24)
# Establish Windows for the data
timeval = MINUTE * 60 # hours
timeperiods=24 # 24 hours
predtimeperiods=24 # no hours
print("timeval:",timeval,"timeperiods:",timeperiods)
mp_past_inputwidth_timewindow = (timeval) * timeperiods # minutes of history data INPUT WIDTH
mp_future_offsetwidth_timewindow =  (timeval) * predtimeperiods # future minutes

# Ensure mp_feature_columns is defined and is a list
mp_feature_columns = list(mv_train.columns)
# Ensure 'RowNumber' exists in the list before attempting to remove it
if 'RowNumber' in mp_feature_columns:
    mp_feature_columns.remove('RowNumber')
print("1:Feature Columns: ",mp_feature_columns)

mp_label_columns =mp_ml_label  # column names for label dependent variables
mp_num_features = len(mp_feature_columns) # feature independent variables x
mp_num_labels = len(mp_label_columns) # Number of labels dependent variable y


print("1:Window Params: input_width:",mp_past_inputwidth_timewindow, "label_width:",mp_num_labels, "shift:",mp_future_offsetwidth_timewindow, "label_columns:",mp_label_columns,"mp_num_labels",mp_num_labels)
print("1:Feature Columns: ",mp_feature_columns)
print("1:Label Columns: ",mp_label_columns)
print("1:Num Features: ",mp_num_features)
print("1:Num Labels: ",mp_num_labels)
print("mp_past_inputwidth_timewindow:",mp_past_inputwidth_timewindow, "mp_future_offsetwidth_timewindow:",mp_future_offsetwidth_timewindow)

win_i24_o24_l1 = CMqlWindowGenerator(
    input_width=(mp_past_inputwidth_timewindow),
    label_width=(mp_num_labels),
    shift=(mp_future_offsetwidth_timewindow),
    train_df=mv_train,
    val_df=mv_val,
    test_df=mv_test,
    label_columns=mp_label_columns
)
print("1: win_i24_o24_l1.total_window_size: ",win_i24_o24_l1.total_window_size)
# +-------------------------------------------------------------------
# 2:6 hour/1 hour /1 hour prediction window
# +-------------------------------------------------------------------
print("2:Minutes data entries per time frame: MINUTE:",MINUTE, "HOUR:",MINUTE * 60, "DAYs:",MINUTE * 60 * 24)

timeval = MINUTE * 60 # hours
timeperiods=6
predtimeperiods=1
print("timeval:",timeval,"timeperiods:",timeperiods)
mp_past_inputwidth_timewindow = (timeval) * timeperiods # 6 hours of history data INPUT WIDTH
mp_future_offsetwidth_timewindow =  (timeval) * predtimeperiods # one LABEL=1  prediction 1 hours in the future Offset 6 hours

# Ensure mp_feature_columns is defined and is a list
mp_feature_columns = list(mv_train.columns)
# Ensure 'RowNumber' exists in the list before attempting to remove it
if 'RowNumber' in mp_feature_columns:
    mp_feature_columns.remove('RowNumber')
print("1:Feature Columns: ",mp_feature_columns)

mp_label_columns =mp_ml_label  # column names for label dependent variables
mp_num_features = len(mp_feature_columns) # feature independent variables x
mp_num_labels = len(mp_label_columns) # Number of labels dependent variable y

print("2:Minutes data entries per time frame: MINUTE:",MINUTE, "HOUR:",MINUTE * 60, "DAYs:",MINUTE * 60 * 24)
print("2:Feature Columns: ",mp_feature_columns)
print("2:Label Columns: ",mp_label_columns)
print("2:Num Features: ",mp_num_features)
print("2:Num Labels: ",mp_num_labels)

win_i6_o1_l1 = CMqlWindowGenerator(
    input_width=int(mp_past_inputwidth_timewindow),
    label_width=int(mp_num_labels),
    shift=int(mp_future_offsetwidth_timewindow),
    train_df=mv_train,
    val_df=mv_val,
    test_df=mv_test,
    label_columns=mp_label_columns
)

print("2: win_i6_o1_l1.total_window_size: ",win_i6_o1_l1.total_window_size)
# +-------------------------------------------------------------------
# End  Establish Windows for the data
# +-------------------------------------------------------------------

# +-------------------------------------------------------------------
# Split the data into windows split into input features and labels
# +-------------------------------------------------------------------
# X 24 x 24 x 1
shift_size = 100 # value to start the next window splitting the total window size
window_size = win_i24_o24_l1.total_window_size # fullobject window size
train_df = mv_train
train_df = train_df.to_numpy(dtype=np.float32)
print("X 24 x 24 x 1 train_df.shape:",train_df.shape)
# slice windowsize into 3 window subset
train_slice_win_i24_o24_l1 = win_i24_o24_l1.slice_window(train_df, window_size, shift_size)
print("train_slice_win_i24_o24_l1.shape",train_slice_win_i24_o24_l1.shape)

#split window
inputs_train_slice_win_i24_o24_l1, labels_train_slice_win_i24_o24_l1 = win_i24_o24_l1.split_window(train_slice_win_i24_o24_l1)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {train_slice_win_i24_o24_l1.shape}')
print(f'inputs (Features: ind var X) shape: {inputs_train_slice_win_i24_o24_l1}')
print(f'Labels (Labels dep var y) shape: {labels_train_slice_win_i24_o24_l1}')
print("total_window_size=win_i24_o24_l1.total_window_size",win_i24_o24_l1.total_window_size)

#Create TF datasets
# 24 x 1 x 1cls
targets=None

train_ds_win_i24_o24_l1 = win_i24_o24_l1.make_dataset(train_slice_win_i24_o24_l1)
val_ds_win_i24_o24_l1 = win_i24_o24_l1.make_dataset(train_slice_win_i24_o24_l1)
test_ds_win_i24_o24_l1 = win_i24_o24_l1.make_dataset(train_slice_win_i24_o24_l1)


# X 6 x 1 x 1
shift_size = 100 # value to start the next window splitting the total window size
window_size = win_i6_o1_l1.total_window_size # fullobject window size
train_df = mv_train
train_df = train_df.to_numpy(dtype=np.float32)
print("X 6 x 6 x 1 train_df.shape:",train_df.shape)
# slice windowsize into 3 window subset
train_slice_win_i6_o1_l1 = win_i6_o1_l1.slice_window(train_df, window_size, shift_size)
print("train_slice_win_i6_o1_l1.shape",train_slice_win_i6_o1_l1.shape)

#split window
inputs_train_slice_win_i6_o1_l1, labels_train_slice_win_i6_o1_l1 = win_i6_o1_l1.split_window(train_slice_win_i6_o1_l1)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {train_slice_win_i6_o1_l1.shape}')
print(f'inputs (Features: ind var X) shape: {inputs_train_slice_win_i6_o1_l1}')
print(f'Labels (Labels dep var y) shape: {labels_train_slice_win_i6_o1_l1}')
print("total_window_size=win_i6_o1_l1.total_window_size",win_i6_o1_l1.total_window_size)

#Create TF datasets
# 6 x 1 x 1cls
targets=None

train_ds_win_i6_o1_l1 = win_i6_o1_l1.make_dataset(train_slice_win_i6_o1_l1)
val_ds_win_i6_o1_l1 = win_i6_o1_l1.make_dataset(train_slice_win_i6_o1_l1)
test_ds_win_i6_o1_l1 = win_i6_o1_l1.make_dataset(train_slice_win_i6_o1_l1)


# +-------------------------------------------------------------------
# End Split the data into windows split into inputs and labels
# +-------------------------------------------------------------------
if mp_ml_windowmodel == '24_24_1':
    window = win_i24_o24_l1
    dswindow = train_ds_win_i24_o24_l1
    train_dataset = train_ds_win_i6_o1_l1
    val_dataset = val_ds_win_i6_o1_l1
    test_dataset = test_ds_win_i6_o1_l1
    
elif mp_ml_windowmodel == '6_1_1':
    window = win_i6_o1_l1
    dswindow = train_ds_win_i6_o1_l1
    train_dataset = train_ds_win_i6_o1_l1
    val_dataset = val_ds_win_i6_o1_l1
    test_dataset = test_ds_win_i6_o1_l1
    

print("mp_ml_windowmodel:", mp_ml_windowmodel,"window:", window, "dswindow:",dswindow )
print("train_dataset:", train_dataset, "val_dataset:", val_dataset, "test_dataset:", test_dataset)

# Print elements of dswindow
print("dswindow:", dswindow)
#read tensor spec
print("dswindow element_spec:")
for spec in dswindow.element_spec:
    print(f"Shape: {spec.shape}, Dtype: {spec.dtype}")

# Initialize input shapes with Xspec
amp_inputs = spec.shape if spec else None
amp_lstm_input_shape = spec.shape if spec else None
amp_cnn_input_shape = spec.shape if spec else None
amp_gru_input_shape = spec.shape if spec else None
amp_single_input_shape = spec.shape if spec else None
amp_transformer_input_shape = spec.shape if spec else None

bmp_inputs= amp_inputs[2],amp_inputs[3]
print("bmp_inputs:", bmp_inputs)

cmp_inputs= amp_inputs[0],amp_inputs[1],amp_inputs[2],amp_inputs[3]
print("cmp_inputs:", cmp_inputs)
 
# +-------------------------------------------------------------------
# Hyperparameter tuning and model setup
# +-------------------------------------------------------------------
# Define parameters for the model tuning process
#
# Select Model 
mp_cnn_model = True
mp_lstm_model = True
#plot = win_i24_o24_l1.plot(plot_col='close', model=None, max_subplots=3)
mp_gru_model = True
mp_transformer_model = True
mp_run_single_input_model = True
mp_run_single_input_submodels = False # not implemented yet    

# define inputs
mp_single_input_shape = amp_inputs
mp_lstm_input_shape = amp_inputs
mp_cnn_input_shape = amp_inputs
mp_gru_input_shape = amp_inputs
mp_transformer_input_shape = amp_inputs

# define features
mp_null = None
mp_single_features = 1
mp_lstm_features = 1
mp_cnn_features = 1
mp_gru_features = 1
mp_transformer_features = 1

# Hypermodel parameters
mp_activation1 = 'relu'
mp_activation2 = 'linear'
mp_activation3 = 'softmax'
mp_activation4 = 'sigmoid'     
mp_Hypermodel = 'HyperModel'
mp_objective = 'val_loss'
mp_max_epochs = mp_ml_tf_param_max_epochs 
mp_factor = 10
mp_seed = 42
mp_hyperband_iterations = 1
mp_tune_new_entries = False
mp_allow_new_entries = False
mp_max_retries_per_trial = 5
mp_max_consecutive_failed_trials = 6
# base tuner parameters
mp_validation_split = 0.2
mp_epochs = mp_ml_tf_param_epochs 
mp_ml_batch_size = mp_ml_batch_size  
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
mp_chk_patience = mp_ml_tf_param_chk_patience

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
if mp_env_test:
    mp_directory = f"tshybrid_ensemble_tuning_test"
    mp_project_name = "prjEquinox1_test"

# Create an instance of the tuner class
print("Creating an instance of the tuner class")
mt = CMdtuner(
    # Data
    traindataset=train_dataset,
    valdataset=val_dataset,
    testdataset=test_dataset,
    trainshape=train_shape,
    valshape=val_shape,
    testshape=test_shape,
    # Model
    inputs=bmp_inputs,
    cnn_model=mp_cnn_model,
    lstm_model=mp_lstm_model,
    gru_model=mp_gru_model,
    transformer_model=mp_transformer_model,
    run_single_input_model=mp_run_single_input_model,
    run_single_input_submodels=mp_run_single_input_submodels,
    objective=mp_objective,
    max_epochs=mp_ml_tf_param_max_epochs,
    min_epochs=mp_ml_tf_param_min_epochs,
    factor=mp_factor,
    seed=mp_seed,
    hyperband_iterations=mp_hyperband_iterations,
    tune_new_entries=mp_tune_new_entries,
    allow_new_entries=mp_allow_new_entries,
    max_retries_per_trial=mp_max_retries_per_trial,
    max_consecutive_failed_trials=mp_max_consecutive_failed_trials,
    validation_split=mp_validation_split,
    epochs=mp_epochs,
    batch_size=mp_ml_batch_size,
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
    step=mp_ml_tf_param_steps,
    multiactivate=True,
    tf1=False,
    tf2=False,
    tensorshape=mp_ml_tensor_shape,
    shape=mp_ml_input_shape,
    cnn_shape=mp_ml_input_cnn_shape,
    lstm_shape=mp_ml_input_lstm_shape,
    gru_shape=mp_ml_input_gru_shape,
    transformer_shape=mp_ml_input_transformer_shape,
    multi_inputs=mp_ml_multi_inputs,
)


# Run the tuner to find the best model configuration
print("Running Main call to tuner")
best_model = mt.tuner.get_best_models()
best_params = mt.tuner.get_best_hyperparameters(num_trials=1)[0]
best_model[0].summary()


#tensorboard command line
#tensorboard --logdir=<basepath>/logs


# +-------------------------------------------------------------------
# Scale the data
# +-------------------------------------------------------------------
scaler = StandardScaler()
mv_mv_train = scaler.fit_transform(train_ds_win_i24_o24_l1)
mv_mv_val = scaler.transform(val_ds_win_i24_o24_l1)
mv_mv_test = scaler.transform(test_ds_win_i24_o24_l1)

# +-------------------------------------------------------------------
# Train and evaluate the model
# +-------------------------------------------------------------------

best_model[0].fit(mv_mv_train,mv_mv_test, validation_split=mp_validation_split, epochs=mp_epochs, batch_size=mp_ml_batch_size)
best_model[0].evaluate(mv_mv_val, mv_mv_test)


# +-------------------------------------------------------------------
# Predict the test data using the trained model
# +-------------------------------------------------------------------
# Assuming mv_mv_train had 60 features, and the target is one of them (let's say the last one)
scaler = StandardScaler()
scaler.fit(mv_mv_train)  # Fit scaler on training data (number of seconds features)
predicted_fx_price = best_model[0].predict(mv_mv_test)

# If predicted_fx_price is only 1D, reshape to 2D
predicted_fx_price = predicted_fx_price.reshape(-1, 1)

# Inverse transform only the target column
predicted_fx_price = scaler.inverse_transform(
    np.hstack([np.zeros((predicted_fx_price.shape[0], mv_mv_train.shape[1] - 1)), predicted_fx_price])
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
"""