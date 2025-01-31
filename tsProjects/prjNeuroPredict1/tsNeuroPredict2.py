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
import onnxruntime as ort
import onnxruntime.backend as backend
import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer
import warnings
from numpy import concatenate
# Import equinox functionality
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlData import CMqldatasetup
from tsMqlML import CMqlmlsetup
from tsMqlSetup import tsMqlSetup
from tsMqlReference import CMqlTimeConfig
from tsMqlMLTune import initialize_tuner
from tsMqlMLTune import CMdtuner, CMdtunerHyperModel

MT5 =True
s1 = tsMqlSetup(loglevel='INFO', warn='ignore')
strategy = s1.get_computation_strategy()
def main():
   with strategy.scope():
      #Reference class
      tm = CMqlTimeConfig(basedatatime='SECONDS', loadeddatatime='MINUTES')
      MINUTES, HOURS, DAYS, TIMEZONE, TIMEFRAME, CURRENTYEAR, CURRENTDAYS, CURRENTMONTH = tm.get_current_time(tm)
      print("MINUTES:",MINUTES, "HOURS:",HOURS, "DAYS:",DAYS, "TIMEZONE:",TIMEZONE)
      print("CURRENTYEAR:",CURRENTYEAR, "CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH)
      TIMEFRAME = tm.TIME_CONSTANTS['TIMEFRAME']['H4'] # override as M1 needs checking
      mp_ml_data_type ='M1'

      # +-------------------------------------------------------------------
      # STEP: CBroker Login
      # +-------------------------------------------------------------------
      #Broker configuration
      broker = "METAQUOTES" # "ICM" or "METAQUOTES"
      mp_symbol_primary='EURUSD'
      MPDATAFILE1 =  "tickdata1.csv"
      MPDATAFILE2 =  "ratesdata1.csv"
      # initialise object
      c0=CMqlBrokerConfig(lpbroker=broker, mp_symbol_primary=mp_symbol_primary, MPDATAFILE1=MPDATAFILE1, MPDATAFILE2=MPDATAFILE1)
      broker_config, mp_symbol_primary, mp_symbol_secondary, mp_shiftvalue, mp_unit = c0.initialize_mt5(broker, tm)
      print("Broker Config:",broker_config)
      print("mp_symbol_primary:",mp_symbol_primary, "mp_symbol_secondary:",mp_symbol_secondary, "mp_shiftvalue:",mp_shiftvalue, "mp_unit:",mp_unit)
      if MT5:
            c1=c0.login_mt5(broker_config)
      
      BROKER = broker_config['BROKER']
      MPPATH = broker_config['MPPATH']
      MPBASEPATH = broker_config['MPBASEPATH']
      MPDATAPATH = broker_config['MPDATAPATH']
      MPFILEVALUE1 = broker_config['MPFILEVALUE1']
      MPFILEVALUE2 = broker_config['MPFILEVALUE2']
      MKFILES = broker_config['MKFILES']
      file_path = broker_config['MKFILES']
      print("BROKER:",BROKER, "MPPATH:",MPPATH, "MPBASEPATH:",MPBASEPATH, "MPDATAPATH:",MPDATAPATH, "MPFILEVALUE1:",MPFILEVALUE1, "MPFILEVALUE2:",MPFILEVALUE2, "MKFILES:",MKFILES)
      print(f"MQL file_path:" ,file_path)

      # +-------------------------------------------------------------------
      # STEP: Configuration settings
      # +-------------------------------------------------------------------
      # model api settings
      feature_scaler = MinMaxScaler()
      label_scaler = MinMaxScaler()
      # environment settings
      mp_ml_model_datapath = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsModelData/"
      mp_ml_directory = f"tshybrid_ensemble_tuning_prod"
      mp_ml_project_name = "prjEquinox1_prod.keras"
      mp_ml_baseuniq = str(1)# str(mp_random)
      mp_ml_base_path = os.path.join(mp_ml_model_datapath, mp_ml_directory,mp_ml_baseuniq)
      mp_ml_mbase_path = os.path.join(mp_ml_model_datapath, mp_ml_directory)
      mp_ml_checkpoint_filepath = posixpath.join(mp_ml_base_path, mp_ml_directory, mp_ml_project_name)
      # data load states
      mp_data_rownumber = False
      mp_data_show_dtype = False
      mp_data_loadapiticks = True
      mp_data_loadapirates = True
      mp_data_loadfileticks = True
      mp_data_loadfilerates = True
      # ml states
      mp_ml_shuffle = False
      mp_ml_cnn_model = True
      mp_ml_lstm_model = True
      mp_ml_gru_model = True
      mp_ml_transformer_model = True
      mp_ml_multi_inputs = False
      mp_ml_multi_inputs_preprocess = True
      mp_ml_multi_outputs = False
      mp_ml_multi_branches = True
      mp_ml_tunemode = True
      mp_ml_tunemodeepochs = True
      mp_ml_modelsummary = False
      mp_ml_hard_run= False 
      #model parameters
      mp_ml_loadtensor = True
      mp_ml_loadtemporian = False
      mp_ml_tensor_shape = False
      mp_ml_multiactivate=True

      """
      Time Series Forecasting: Features and Labels

      Features (X):
         - A sequence of input time steps used as features.
         - Sliding window approach is used to generate samples (batches).
         - Window size (e.g., `window_size = 24`) determines the number of past observations included in the input sequence.

      Labels (y):
         - Corresponding label values for the given input time steps.
         - Defines what the model is trying to predict for future steps.

      Forecast Label:
         - Typically, the `Close` price (or another relevant metric) for a future 24-hour period.

      Future Step Size:
         - 24 hours of data corresponds to 1406 minutes of forecast labels.

      Label Types:
         1. Sequence-to-Value:
            - The label `y` is only the final `Close` price after 24 hours.
            - Shape: `(1,)` (a single value prediction).
         
         2. Sequence-to-Sequence:
            - The label `y` is the entire series of `Close` prices for the next 24 hours.
            - Shape: `(1406, 1)` (scaled `Close` price for each minute).

   """

      mp_ml_custom_input_keyfeat = {'Close'} # the feature to predict
      mp_ml_custom_output_label = {'Label'} # the feature to predict
      mp_ml_custom_input_keyfeat_scaled = {feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat}  # the feature to predict
      mp_ml_custom_output_label_scaled = {targ + '_Scaled' for targ in mp_ml_custom_output_label}  # the label shifted to predict
      mp_ml_custom_output_label_count=len(mp_ml_custom_output_label)
      mp_ml_batch_size = 8
      #Splitting the data
      mp_ml_train_split = 0.7
      mp_ml_validation_split = 0.2
      mp_ml_test_split = 0.1
      #Best Models
      mp_ml_mp_ml_num_models = 4
      mp_ml_num_trials = 3
      mp_ml_steps_per_execution = 50
      # Set parameters for the Tensorflow keras model
      mp_ml_tf_param_steps = 10
      mp_ml_tf_param_max_epochs=100
      mp_ml_tf_param_min_epochs=1
      mp_ml_tf_param_epochs=mp_ml_tf_param_max_epochs
      mp_ml_tf_param_chk_patience = 3
      mp_ml_tf_shiftin=1
      mp_ml_tf_ma_windowin=14 # 14 DAYS typical indicator window
      # Set the shape of the data
      mp_ml_custom_input_shape=2 # mp_data_tab_rows, batches, timesteps, features
      mp_ml_custom_input_cnn_shape=2 # mp_data_tab_rows, batches, timesteps, features
      mp_ml_custom_input_lstm_shape=2 # mp_data_tab_rows, batches, timesteps, features
      mp_ml_custom_input_gru_shape=2 # mp_data_tab_rows, batches, timesteps, features
      mp_ml_custom_input_transformer_shape = 2 # mp_data_tab_rows, batches, timesteps, features
      mp_ml_cfg_period1=24 # 24 HOURS
      mp_ml_cfg_period2=6 # 6 HOURS
      mp_ml_cfg_period=1 # 1 HOURS
      mp_ml_unit_min = 32
      mp_ml_unit_max = 512
      mp_ml_unit_step = 32
      mp_ml_default_units = 128
      # setting dictionary for the model
      common_ml_params = {
            "lookahead_periods": mp_ml_cfg_period,
            "ma_window": mp_ml_tf_ma_windowin,
            "hl_avg_col": "HLAvg",
            "ma_col": "SMA",
            "returns_col": "LogReturns",
            "shift_in": mp_ml_tf_shiftin,
            "rownumber": mp_data_rownumber,
            "create_label": False,
       }

      other_ml_params = {
            "returns_col_scaled": "LogReturns_scaled",
      }
      mp_ml_return_col_scaled = other_ml_params["returns_col_scaled"]

      #Data variables
      mp_data_data_label = 3
      # 1: just the label, 2: label and features, 3:time label, features 4: full dataset
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
      d1 = CMqldatasetup(lp_features=mp_ml_custom_input_keyfeat, lp_label=mp_ml_custom_output_label, lp_label_count=mp_ml_custom_output_label_count)
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
                
      # Create labels
      mv_tdata1apiticks = d1.create_label_wrapper(
            df=mv_tdata1apiticks,
            bid_column="T1_Bid_Price",
            ask_column="T1_Ask_Price",
            column_in="T1_Bid_Price",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R1_Open",
            high_column="R1_High",
            low_column="R1_Low",
            close_column="R1_Close",
            run_mode=1,
            **common_ml_params
        )

      mv_tdata1apirates = d1.create_label_wrapper(
            df=mv_tdata1apirates,
            bid_column="R1_Bid_Price",
            ask_column="R1_Ask_Price",
            column_in="R1_Close",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R1_Open",
            high_column="R1_High",
            low_column="R1_Low",
            close_column="R1_Close",
            run_mode=2,
            **common_ml_params
        )

      mv_tdata1loadticks = d1.create_label_wrapper(
            df=mv_tdata1loadticks,
            bid_column="T2_Bid_Price",
            ask_column="T2_Ask_Price",
            column_in="T2_Bid_Price",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
            open_column="R2_Open",
            high_column="R2_High",
            low_column="R2_Low",
            close_column="R2_Close",
            run_mode=3,
            **common_ml_params
        )

      mv_tdata1loadrates = d1.create_label_wrapper(
            df=mv_tdata1loadrates,
            bid_column="R2_Bid_Price",
            ask_column="R2_Ask_Price",
            column_in="R2_Close",
            column_out1=list(mp_ml_custom_input_keyfeat)[0],
            column_out2=list(mp_ml_custom_output_label_scaled)[0],
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
      # Normalize the 'Close' column
      scaler = MinMaxScaler()
      mp_ml_custom_input_keyfeat_list = list(mp_ml_custom_input_keyfeat) 
      mp_ml_custom_input_keyfeat_scaled = [feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat_list]
      
      mv_tdata2[mp_ml_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_ml_custom_input_keyfeat_list])
      print("print Normalise")
      d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
      print("End of Normalise print")
      print("mv_tdata2.shape",mv_tdata2.shape)

      # +------------------------------------------------------------------
      # STEP: remove datetime dtype to numeric from the data
      # +-------------------------------------------------------------------
      #if len(mv_tdata2) > 0:
      #    mv_tdata2 = d1.wrangle_time(mv_tdata2, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=True, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=False)

      #d1.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "fancy_grid",floatfmt=".5f",numalign="left",stralign="left")
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
      # STEP: set the dataset to just the features and the label and sort by time
      # +-------------------------------------------------------------------
      if mp_data_data_label == 1:
            mv_tdata2 = mv_tdata2[[list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            d1.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")
            
      elif mp_data_data_label == 2:
            mv_tdata2 = mv_tdata2[[mv_tdata2.columns[0]] + [list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            # Ensure the data is sorted by time
            mv_tdata2 = mv_tdata2.sort_index()
            d1.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

      elif mp_data_data_label == 3:
            # Ensure the data is sorted by time use full dataset
            mv_tdata2 = mv_tdata2.sort_index()
            d1.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

      # +-------------------------------------------------------------------
      # STEP: Generate X and y from the Time Series
      # +-------------------------------------------------------------------
      # At thsi point the normalised data columns are split across the X and Y data
      m1 = CMqlmlsetup() # Create an instance of the class
      # 1: 24 HOURS/24 HOURS prediction window
      print("1: MINUTES data entries per time frame: MINUTES:", MINUTES, "HOURS:", MINUTES * 60, "DAYS:", MINUTES * 60 * 24)
      timeval = MINUTES * 60 # hours
      pasttimeperiods = 24
      futuretimeperiods = 24
      predtimeperiods = 1
      features_count = len(mp_ml_custom_input_keyfeat)  # Number of features in input
      labels_count = len(mp_ml_custom_output_label)  # Number of labels in output
      batch_size = mp_ml_batch_size

      print("timeval:",timeval, "pasttimeperiods:",pasttimeperiods, "futuretimeperiods:",futuretimeperiods, "predtimeperiods:",predtimeperiods)
      past_width = pasttimeperiods * timeval
      future_width = futuretimeperiods * timeval
      pred_width = predtimeperiods * timeval
      print("past_width:", past_width, "future_width:", future_width, "pred_width:", pred_width)

      # Create the input features (X) and label values (y)
      print("list(mp_ml_custom_input_keyfeat_scaled)", list(mp_ml_custom_input_keyfeat_scaled))

      # STEP: Create input (X) and label (Y) tensors Ensure consistent data shape
      # Create the input (X) and label (Y) tensors Close_scaled is the feature to predict and Close last entry in future the label
      mv_tdata2_X, mv_tdata2_y = m1.create_Xy_time_windows3(mv_tdata2, past_width, future_width, target_column=list(mp_ml_custom_input_keyfeat_scaled), feature_column=list(mp_ml_custom_input_keyfeat))
      print("mv_tdata2_X.shape", mv_tdata2_X.shape, "mv_tdata2_y.shape", mv_tdata2_y.shape)

      # Scale the Y labels
      mv_tdata2_y = scaler.transform(mv_tdata2_y.reshape(-1, 1))  # Transform Y values
      # +-------------------------------------------------------------------
      # STEP: Split the data into training and test sets Fixed Partitioning
      # +-------------------------------------------------------------------
      # Batch size alignment fit the number of rows as whole number divisible by the batch size to avoid float errors
      batch_size = mp_ml_batch_size
      precountX = len(mv_tdata2_X)
      precounty = len(mv_tdata2_y)
      mv_tdata2_X,mv_tdata2_y = m1.align_to_batch_size(mv_tdata2_X,mv_tdata2_y, batch_size)
      print(f"Aligned data: X shape: {mv_tdata2_X.shape}, Y shape: {mv_tdata2_y.shape}")

      # Check the number of rows
      print("Batch size alignment: mv_tdata2_X shape:", mv_tdata2_X.shape,"Precount:",precountX,"Postcount:",len(mv_tdata2_X))
      print("Batch size alignment: mv_tdata2_y shape:", mv_tdata2_y.shape,"Precount:",precounty,"Postcount:",len(mv_tdata2_y))

      # Split the data into training, validation, and test sets

      # STEP: Split data into training, validation, and test sets
      X_train, X_temp, y_train, y_temp = train_test_split(mv_tdata2_X,mv_tdata2_y, test_size=(mp_ml_validation_split + mp_ml_test_split), shuffle=False)
      X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(mp_ml_test_split / (mp_ml_validation_split + mp_ml_test_split)), shuffle=False)

      print(f"Training set: X_train: {X_train.shape}, y_train: {y_train.shape}")
      print(f"Validation set: X_val: {X_val.shape}, y_val: {y_val.shape}")
      print(f"Test set: X_test: {X_test.shape}, y_test: {y_test.shape}")
      
      # +-------------------------------------------------------------------
      # STEP: convert numpy arrays to TF datasets
      # +-------------------------------------------------------------------
      # initiate the object using a window generatorwindow is not  used in this model Parameters
      tf_batch_size = mp_ml_batch_size

      train_dataset = m1.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)
      val_dataset = m1.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)
      test_dataset = m1.create_tf_dataset(X_test, y_test, batch_size=tf_batch_size, shuffle=False)
      print(f"TF Datasets created: Train: {tf.data.experimental.cardinality(train_dataset).numpy()}, Val: {tf.data.experimental.cardinality(val_dataset).numpy()}, Test: {tf.data.experimental.cardinality(test_dataset).numpy()}")
  
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
      # +-------------------------------------------------------------------
      # STEP: Final shape summaries
      # +-------------------------------------------------------------------
      # Final summary of shapes
      # STEP: Confirm tensor shapes for the tuner
      input_shape = X_train.shape[1:]  # Shape of a single sample (time_steps, features)
      label_shape = y_train.shape[1:]  # Shape of a single label (future_steps)
      # Example data: shape = (num_samples, time_steps, features) cnn_data = np.random.random((1000, 1440, 1))  # 1000 samples, 1440 timesteps, 1 feature
      # Example data: labels = np.random.random((1000, 1))
      print(f"Full Input shape for model: {X_train.shape}, Label shape for model: {y_train.shape}")
      print(f"No Batch Input shape for model: {input_shape}, Label shape for model: {label_shape}")
      #batch components
      input_keras_batch=mp_ml_batch_size
      input_def_keras_batch=None
      # Get the input shape for the model
      input_rows_X=len(X_train)
      input_rows_y=len(y_train)
      input_batch_size=mp_ml_batch_size
      input_batches= X_train.shape[0]
      input_timesteps = X_train.shape[1]
      input_features = X_train.shape[2]
      # Get the output shape for the model
      output_label=y_train.shape[1]
      output_shape = y_train.shape
      output_features = y_train.shape[1]
      print(f"input_def_keras_batch  {input_def_keras_batch}, input_keras_batch: {input_keras_batch}")
      print(f"Input rows X: {input_rows_X},Input rows y: {input_rows_y} , Input batch_size {input_batch_size}, Input batches: {input_batches}, Input timesteps: {input_timesteps}, Input steps or features: {input_features}")
      print(f"Output label: {output_label}, Output shape: {output_shape}, Output features: {output_features}")
      # pass in the data shape for the model

      input_shape = (input_timesteps, input_features)  
      output_label_shape = (output_label, mp_ml_custom_output_label_count)
      print(f"Input shape for model: {input_shape}, Output shape for model: {output_label_shape}")
      # +-------------------------------------------------------------------
      # STEP: Tune best model Hyperparameter tuning and model setup
      # +-------------------------------------------------------------------
      base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsModelData/"
      project_name = "prjEquinox1_prod.keras"
      subdir = os.path.join(base_path, 'tshybrid_ensemble_tuning_prod', str(1))

      print("base_path:", base_path, "project_name:", project_name, "subdir:", subdir)

      t1 = CMqlmlsetup(
            input_shape=(input_timesteps, input_features),
            today_date=date.today().strftime('%Y-%m-%d %H:%M:%S'),
            random_seed=np.random.randint(0, 1000),
            base_path=base_path,
            project_name=project_name,
            subdir=subdir  # Use the already defined subdir instead of recomputing it
      )
      # +-------------------------------------------------------------------
      # STEP:Run the Tuner to find the best model configuration
      # +-------------------------------------------------------------------
      # Run the tuner to find the best model configuration Load hyperparameters
      h1 = CMdtunerHyperModel()
      hypermodel_params = h1.get_hypermodel_params()

      # Log the configuration
      log_config(hypermodel_params)

      # Initialize tuner
      mt = initialize_tuner(
            hypermodel_params=hypermodel_params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
      )

      ## Run the tuner to find the best model configuration
      print("Running Main call to tuner")
      mt.tuner.search_space_summary()
      # Check and load the model
      best_model = mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')
      print ("Best model loaded successfully evaluated as ",best_model)
      # If no model or hard run then run the search
      if best_model is None or mp_ml_hard_run:
         print("Running the tuner search")
         mt.run_search()
         print("Tuner search completed")
         print("Exporting the best model")
         mt.export_best_model(ftype='tf')
         print("Best model exported")
         # Reload the best model after exporting
         best_model = mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')
      else:
         print("Existing Best model loaded successfully.")

      # +-------------------------------------------------------------------
      # STEP: Train and evaluate the best model
      # +-------------------------------------------------------------------

      # Model Training
      print("Training the best model...")
      best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=mp_ml_tf_param_epochs,
                batch_size=mp_ml_batch_size
            )
      print("Training completed.")

      # Model Evaluation
      print("Evaluating the model...")
      val_metrics = best_model.evaluate(val_dataset, verbose=0)
      test_metrics = best_model.evaluate(test_dataset, verbose=0)
      print(f"Validation Metrics - Loss: {val_metrics[0]}, Accuracy: {val_metrics[1]}")
      print(f"Test Metrics - Loss: {test_metrics[0]}, Accuracy: {test_metrics[1]}")

      # Fit the label scaler on the training labels
      label_scaler.fit(y_train.reshape(-1, 1))

      # Predictions and Scaling
      print("Running predictions and scaling...")
      predicted_fx_price = best_model.predict(test_dataset)
      predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)

      real_fx_price = label_scaler.inverse_transform(y_test)
      print("Predictions and scaling completed.")

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
      # perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the label
      # variable. Negative values indicate poor model performance.
      # Check for NaN values and handle them
      if np.isnan(real_fx_price).any() or np.isnan(predicted_fx_price).any():
            print("Warning: NaN values found in input data. Handling NaNs by removing corresponding entries.")
            mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)
            real_fx_price = real_fx_price[mask]
            predicted_fx_price = predicted_fx_price[mask]
        
      mse = mean_squared_error(real_fx_price, predicted_fx_price)
      mae = mean_absolute_error(real_fx_price, predicted_fx_price)
      r2 = r2_score(real_fx_price, predicted_fx_price)
      print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
      print(f"Mean Squared Error: The lower the MSE, the better the model: {mse}")
      print(f"Mean Absolute Error: The lower the MAE, the better the model: {mae}")
      print(f"R2 Score: The closer to 1, the better the model: {r2}")

      plt.plot(real_fx_price, color='red', label='Real FX Price')
      plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
      plt.title('FX Price Prediction')
      plt.xlabel('Time')
      plt.ylabel('FX Price')
      plt.legend()
      plt.savefig(mp_ml_base_path + '/' + 'plot.png')
      plt.show()
      print("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')

      # +-------------------------------------------------------------------
      # STEP: Save model to ONNX
      # +-------------------------------------------------------------------

      # Save the model to ONNX format
      mp_output_path = mp_ml_data_path + "model_" + mp_symbol_primary + "_" + mp_ml_data_type + ".onnx"
      print(f"output_path: ",mp_output_path)
      onnx_model, _ = tf2onnx.convert.from_keras(best_model[0], opset=self.batch_size)
      onnx.save_model(onnx_model, mp_output_path)
      print(f"model saved to ",mp_output_path)
   
      #Assuming your model has a single input  Convert the model
      print("mp_inputs: ", mp_inputs)
      spec = mp_inputs.shape
      spec = (tf.TensorSpec(spec, tf.float32, name="input"),)
      print("spec: ", spec)
      # Convert the model to ONNX format
      opver = 17
      onnx_model = tf2onnx.convert.from_keras(best_model, input_signature=spec, output_path=mp_output_path, opset=opver)
      print("ONNX Runtime version:", ort.__version__)
      onnx.save_model(onnx_model, mp_output_path)
      print(f"model saved to ", mp_output_path)

      from onnx import checker 
      checker.check_model(best_model[0])
      # finish
      mt5.shutdown()
      print("Finished")

if __name__ == "__main__":
    main()