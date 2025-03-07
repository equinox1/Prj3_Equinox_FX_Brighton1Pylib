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
from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
from tsMqlPlatform import run_platform, platform_checker

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

if os_platform == 'Windows':
    if pchk.mt5 is None or pchk.onnx is None:
        raise RuntimeError("MetaTrader5 or ONNX dependencies are missing.")
    else:
        mt5 = pchk.mt5
        onnx = pchk.onnx
else:
    nomql = True

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
import warnings
from numpy import concatenate
# Import equinox functionality
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlML import CMqlmlsetup, CMqlWindowGenerator
from tsMqlMLTune import CMdtuner
from tsMqlReference import CMqlTimeConfig
from tsMqlSetup import CMqlSetup
from tsMqlMLTuneParams import CMdtunerHyperModel
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlSetup import CMqlEnvData
from tsMqlSetup import CMqlEnvML
from tsMqlSetup import CMqlEnvGlobal

# Import MetaTrader5
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore')
strategy = obj1_CMqlSetup.get_computation_strategy()

tfdebug = False

if tfdebug:
    tf.debugging.set_log_device_placement(True)
    tf.config.experimental_run_functions_eagerly(True)
    tf.config.run_functions_eagerly(True)
    tf.config.optimizer.set_jit(True)
    # List available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs Available:", gpus)

    # Enable memory growth to avoid sudden crashes
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.debugging.set_log_device_placement(True)

    if tf.config.list_physical_devices('GPU'):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print("Current GPU Memory Usage:", memory_info)
    import psutil
    print("RAM Usage:", psutil.virtual_memory().used / 1e9, "GB")

    import gc
    tf.keras.backend.clear_session()
    gc.collect()


from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

_command_ticksdef main():
    with strategy.scope():
      
        mp_ml_show_plot=False
        ONNX_save=False
        mp_ml_hard_run= True
        mp_ml_tunemode = True
        mp_ml_tunemodeepochs = True
        mp_data_rows = 2000 # number of mp_data_tab_rows to fetch
        mp_data_rowcount = 10000 # number of mp_data_tab_rows to fetch
        mp_ml_Keras_tuner = 'hyperband' # 'hyperband' or 'randomsearch' or 'bayesian' or 'skopt' or 'optuna'
        mp_ml_batch_size = 4
        # scaling
        all_modelscale = 2 # divide the model by this number
        cnn_modelscale = 2 # divide the model by this number
        lstm_modelscale = 2 # divide the model by this number
        gru_modelscale = 2 # divide the model by this number
        trans_modelscale = 2 # divide the model by this number
        transh_modelscale = 1 # divide the model by this number
        transff_modelscale = 4 # divide the model by this number
        dense_modelscale = 2 # divide the model by this number

    
        
        tm = CMqlTimeConfig(basedatatime='SECONDS', loadeddatatime='MINUTES')
        MINUTES, HOURS, DAYS, TIMEZONE, TIMEFRAME, CURRENTYEAR, CURRENTDAYS, CURRENTMONTH = tm.get_current_time(tm)
        print("CURRENTYEAR:",CURRENTYEAR, "CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH)
        #Main code
        MINUTES = int(tm.get_timevalue('MINUTES'))
        HOURS = int(tm.get_timevalue('HOURS'))
        DAYS = int(tm.get_timevalue('DAYS'))

        TIMEZONE = tm.TIME_CONSTANTS['TIMEZONES'][0]
        TIMEFRAME = tm.TIME_CONSTANTS['TIMEFRAME']['H4']

        mp_ml_data_type ='obj1_Mqlmlsetup'
        #MQL constants
        broker = "METAQUOTES" # "ICM" or "METAQUOTES"
        
        mp_symbol_primary = tm.TIME_CONSTANTS['SYMBOLS'][0]
        mp_symbol_secondary = tm.TIME_CONSTANTS['SYMBOLS'][1]
        mp_shiftvalue = tm.TIME_CONSTANTS['DATATYPE']['MINUTES']
        mp_unit = tm.TIME_CONSTANTS['UNIT'][1] 
        print("mp_symbol_primary:",mp_symbol_primary, "mp_symbol_secondary:",mp_symbol_secondary, "mp_shiftvalue:",mp_shiftvalue, "mp_unit:",mp_unit)
        MPDATAFILE1 =  "tickdata1.csv"
        MPDATAFILE2 =  "ratesdata1.csv"

        obj1_CMqlBrokerConfig = CMqlBrokerConfig(broker, mp_symbol_primary, MPDATAFILE1, MPDATAFILE2)
        broker_config = obj1_CMqlBrokerConfig.set_mql_broker()
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
        obj1_CMqlinit= CMqlinit(
            MPPATH=broker_config["MPPATH"],
            MPLOGIN=MPLOGIN,
            MPPASS=MPPASS,
            MPSERVER=broker_config["MPSERVER"],
            MPTIMEOUT=broker_config["MPTIMEOUT"],
            MPPORTABLE=broker_config["MPPORTABLE"],
            MPENV=broker_config["MPENV"]
        )
        if not obj1_CMqlinit.run_mql_login():
            raise ConnectionError("Failed to login to MT5 terminal")
        print("Terminal Info:", mt5.terminal_info())

        terminal_info = mt5.terminal_info()
        print(terminal_info)
        file_path=terminal_info.data_path +r"/MQL5/Files/"
        print(f"MQL file_path:" ,file_path)

        #data_path to save model
        mp_ml_data_path=file_path
        print(f"data_path to save onnx model: ",mp_ml_data_path)
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
        #mp_ml_subdir = os.path.join(mp_ml_base_path, mp_ml_directory, str(1))
        mp_ml_checkpoint_filepath = posixpath.join(mp_ml_base_path, mp_ml_directory, mp_ml_project_name)

        # data load states
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
        mp_ml_multi_inputs = False
        mp_ml_multi_inputs_preprocess = True
        mp_ml_multi_outputs = False
        mp_ml_multi_branches = True
        mp_ml_modelsummary = False
        
        #model parameters

        #Machine Learning (ML) variables
        mp_ml_loadtensor = True
        mp_ml_loadtemporian = False
        mp_ml_tensor_shape = False
        mp_ml_multiactivate=True

        #Features and label and label definitions
        #X (features): A sequence of input time steps.
        #y (labels): The label value(s) for the corresponding time step(s).
        #For time windows: #Choose a fixed window size (e.g.window_size = 24) Slide the window across the dataset to create samples( Batches)
        #Features:(X): A series of consecutive time steps as input features.
        #Label:   (Y): The label value(s) for the future time step(s) you want to predict.

        #Forecast Label: Close price (or another label metric) for the future 24-hour period.
        #Future Step Size: 24 hours of data, which equals 1406 minutes of forecast labels.
        #Sequence-to-value: If the label Y is only the final Close price after 24 hours, the label shape is:(1,) (a single value). 
        #Sequence to Sequence: If the label is the entire series of Close prices for the future 24 hours, the label shape is:(1406, 1) (scaled Close price for each min).

        mp_ml_custom_input_keyfeat = {'Close'} # the feature to predict
        mp_ml_custom_output_label = {'Label'} # the feature to predict
        mp_ml_custom_input_keyfeat_scaled = {feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat}  # the feature to predict
        mp_ml_custom_output_label_scaled = {targ + '_Scaled' for targ in mp_ml_custom_output_label}  # the label shifted to predict
        mp_ml_custom_output_label_count=len(mp_ml_custom_output_label)
        

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
        mp_ml_tf_param_epochs=1
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
        mv_data_dfname1 = "df_rateobj1_CMqlSetup"
        mv_data_dfname2 = "df_rates2"
       
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
        obj1_CDataLoader = CDataLoader(lp_features=mp_ml_custom_input_keyfeat, lp_label=mp_ml_custom_output_label, lp_label_count=mp_ml_custom_output_label_count)
        print("CURRENTYEAR:",CURRENTYEAR, "CURRENTYEAR-mp_data_history_size",CURRENTYEAR-mp_data_history_size,"CURRENTDAYS:",CURRENTDAYS, "CURRENTMONTH:",CURRENTMONTH,"TIMEZONE:",TIMEZONE)
        mv_data_utc_from = obj1_CDataLoader.set_mql_timezone(CURRENTYEAR-mp_data_history_size, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        mv_data_utc_to = obj1_CDataLoader.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        print("UTC From:",mv_data_utc_from)
        print("UTC To:",mv_data_utc_to)

        #params model from refactor mode
        # model api settings
        dataenv = CMqlEnvData()
        mlenv = CMqlEnvML()
        globalenv = CMqlEnvGlobal()

        data_params = dataenv.get_params()
        ml_params = mlenv.get_params()
        global_params = globalenv.get_params()

        # Load tick data from MQL and FILE
        params_obj = CDataLoader(
            api_ticks=mp_data_loadapiticks,
            api_rates=mp_data_loadapirates,
            file_ticks=mp_data_loadfileticks,
            file_rates=mp_data_loadfilerates,
            dfname1=mv_data_dfname1,
            dfname2=mv_data_dfname2,
            utc_from=mv_data_utc_from,
            symbol_primary=mp_symbol_primary,
            rows=mp_data_rows,
            rowcount=mp_data_rowcount,
            command_ticks=mp_data_command_ticks,
            command_rates=mp_data_command_rates,
            data_path=MPDATAPATH,
            file_value1=MPFILEVALUE1,
            file_value2=MPFILEVALUE2,
            timeframe=TIMEFRAME
        )

        mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = obj1_CDataLoader.load_market_data()
       
       

        obj1_CDataProcess = CDataProcess(dataenv,mlenv,globalenv)
        #wrangle the data merging and transforming time to numeric
        if len(mv_tdata1apiticks) > 0:  
            mv_tdata1apiticks = obj1_CDataProcess.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1apirates) > 0:
            mv_tdata1apirates = obj1_CDataProcess.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1loadticks) > 0:
            mv_tdata1loadticks = obj1_CDataProcess.wrangle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
        if len(mv_tdata1loadrates) > 0:
            mv_tdata1loadrates = obj1_CDataProcess.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
                
        # Create labels
        mv_tdata1apiticks = obj1_CDataProcess.create_label_wrapper(
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

        mv_tdata1apirates = obj1_CDataProcess.create_label_wrapper(
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

        mv_tdata1loadticks = obj1_CDataProcess.create_label_wrapper(
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

        mv_tdata1loadrates = obj1_CDataProcess.create_label_wrapper(
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
        obj1_CDataProcess.run_mql_print(mv_tdata1apiticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
        obj1_CDataProcess.run_mql_print(mv_tdata1apirates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
        obj1_CDataProcess.run_mql_print(mv_tdata1loadticks,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
        obj1_CDataProcess.run_mql_print(mv_tdata1loadrates,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")

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
        # STEP: Normalize the X data
        # +-------------------------------------------------------------------
        # Normalize the 'Close' column
        scaler = MinMaxScaler()
        mp_ml_custom_input_keyfeat_list = list(mp_ml_custom_input_keyfeat) 
        mp_ml_custom_input_keyfeat_scaled = [feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat_list]
        mv_tdata2[mp_ml_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2mp_ml_custom_input_keyfeat_list])
        print("print Normalise")
        obj1_CDataProcess= CDataProcess(dataenv,mlenv,globalenv)
        print("Type after wrangle_time:", type(mv_tdata2))
        obj1_CDataProcess.run_mql_print(mv_tdata2,mp_data_tab_rows,mp_data_tab_width, "plain",floatfmt=".5f",numalign="left",stralign="left")
        print("End of Normalise print")
        print("mv_tdata2.shape",mv_tdata2.shape)

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

        # Set the first column as index which is the datetime column
        mv_tdata2.set_index(first_column, inplace=True)
        mv_tdata2=mv_tdata2.dropna()
        print("POST INDEX: Count: ",len(mv_tdata2))

        # +-------------------------------------------------------------------
        # STEP: set the dataset to just the features and the label and sort by time
        # +-------------------------------------------------------------------
        print("Type before set default:", type(mv_tdata2))
        if mp_data_data_label == 1:
            mv_tdata2 = mv_tdata2[[list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            obj1_CDataLoader.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")
            
        elif mp_data_data_label == 2:
            mv_tdata2 = mv_tdata2[[mv_tdata2.columns[0]] + [list(mp_ml_custom_input_keyfeat_scaled)[0]]]
            # Ensure the data is sorted by time
            mv_tdata2 = mv_tdata2.sort_index()
            obj1_CDataLoader.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

        elif mp_data_data_label == 3:
            # Ensure the data is sorted by time use full dataset
            mv_tdata2 = mv_tdata2.sort_index()
            print("Count of Tdata2:",len(mv_tdata2))
            obj1_CDataLoader.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

       # +-------------------------------------------------------------------
        # STEP: Generate X and y from the Time Series
        # +-------------------------------------------------------------------
        # At thsi point the normalised data columns are split across the X and Y data
        obj1_Mqlmlsetup = CMqlmlsetup() # Create an instance of the class
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
        mv_tdata2_X, mv_tdata2_y = obj1_Mqlmlsetup.create_Xy_time_windows3(mv_tdata2, past_width, future_width, target_column=list(mp_ml_custom_input_keyfeat_scaled), feature_column=list(mp_ml_custom_input_keyfeat))
        print("mv_tdata2_X.shape", mv_tdata2_X.shape, "mv_tdata2_y.shape", mv_tdata2_y.shape)
        
        # +-------------------------------------------------------------------
        # STEP: Normalize the Y data
        # +-------------------------------------------------------------------
        # Scale the Y labels
        mv_tdata2_y = scaler.transform(mv_tdata2_y.reshape(-1, 1))  # Transform Y values
                
        # +-------------------------------------------------------------------
        # STEP: Split the data into training and test sets Fixed Partitioning
        # +-------------------------------------------------------------------
        # Batch size alignment fit the number of rows as whole number divisible by the batch size to avoid float errors
        batch_size = mp_ml_batch_size
        precountX = len(mv_tdata2_X)
        precounty = len(mv_tdata2_y)
        mv_tdata2_X,mv_tdata2_y = obj1_Mqlmlsetup.align_to_batch_size(mv_tdata2_X,mv_tdata2_y, batch_size)
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

        train_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)
        val_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)
        test_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_test, y_test, batch_size=tf_batch_size, shuffle=False)
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
        # Hyperparameter configuration
        def get_hypermodel_params():
            today_date = date.today().strftime('%Y-%m-%d %H:%M:%S')
            random_seed = np.random.randint(0, 1000)
            base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/PythonLib/tsModelData/"
            project_name = "prjEquinox1_prod.keras"
            subdir = os.path.join(base_path, 'tshybrid_ensemble_tuning_prod', str(1))

            # Ensure the directory exists
            os.makedirs(subdir, exist_ok=True)

            return {
                'objective': 'val_loss',
                'max_epochs': mp_ml_tf_param_max_epochs,
                'factor': 3,
                'seed': 42,
                'hyperband_iterations': 1,
                'tune_new_entries':True,
                'allow_new_entries': True,
                'max_retries_per_trial': 0,
                'max_consecutive_failed_trials': 3,
                'validation_split': 0.2,
                'epochs': mp_ml_tf_param_epochs,
                'batch_size': mp_ml_batch_size,
                'dropout': 0.2,
                'optimizer': 'adam',
                'loss': 'mean_squared_error',
                'metrics': 'mean_squared_error',
                'directory': subdir,
                'logger': None,
                'tuner_id': None,
                'overwrite': True,
                'executions_per_trial': 1,
                'chk_fullmodel': True,
                'chk_verbosity': 0,
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
                'unitmin': mp_ml_unit_min,
                'unitmax': mp_ml_unit_max,
                'unitstep': mp_ml_unit_step,
                'defaultunits': mp_ml_default_units,
                'num_trials': mp_ml_num_trials,
                'keras_tuner': mp_ml_Keras_tuner, 
                'all_modelscale': all_modelscale,
                'cnn_modelscale': cnn_modelscale,
                'lstm_modelscale': lstm_modelscale,
                'gru_modelscale': gru_modelscale,
                'trans_modelscale': trans_modelscale,
                'transh_modelscale': transh_modelscale,
                'transff_modelscale': transff_modelscale,
                'dense_modelscale': dense_modelscale

            }

        # Print configuration details for logging
        def log_config(hypermodel_params):
            print(f"mp_ml_random: {hypermodel_params['random']}")
            print(f"mp_ml_today: {hypermodel_params['today']}")
            print(f"mp_checkpoint_filepath: {hypermodel_params['checkpoint_filepath']}")

        # Initialize the tuner class
        def initialize_tuner(hypermodel_params, train_dataset, val_dataset, test_dataset):
            try:
                print("Creating an instance of the tuner class")
                mt = CMdtuner(
                    # tf datasets
                    traindataset=train_dataset,
                    valdataset=val_dataset,
                    testdataset=test_dataset,
                    # Model selection
                    cnn_model=mp_ml_cnn_model,
                    lstm_model=mp_ml_lstm_model,
                    gru_model=mp_ml_gru_model,
                    transformer_model=mp_ml_transformer_model,
                    multiactivate=True,
                    # Model inputs directly from the data traindataset shape
                    data_input_shape=input_shape,
                    # Model inputs from the shape selection options 1-4
                    main_custom_shape_selector=mp_ml_custom_input_shape,
                    cnn_custom_shape_selector=mp_ml_custom_input_cnn_shape,
                    lstm_custom_shape_selector=mp_ml_custom_input_lstm_shape,
                    gru_custom_shape_selector=mp_ml_custom_input_gru_shape,
                    transformer_custom_shape_selector=mp_ml_custom_input_transformer_shape,
                    # Use merge of different shapes in final input output
                    multi_inputs=mp_ml_multi_inputs,
                    multi_outputs=mp_ml_multi_outputs,
                    multi_branches=mp_ml_multi_branches,
                    #Logging
                    tf1=False,
                    tf2T=False,
                    # Model parameters and hypermodel params
                    step=mp_ml_tf_param_steps,
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
                    tunemode =  mp_ml_tunemode,
                    tunemodeepochs = mp_ml_tunemodeepochs,
                    modelsummary = mp_ml_modelsummary,
                    unitmin=hypermodel_params['unitmin'],
                    unitmax=hypermodel_params['unitmax'],
                    unitstep=hypermodel_params['unitstep'],
                    defaultunits=hypermodel_params['defaultunits'],
                    num_trials=hypermodel_params['num_trials'],
                    steps_per_execution=mp_ml_steps_per_execution,
                    keras_tuner=hypermodel_params['keras_tuner'],
                    all_modelscale=hypermodel_params['all_modelscale'],
                    cnn_modelscale=hypermodel_params['cnn_modelscale'],
                    lstm_modelscale=hypermodel_params['lstm_modelscale'],
                    gru_modelscale=hypermodel_params['gru_modelscale'],
                    trans_modelscale = hypermodel_params['trans_modelscale'],
                    transh_modelscale=hypermodel_params['transh_modelscale'],
                    transff_modelscale=hypermodel_params['transff_modelscale'],
                    dense_modelscale=hypermodel_params['dense_modelscale']  

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
        mt = initialize_tuner(
            hypermodel_params=hypermodel_params,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )

        print("Running Main call to tuner")

        # Check and load the model
        best_model = mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')

        # If no model or a hard run is required, run the search
        runtuner = False

        if best_model is None:
            print("Running the tuner search bo model")
            runtuner = mt.run_search()
            mt.export_best_model(ftype='tf')
        elif mp_ml_hard_run:  
            print("Running the tuner search hard run")
            runtuner = mt.run_search()
            mt.export_best_model(ftype='tf')
        else:
            print("Best model loaded successfully")
            runtuner = True

        

        # +-------------------------------------------------------------------
        # STEP: Train and evaluate the best model
        # +-------------------------------------------------------------------
        print("Model: Loading file from directory", mp_ml_mbase_path,"Model: filename", mp_ml_project_name)   
        if (load_model := mt.check_and_load_model(mp_ml_mbase_path, ftype='tf')) is not None:
            best_model = load_model
            print("Model: Best model: ", best_model.name)
            print("best_model.summary()", best_model.summary())
            
    
            # Fit the label scaler on the training labels
        
            # Model Training
            tf.keras.backend.clear_session(free_memory=True)

            print("Training the best model...")
            best_model.fit(
                train_dataset,
                validation_data=val_dataset,
                batch_size=mp_ml_batch_size,
                epochs=mp_ml_tf_param_epochs
            )
            print("Training completed.")
        
            # Model Evaluation
            print("Evaluating the model...")
            val_metrics = best_model.evaluate(val_dataset, verbose=1)
            test_metrics = best_model.evaluate(test_dataset, verbose=1)

            print(f"Validation Metrics - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")
            print(f"Test Metrics - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}")

            # Fit the label scaler on the training labels
            label_scaler.fit(y_train.reshape(-1, 1))

            # Predictions and Scaling
            print("Running predictions and scaling...")
            predicted_fx_price = best_model.predict(test_dataset)
            predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)

            real_fx_price = label_scaler.inverse_transform(y_test.reshape(-1, 1))

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
            if mp_ml_show_plot:
                plt.show()
            print("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')

            if ONNX_save:
                # Save the model to ONNX
            
                # Define the output path
                mp_output_path = mp_ml_data_path + f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx"
                print(f"Output Path: {mp_output_path}")

                # Convert Keras model to ONNX
                opset_version = 17  # Choose an appropriate ONNX opset version

                # Assuming your model has a single input
                spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)

                # Save the ONNX model
                onnx.save_model(onnx_model, mp_output_path)
                print(f"Model saved to {mp_output_path}")

                # Verify the ONNX model
                checker.check_model(onnx_model)
                print("ONNX model is valid.")

                # Check ONNX Runtime version
                print("ONNX Runtime version:", ort.__version__)

            # finish
            mt5.shutdown()
            print("Finished")
        else:
            print("No data loaded")
            mt5.shutdown()
            print("Finished")
        
if __name__ == "__main__":
    main()