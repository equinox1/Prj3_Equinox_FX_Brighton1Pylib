# +------------------------------------------------------------------+
# |                                                 neuropredict2.py |
# |                                                    Tony Shepherd |
# |                                    https://www.xercescloud.co.uk |
# +------------------------------------------------------------------+
# property copyright "Tony Shepherd"
# property link      "https://www.xercescloud.co.uk"
# property version   "1.01"
# +------------------------------------------------------------------+
# STEP: Platform settings
# +-------------------------------------------------------------------
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config

pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

if pchk.mt5 is None or pchk.onnx is None:
    nomql = True
else:
    mt5 = pchk.mt5
    onnx = pchk.onnx
    nomql = False  # Ensure nomql is set to False if mt5 and onnx are not None

# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
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
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
import tensorflow as tf
import warnings
from numpy import concatenate
from tsMqlConnect import CMqlinit, CMqlBrokerConfig
from tsMqlML import CMqlmlsetup, CMqlWindowGenerator
from tsMqlMLTune import CMdtuner
from tsMqlReference import CMqlTimeConfig
from tsMqlSetup import CMqlSetup
from tsMqlMLTuneParams import CMdtunerHyperModel
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
from tsMqlSetup import CMqlEnvData, CMqlEnvML, CMqlEnvGlobal

# Setup the logging and tensor platform dependencies
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = obj1_CMqlSetup.get_computation_strategy()

# +-------------------------------------------------------------------
# STEP: switch values for the model
# +-------------------------------------------------------------------
broker = "METAQUOTES"  # "ICM" or "METAQUOTES"
mp_symbol_primary = 'EURUSD'
MPDATAFILE1 = "tickdata1.csv"
MPDATAFILE2 = "ratesdata1.csv"
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
TIMESAMPLE = 'M1'  # exception for api mql load

def main():
    with strategy.scope():
        # +-------------------------------------------------------------------
        # STEP: Load Reference class and time variables
        # +-------------------------------------------------------------------
        obj1_CMqlTimeConfig = CMqlTimeConfig(basedatatime='SECONDS', loadeddatatime='MINUTES')
        MINUTES, HOURS, DAYS, TIMEZONE, TIMEFRAME, CURRENTYEAR, CURRENTDAYS, CURRENTMONTH = obj1_CMqlTimeConfig.get_current_time(obj1_CMqlTimeConfig)
        print("MINUTES:", MINUTES, "HOURS:", HOURS, "DAYS:", DAYS, "TIMEZONE:", TIMEZONE)

        # Ensure TIME_CONSTANTS dictionary exists and has expected keys
        if 'SYMBOLS' in obj1_CMqlTimeConfig.TIME_CONSTANTS and obj1_CMqlTimeConfig.TIME_CONSTANTS['SYMBOLS']:
            mp_symbol_primary = str(obj1_CMqlTimeConfig.TIME_CONSTANTS['SYMBOLS'][0])
            print(f"mp_symbol_primary: {mp_symbol_primary}")
        else:
            raise ValueError("TIME_CONSTANTS['SYMBOLS'] is missing or empty")
        if 'TIMEFRAME' in obj1_CMqlTimeConfig.TIME_CONSTANTS and TIMESAMPLE in obj1_CMqlTimeConfig.TIME_CONSTANTS['TIMEFRAME']:
            TIMEFRAME = obj1_CMqlTimeConfig.TIME_CONSTANTS['TIMEFRAME'][TIMESAMPLE]
        else:
            raise ValueError("TIME_CONSTANTS['TIMEFRAME'][TIMESAMPLE] is missing")

        # Initialize environments
        environments = {
            "dataenv": CMqlEnvData(),
            "mlenv": CMqlEnvML(),
            "globalenv": CMqlEnvGlobal()
        }

        # Retrieve parameters safely
        params = {name: env.get_params() for name, env in environments.items()}

        # Print environments and their parameters
        for name, env in environments.items():
            print(f"{name}: {params[name]}")

        # +-------------------------------------------------------------------
        # STEP: CBroker Login
        # +-------------------------------------------------------------------
        obj1_CMqlBrokerConfig = CMqlBrokerConfig(broker, mp_symbol_primary, MPDATAFILE1, MPDATAFILE2)
        # Initialize MT5 connection
        broker_config, mp_symbol_primary, mp_symbol_secondary, mp_shiftvalue, mp_unit = obj1_CMqlBrokerConfig.initialize_mt5(broker, obj1_CMqlTimeConfig)

        # Attempt to log in to the broker
        broker_config = obj1_CMqlBrokerConfig.set_mql_broker()
        print("Broker Login Successful")

        # Retrieve broker file paths
        file_path = broker_config.get('MKFILES', 'Unknown Path')
        MPDATAPATH = broker_config.get('MPDATAPATH', 'Unknown Path')
        print(f"Broker File Path: {file_path}, MP Data Path: {MPDATAPATH}")

        # +-------------------------------------------------------------------
        # STEP: Data Preparation and Loading
        # +-------------------------------------------------------------------
        print("Data Preparation and Loading... Initializing Data Process...")
        obj1_CDataProcess = CDataProcess(environments["dataenv"], environments["mlenv"], environments["globalenv"])
        mp_data_history_size = environments["dataenv"].mp_data_history_size if hasattr(environments["dataenv"], 'mp_data_history_size') else 0
        print(f"CURRENTYEAR: {CURRENTYEAR}, CURRENTYEAR-mp_data_history_size: {CURRENTYEAR - mp_data_history_size}, CURRENTDAYS: {CURRENTDAYS}, CURRENTMONTH: {CURRENTMONTH}, TIMEZONE: {TIMEZONE}")

        # Set the UTC time for the data
        datenv = environments["dataenv"]
        mlenv = environments["mlenv"]
        globalenv = environments["globalenv"]
        obj2_CDataLoader = CDataLoader(datenv, mlenv, globalenv)
        mv_data_utc_from = obj2_CDataLoader.set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        mv_data_utc_to = obj2_CDataLoader.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        print(f"UTC From: {mv_data_utc_from}")
        print(f"UTC To: {mv_data_utc_to}")

        try:
            # Load tick data from MQL and FILE
            obj1_params = CDataLoader(
                datenv, mlenv, globalenv,
                api_ticks=environments["dataenv"].mp_data_loadapiticks,
                api_rates=environments["dataenv"].mp_data_loadapirates,
                file_ticks=environments["dataenv"].mp_data_loadfileticks,
                file_rates=environments["dataenv"].mp_data_loadfilerates,
                dfname1=environments["dataenv"].mv_data_dfname1,
                dfname2=environments["dataenv"].mv_data_dfname2,
                utc_from=mv_data_utc_from,
                symbol_primary=mp_symbol_primary,
                rows=environments["dataenv"].mp_data_rows,
                rowcount=environments["dataenv"].mp_data_rowcount,
                command_ticks=environments["dataenv"].mp_data_command_ticks,
                command_rates=environments["dataenv"].mp_data_command_rates,
                data_path=MPDATAPATH,
                file_value1=broker_config.get('MPFILEVALUE1', 'Unknown'),
                file_value2=broker_config.get('MPFILEVALUE2', 'Unknown'),
                timeframe=TIMEFRAME
            )

            try:
                mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = obj1_params.load_market_data(obj1_CDataProcess, obj1_params)
            except Exception as e:
                print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An error occurred in the outer try block: {e}")

        # Display the data
        obj1_CDataProcess = CDataProcess(environments["dataenv"], environments["mlenv"], environments["globalenv"])

        # Wrangle the data merging and transforming time to numeric
        if len(mv_tdata1apiticks) > 0 and not nomql:
            mv_tdata1apiticks = obj1_CDataProcess.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1apirates) > 0 and not nomql:
            mv_tdata1apirates = obj1_CDataProcess.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1loadticks) > 0:
            mv_tdata1loadticks = obj1_CDataProcess.wrangle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
        if len(mv_tdata1loadrates) > 0:
            mv_tdata1loadrates = obj1_CDataProcess.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)

        # Create labels
        if len(mv_tdata1apiticks) > 0 and not nomql:
            mv_tdata1apiticks = obj1_CDataProcess.create_label_wrapper(
                df=mv_tdata1apiticks,
                bid_column="T1_Bid_Price",
                ask_column="T1_Ask_Price",
                column_in="T1_Bid_Price",
                column_out1=environments["mlenv"].mp_ml_custom_input_keyfeat,  # Ensure this is a string
                column_out2=environments["mlenv"].mp_ml_custom_output_label_scaled,
                open_column="R1_Open",
                high_column="R1_High",
                low_column="R1_Low",
                close_column="R1_Close",
                run_mode=1,
                lookahead_periods=environments["mlenv"].mp_ml_cfg_period,
                ma_window=environments["mlenv"].mp_ml_tf_ma_windowin,
                hl_avg_col="HLAvg",
                ma_col="SMA",
                returns_col="LogReturns",
                shift_in=environments["mlenv"].mp_ml_tf_shiftin,
                rownumber=environments["dataenv"].mp_data_rownumber,
                create_label=False,
            )

        if len(mv_tdata1apirates) > 0 and not nomql:
            mv_tdata1apirates = obj1_CDataProcess.create_label_wrapper(
                df=mv_tdata1apirates,
                bid_column="R1_Bid_Price",
                ask_column="R1_Ask_Price",
                column_in="R1_Close",
                column_out1=environments["mlenv"].mp_ml_custom_input_keyfeat,  # Ensure this is a string
                column_out2=environments["mlenv"].mp_ml_custom_output_label_scaled,
                open_column="R1_Open",
                high_column="R1_High",
                low_column="R1_Low",
                close_column="R1_Close",
                run_mode=2,
                lookahead_periods=environments["mlenv"].mp_ml_cfg_period,
                ma_window=environments["mlenv"].mp_ml_tf_ma_windowin,
                hl_avg_col="HLAvg",
                ma_col="SMA",
                returns_col="LogReturns",
                shift_in=environments["mlenv"].mp_ml_tf_shiftin,
                rownumber=environments["dataenv"].mp_data_rownumber,
                create_label=False,
            )

        mv_tdata1loadticks = obj1_CDataProcess.create_label_wrapper(
            df=mv_tdata1loadticks,
            bid_column="T2_Bid_Price",
            ask_column="T2_Ask_Price",
            column_in="T2_Bid_Price",
            column_out1=environments["mlenv"].mp_ml_custom_input_keyfeat,  # Ensure this is a string
            column_out2=environments["mlenv"].mp_ml_custom_output_label_scaled,
            open_column="R2_Open",
            high_column="R2_High",
            low_column="R2_Low",
            close_column="R2_Close",
            run_mode=3,
            lookahead_periods=environments["mlenv"].mp_ml_cfg_period,
            ma_window=environments["mlenv"].mp_ml_tf_ma_windowin,
            hl_avg_col="HLAvg",
            ma_col="SMA",
            returns_col="LogReturns",
            shift_in=environments["mlenv"].mp_ml_tf_shiftin,
            rownumber=environments["dataenv"].mp_data_rownumber,
            create_label=False,
        )

        mv_tdata1loadrates = obj1_CDataProcess.create_label_wrapper(
            df=mv_tdata1loadrates,
            bid_column="R2_Bid_Price",
            ask_column="R2_Ask_Price",
            column_in="R2_Close",
            column_out1=environments["mlenv"].mp_ml_custom_input_keyfeat,  # Ensure this is a string
            column_out2=environments["mlenv"].mp_ml_custom_output_label_scaled,
            open_column="R2_Open",
            high_column="R2_High",
            low_column="R2_Low",
            close_column="R2_Close",
            run_mode=4,
            lookahead_periods=environments["mlenv"].mp_ml_cfg_period,
            ma_window=environments["mlenv"].mp_ml_tf_ma_windowin,
            hl_avg_col="HLAvg",
            ma_col="SMA",
            returns_col="LogReturns",
            shift_in=environments["mlenv"].mp_ml_tf_shiftin,
            rownumber=environments["dataenv"].mp_data_rownumber,
            create_label=False,
        )

        # Display the data
        if not nomql:
            obj1_CDataProcess.run_mql_print(mv_tdata1apiticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
            obj1_CDataProcess.run_mql_print(mv_tdata1apirates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        obj1_CDataProcess.run_mql_print(mv_tdata1loadticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        obj1_CDataProcess.run_mql_print(mv_tdata1loadrates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")

        # Copy the data for config selection
        if not nomql:
            data_sources = [mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates]
        else:
            data_sources = [mv_tdata1loadticks, mv_tdata1loadrates]

        # Create copies of the data
        data_copies = [data.copy() for data in data_sources]

        # Dynamically map available data sources
        data_keys = ['loadapiticks', 'loadapirates', 'loadfileticks', 'loadfilerates']
        data_mapping = {key: value for key, value in zip(data_keys, data_copies)}

        # Check if the selected configuration exists
        if mp_data_cfg_usedata in data_mapping:
            formatted_name = (
                mp_data_cfg_usedata.replace('load', '')
                .replace('api', 'API ')
                .replace('file', 'File ')
                .replace('ticks', 'Tick data')
                .replace('rates', 'Rates data')
            )
            print(f"Using {formatted_name}")
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
        mv_tdata2[mp_ml_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_ml_custom_input_keyfeat_list])

        print("print Normalise")
        obj1_CDataProcess = CDataProcess(dataenv, mlenv, globalenv)
        print("Type after wrangle_time:", type(mv_tdata2))
        obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        print("End of Normalise print")
        print("mv_tdata2.shape", mv_tdata2.shape)

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