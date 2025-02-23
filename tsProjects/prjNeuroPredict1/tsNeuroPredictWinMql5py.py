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
# gpu and tensor platform
from tsMqlSetup import CMqlSetup

import logging
# Initialize logger

logger = logging.getLogger("Main")
logging.basicConfig(level=logging.INFO)

from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")

# +-------------------------------------------------------------------
# STEP: Import standard Python packages
# +-------------------------------------------------------------------
# System packages
import os
import pathlib
from pathlib import Path
import json
from datetime import datetime, date
import pytz

# Data packages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Machine Learning packages
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# set package options
feature_scaler = MinMaxScaler()
label_scaler = MinMaxScaler()
# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr
#Reference class
from tsMqlReference import CMqlRefConfig
# Equinox sub packages
from tsMqlConnect import CMqlBrokerConfig
from tsMqlDataLoader import CDataLoader
from tsMqlDataProcess import CDataProcess
# Equinox ML packages
from tsMqlMLTune import CMdtuner

# Setup the logging and tensor platform dependencies
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = obj1_CMqlSetup.get_computation_strategy()
# format values
mp_data_tab_rows = 2
mp_data_tab_width = 30

# +-------------------------------------------------------------------
# STEP: End of driving parameters
# +-------------------------------------------------------------------

def main(logger):
    with strategy.scope():

      #-----------------------------------------------------------------
        
        # +----------------------------------------------------------
        # STEP: setup environment
        # +----------------------------------------------------------
        # Usage
        # env = EnvManager(custom_params={"ml": {"epochs": 20}})
        # print("All Parameters:", env.all_params())
        # print("ML Epochs:", env.get_param("ml", "epochs"))
        # print("Base Learning Rate:", env.get_param("base", "learning_rate"))
        # Override parameters
        # env.override_params({"base": {"learning_rate": 0.005}})
        # print("Updated Learning Rate:", env.get_param("base", "learning_rate"))
        env = CMqlEnvMgr()
        print("All Parameters:", env.all_params())
        # get environment parameters
        params= env.all_params()

        base_params = env.all_params()["base"]
        data_params = env.all_params()["data"]
        ml_params = env.all_params()["ml"]
        mltune_params = env.all_params()["mltune"]
        app_params = env.all_params()["app"]

        print("base_params:", base_params)
        print("data_params:", data_params)
        print("ml_params:", ml_params)
        print("ml_tune_params:", mltune_params)
        print("app_params:", app_params)

        # STEP: Load Reference class and time variables
        # +-------------------------------------------------------------------
        obj1_CMqlRefConfig = CMqlRefConfig(basedatatime='SECONDS', loadeddatatime='MINUTES', timesample='M1')
        timevalue, MINUTES, HOURS, DAYS, TIMEZONE, TIMEFRAME, CURRENTYEAR, CURRENTDAYS, CURRENTMONTH, PRIMARY_SYMBOL = obj1_CMqlRefConfig.run_service()
        logger.info(f"MINUTES: {MINUTES}, HOURS: {HOURS}, DAYS: {DAYS}, TIMEZONE: {TIMEZONE} , TIMEFRAME: {TIMEFRAME}, CURRENTYEAR: {CURRENTYEAR}, CURRENTDAYS: {CURRENTDAYS}, CURRENTMONTH: {CURRENTMONTH}, PRIMARY_SYMBOL: {PRIMARY_SYMBOL}")
        
        env.override_params({"data": {"mp_data_timeframe": 'H4'}})
        logger.info(f"Override update: {env.get_param('data', 'mp_data_timeframe')}")
        TIMEFRAME = env.get_param('data', 'mp_data_timeframe')
        # +-------------------------------------------------------------------
        # STEP: CBroker Login
        # +-------------------------------------------------------------------
        print("PARAM HEADER: MP_APP_BROKER:", app_params.get('mp_app_broker'))
        broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
        mqqlobj = broker_config.run_mql_login()
        if mqqlobj is True:
            print("Successfully logged in to MetaTrader 5.")
        else:
            print(f"Failed to login. Error code: {mqqlobj}")

        # +-------------------------------------------------------------------
        # STEP: Data Preparation and Loading
        # +-------------------------------------------------------------------
        # Retrieve broker file paths
       
        obj1_CDataLoader = CDataLoader()
        obj1_CDataProcess = CDataProcess()

        # +-------------------------------------------------------------------
        # STEP: Data Preparation and Loading
        # +-------------------------------------------------------------------
        # Set the data history size
        mp_data_history_size = data_params.get('mp_data_history_size')
        # Set the UTC time for the data        # Set the UTC time for the data
        mv_data_utc_from = obj1_CDataLoader.set_mql_timezone(CURRENTYEAR - mp_data_history_size, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        mv_data_utc_to = obj1_CDataLoader.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        logger.info(f"Main:UTC From: {mv_data_utc_from}")
        logger.info(f"Main:UTC To: {mv_data_utc_to}")
        

        try:
             mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = obj1_CDataLoader.load_data(
             lp_utc_from=mv_data_utc_from, 
             lp_utc_to=mv_data_utc_to,
             lp_timeframe=TIMEFRAME,
             lp_app_primary_symbol='EURUSD',
             lp_app_rows=100
             )
        except Exception as e:
            logger.error(f"An error occurred: {e}")

       # print("mv_tdata1apiticks", mv_tdata1apiticks.Head(3))
       # print("mv_tdata1apirates", mv_tdata1apirates.Head(3))
       # print("mv_tdata1loadticks", mv_tdata1loadticks.Head(3))
       # print("mv_tdata1loadrates", mv_tdata1loadrates.Head(3))






"""
        # Wrangle the data merging and transforming time to numericforming time to numeric
        if len(mv_tdata1apiticks) > 0 and loadmql:
            mv_tdata1apiticks = obj1_CDataProcess.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)ataProcess.wrangle_time(mv_tdata1apiticks, mp_unit, mp_filesrc="ticks1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1apirates) > 0 and loadmql:irates) > 0 and loadmql:
            mv_tdata1apirates = obj1_CDataProcess.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)aProcess.wrangle_time(mv_tdata1apirates, mp_unit, mp_filesrc="rates1", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=False, mp_convert=False, mp_drop=True)
        if len(mv_tdata1loadticks) > 0 and (loadmql == True or loadmql == False):oadmql == True or loadmql == False):
            mv_tdata1loadticks = obj1_CDataProcess.wrangle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)ngle_time(mv_tdata1loadticks, mp_unit, mp_filesrc="ticks2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)
        if len(mv_tdata1loadrates) > 0 and (loadmql == True or loadmql == False):nd (loadmql == True or loadmql == False):
            mv_tdata1loadrates = obj1_CDataProcess.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)1_CDataProcess.wrangle_time(mv_tdata1loadrates, mp_unit, mp_filesrc="rates2", filter_int=False, filter_flt=False, filter_obj=False, filter_dtmi=False, filter_dtmf=False, mp_dropna=False, mp_merge=True, mp_convert=True, mp_drop=True)

        # Create labels
        mv_tdata1apiticks = obj1_CDataProcess.create_label_wrapper(ess.create_label_wrapper(
            df=mv_tdata1apiticks,
            bid_column="T1_Bid_Price",
            ask_column="T1_Ask_Price",   ask_column="T1_Ask_Price",
            column_in="T1_Bid_Price",column_in="T1_Bid_Price",
            column_out1=list(mlenv.mp_ml_input_keyfeat)[0],
            column_out2=list(mlenv.mp_ml_output_label_scaled)[0],lenv.mp_ml_output_label_scaled)[0],
            open_column="R1_Open",
            high_column="R1_High",         high_column="R1_High",
            low_column="R1_Low",
            close_column="R1_Close",
            run_mode=1,
            lookahead_periods=tuneenv.mp_ml_cfg_period,riod,
            ma_window=tuneenv.mp_ml_tf_ma_windowin,
            hl_avg_col=tuneenv.mp_ml_hl_avg_col,
            ma_col=tuneenv.mp_ml_ma_col,
            returns_col=tuneenv.mp_ml_returns_col,
            shift_in=tuneenv.mp_ml_tf_shiftin,
            rownumber=dataenv.mp_data_rownumber,            rownumber=dataenv.mp_data_rownumber,
            create_label=Truel=True
        )

        mv_tdata1apirates = obj1_CDataProcess.create_label_wrapper(Process.create_label_wrapper(
            df=mv_tdata1apirates,
            bid_column="R1_Bid_Price",,
            ask_column="R1_Ask_Price",
            column_in="R1_Close",
            column_out1=list(mlenv.mp_ml_input_keyfeat)[0],.mp_ml_input_keyfeat)[0],
            column_out2=list(mlenv.mp_ml_output_label_scaled)[0],.mp_ml_output_label_scaled)[0],
            open_column="R1_Open",",
            high_column="R1_High",
            low_column="R1_Low","R1_Low",
            close_column="R1_Close",
            run_mode=2,
            lookahead_periods=tuneenv.mp_ml_cfg_period,period,
            ma_window=tuneenv.mp_ml_tf_ma_windowin,a_windowin,
            hl_avg_col=tuneenv.mp_ml_hl_avg_col,
            ma_col=tuneenv.mp_ml_ma_col,
            returns_col=tuneenv.mp_ml_returns_col,l,
            shift_in=tuneenv.mp_ml_tf_shiftin,mp_ml_tf_shiftin,
            rownumber=dataenv.mp_data_rownumber,   rownumber=dataenv.mp_data_rownumber,
            create_label=True            create_label=True
        )

        mv_tdata1loadticks = obj1_CDataProcess.create_label_wrapper(aProcess.create_label_wrapper(
            df=mv_tdata1loadticks,
            bid_column="T2_Bid_Price",ice",
            ask_column="T2_Ask_Price",
            column_in="T2_Bid_Price",
            column_out1=list(mlenv.mp_ml_input_keyfeat)[0],.mp_ml_input_keyfeat)[0],
            column_out2=list(mlenv.mp_ml_output_label_scaled)[0],.mp_ml_output_label_scaled)[0],
            open_column="R2_Open",",
            high_column="R2_High",
            low_column="R2_Low","R2_Low",
            close_column="R2_Close",
            run_mode=3,
            lookahead_periods=tuneenv.mp_ml_cfg_period,period,
            ma_window=tuneenv.mp_ml_tf_ma_windowin,a_windowin,
            hl_avg_col=tuneenv.mp_ml_hl_avg_col,
            ma_col=tuneenv.mp_ml_ma_col,
            returns_col=tuneenv.mp_ml_returns_col,l,
            shift_in=tuneenv.mp_ml_tf_shiftin,mp_ml_tf_shiftin,
            rownumber=dataenv.mp_data_rownumber,   rownumber=dataenv.mp_data_rownumber,
            create_label=True            create_label=True
        )

        mv_tdata1loadrates = obj1_CDataProcess.create_label_wrapper(aProcess.create_label_wrapper(
            df=mv_tdata1loadrates,
            bid_column="R2_Bid_Price",,
            ask_column="R2_Ask_Price",
            column_in="R2_Close",
            column_out1=list(mlenv.mp_ml_input_keyfeat)[0],.mp_ml_input_keyfeat)[0],
            column_out2=list(mlenv.mp_ml_output_label_scaled)[0],.mp_ml_output_label_scaled)[0],
            open_column="R2_Open",",
            high_column="R2_High",
            low_column="R2_Low","R2_Low",
            close_column="R2_Close",
            run_mode=4,
            lookahead_periods=tuneenv.mp_ml_cfg_period,period,
            ma_window=tuneenv.mp_ml_tf_ma_windowin,a_windowin,
            hl_avg_col=tuneenv.mp_ml_hl_avg_col,
            ma_col=tuneenv.mp_ml_ma_col,
            returns_col=tuneenv.mp_ml_returns_col,l,
            shift_in=tuneenv.mp_ml_tf_shiftin,mp_ml_tf_shiftin,
            rownumber=dataenv.mp_data_rownumber,   rownumber=dataenv.mp_data_rownumber,
            create_label=True            create_label=True
        )

        scolumn_out1 = mlenv.mp_ml_input_keyfeatut_keyfeat
        scolumn_out2 = mlenv.mp_ml_output_label_scaledput_label_scaled

        # Extract single elements from sets
        column_out1 = list(scolumn_out1)[0] if isinstance(scolumn_out1, set) else scolumn_out1_out1, set) else scolumn_out1
        column_out2 = list(scolumn_out2)[0] if isinstance(scolumn_out2, set) else scolumn_out2_out2)[0] if isinstance(scolumn_out2, set) else scolumn_out2

        column_out1 = str(column_out1)_out1)
        column_out2 = str(column_out2)2)

        logger.info("column_out1: %s", column_out1)
        logger.info("column_out2: %s", column_out2)

        # Create labels
        if loadmql:
            mv_tdata1apiticks = obj1_CDataProcess.create_label_wrapper(ess.create_label_wrapper(
                df=mv_tdata1apiticks,
                bid_column="T1_Bid_Price",1_Bid_Price",
                ask_column="T1_Ask_Price",       ask_column="T1_Ask_Price",
                column_in="T1_Bid_Price",                column_in="T1_Bid_Price",
                column_out1='Close',
                column_out2='Close_Scaled',
                open_column="R1_Open",                open_column="R1_Open",
                high_column="R1_High",
                low_column="R1_Low",
                close_column="R1_Close",
                run_mode=1,                run_mode=1,
                lookahead_periods=tuneenv.mp_ml_cfg_period,env.mp_ml_cfg_period,
                ma_window=tuneenv.mp_ml_tf_ma_windowin,l_tf_ma_windowin,
                hl_avg_col=tuneenv.mp_ml_hl_avg_col,                hl_avg_col=tuneenv.mp_ml_hl_avg_col,
                ma_col=tuneenv.mp_ml_ma_col,
                returns_col=tuneenv.mp_ml_returns_col,ol,
                shift_in=tuneenv.mp_ml_tf_shiftin,                shift_in=tuneenv.mp_ml_tf_shiftin,
                rownumber=dataenv.mp_data_rownumber,er=dataenv.mp_data_rownumber,
                create_label=Trueate_label=True
            )

        if loadmql:
            mv_tdata1apirates = obj1_CDataProcess.create_label_wrapper(Process.create_label_wrapper(
                df=mv_tdata1apirates,
                bid_column="R1_Bid_Price",rice",
                ask_column="R1_Ask_Price",
                column_in="R1_Close",
                column_out1='Close',
                column_out2='Close_Scaled',caled',
                open_column="R1_Open",
                high_column="R1_High",="R1_High",
                low_column="R1_Low",
                close_column="R1_Close",
                run_mode=2,
                lookahead_periods=tuneenv.mp_ml_cfg_period,_ml_cfg_period,
                ma_window=tuneenv.mp_ml_tf_ma_windowin,,
                hl_avg_col=tuneenv.mp_ml_hl_avg_col,l,
                ma_col=tuneenv.mp_ml_ma_col,
                returns_col=tuneenv.mp_ml_returns_col,nv.mp_ml_returns_col,
                shift_in=tuneenv.mp_ml_tf_shiftin,   shift_in=tuneenv.mp_ml_tf_shiftin,
                rownumber=dataenv.mp_data_rownumber,                rownumber=dataenv.mp_data_rownumber,
                create_label=Trueate_label=True
            )

        if loadmql == True or loadmql == False:alse:
            mv_tdata1loadticks = obj1_CDataProcess.create_label_wrapper(aProcess.create_label_wrapper(
                df=mv_tdata1loadticks,,
                bid_column="T2_Bid_Price",rice",
                ask_column="T2_Ask_Price",
                column_in="T2_Bid_Price",e",
                column_out1='Close',
                column_out2='Close_Scaled',caled',
                open_column="R2_Open",
                high_column="R2_High",="R2_High",
                low_column="R2_Low",
                close_column="R2_Close",
                run_mode=3,
                lookahead_periods=tuneenv.mp_ml_cfg_period,_ml_cfg_period,
                ma_window=tuneenv.mp_ml_tf_ma_windowin,,
                hl_avg_col=tuneenv.mp_ml_hl_avg_col,l,
                ma_col=tuneenv.mp_ml_ma_col,
                returns_col=tuneenv.mp_ml_returns_col,nv.mp_ml_returns_col,
                shift_in=tuneenv.mp_ml_tf_shiftin,   shift_in=tuneenv.mp_ml_tf_shiftin,
                rownumber=dataenv.mp_data_rownumber,                rownumber=dataenv.mp_data_rownumber,
                create_label=True
            )

        if loadmql == True or loadmql == False:alse:
            mv_tdata1loadrates = obj1_CDataProcess.create_label_wrapper(aProcess.create_label_wrapper(
                df=mv_tdata1loadrates,
                bid_column="R2_Bid_Price",rice",
                ask_column="R2_Ask_Price",
                column_in="R2_Close",
                column_out1='Close',
                column_out2='Close_Scaled',caled',
                open_column="R2_Open",
                high_column="R2_High",="R2_High",
                low_column="R2_Low",
                close_column="R2_Close",
                run_mode=4,
                lookahead_periods=tuneenv.mp_ml_cfg_period,_ml_cfg_period,
                ma_window=tuneenv.mp_ml_tf_ma_windowin,,
                hl_avg_col=tuneenv.mp_ml_hl_avg_col,l,
                ma_col=tuneenv.mp_ml_ma_col,
                returns_col=tuneenv.mp_ml_returns_col,nv.mp_ml_returns_col,
                shift_in=tuneenv.mp_ml_tf_shiftin,   shift_in=tuneenv.mp_ml_tf_shiftin,
                rownumber=dataenv.mp_data_rownumber,                rownumber=dataenv.mp_data_rownumber,
                create_label=True
            )

        # Display the data
        if loadmql:
            obj1_CDataProcess.run_mql_print(mv_tdata1apiticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")_print(mv_tdata1apiticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        if loadmql:
            obj1_CDataProcess.run_mql_print(mv_tdata1apirates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")(mv_tdata1apirates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        if loadmql == True or loadmql == False:== False:
            obj1_CDataProcess.run_mql_print(mv_tdata1loadticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")print(mv_tdata1loadticks, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        if loadmql == True or loadmql == False:l == False:
            obj1_CDataProcess.run_mql_print(mv_tdata1loadrates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")int(mv_tdata1loadrates, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")

        # copy the data for config selection
        if loadmql == True:
            data_sources = [mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates]ata1apirates, mv_tdata1loadticks, mv_tdata1loadrates]
            data_copies = [data.copy() for data in data_sources]ata in data_sources]
            mv_tdata2a, mv_tdata2b, mv_tdata2c, mv_tdata2d = data_copiesta2d = data_copies
            data_mapping = {
                'loadapiticks': mv_tdata2a,
                'loadapirates': mv_tdata2b,v_tdata2b,
                'loadfileticks': mv_tdata2c,   'loadfileticks': mv_tdata2c,
                'loadfilerates': mv_tdata2d                'loadfilerates': mv_tdata2d
            }
        else:
            data_sources = [mv_tdata1loadticks, mv_tdata1loadrates]
            data_copies = [data.copy() for data in data_sources]pies = [data.copy() for data in data sources]
            mv_tdata1c, mv_tdata1d = data_copies
            data_mapping = {
                'loadfileticks': mv_tdata2c,
                'loadfilerates': mv_tdata2d
            }

        # Check the switch of which file to useuse
        if mp_data_cfg_usedata in data_mapping:ata in data_mapping:
            print(f"Using {mp_data_cfg_usedata.replace('load', '').replace('api', 'API ').replace('file', 'File ').replace('ticks', 'Tick data').replace('rates', 'Rates data')}") 'File ').replace('ticks', 'Tick data').replace('rates', 'Rates data')}")
            mv_tdata2 = data_mapping[mp_data_cfg_usedata]
        else:
            print("Invalid data configuration")ata configuration")
            mv_tdata2 = None

        # print shapes of the data
        print("SHAPE: mv_tdata2 shape:", mv_tdata2.shape)_tdata2.shape)

        # +------------------------------------------------------------------------------------------------------------------------------------
        # STEP: Normalize the X data
        # +---------------------------------------------------------------------------------
        # Normalize the 'Close' column
        scaler = MinMaxScaler()r()

        mp_data_custom_input_keyfeat = list(dataenv.mp_data_custom_input_keyfeat)  # Convert to list(dataenv.mp_data_custom_input_keyfeat)  # Convert to list
        mp_data_custom_input_keyfeat_scaled = list(dataenv.mp_data_custom_input_keyfeat_scaled)  # Convert to listta_custom_input_keyfeat_scaled = list(dataenv.mp_data_custom_input_keyfeat_scaled)  # Convert to list

        print("mp_data_custom_input_keyfeat:", mp_data_custom_input_keyfeat)mp_data_custom_input_keyfeat)
        print("mp_data_custom_input_keyfeat_scaled:", mp_data_custom_input_keyfeat_scaled)led:", mp_data_custom_input_keyfeat_scaled)

        # Ensure these remain as lists when used as column namesn names
        print("mv_tdata2 columns:", mv_tdata2.columns)("mv_tdata2 columns:", mv_tdata2.columns)
        print("mv_tdata2 head:", mv_tdata2.head(3))(3))

        mv_tdata2[mp_data_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_data_custom_input_keyfeat])        mv_tdata2[mp_data_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_data_custom_input_keyfeat])

        print("print Normalise")
        obj1_CDataProcess = CDataProcess(dataenv, mlenv, globalenv)        obj1_CDataProcess = CDataProcess(dataenv, mlenv, globalenv)
        print("Type after wrangle_time:", type(mv_tdata2))
        obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")int(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "plain", floatfmt=".5f", numalign="left", stralign="left")
        print("End of Normalise print")
        print("mv_tdata2.shape", mv_tdata2.shape)ata2.shape)

        # +-------------------------------------------------------------------        # +-------------------------------------------------------------------
        # STEP: add The time index to the data
        # +-------------------------------------------------------------------
        # Check the first column        # Check the first column
        first_column = mv_tdata2.columns[0]
        print("PRE INDEX: Count: ", len(mv_tdata2))

        # Ensure no missing values, duplicates, or type issues
        mv_tdata2[first_column] = mv_tdata2[first_column].fillna('Unknown')  # Handle NaNsmn].fillna('Unknown')  # Handle NaNs
        mv_tdata2[first_column] = mv_tdata2[first_column].astype(str)  # Uniform typeolumn].astype(str)  # Uniform type
        mv_tdata2[first_column] = mv_tdata2[first_column].str.strip()  # Remove whitespaces        mv_tdata2[first_column] = mv_tdata2[first_column].str.strip()  # Remove whitespaces

        # Set the first column as index which is the datetime column        # Set the first column as index which is the datetime column
        mv_tdata2.set_index(first_column, inplace=True)t_column, inplace=True)
        mv_tdata2 = mv_tdata2.dropna()
        print("POST INDEX: Count: ", len(mv_tdata2))

        # +----------------------------------------------------------------------------------------------------------
        # STEP: set the dataset to just the features and the label and sort by timeres and the label and sort by time
        # +-------------------------------------------------------------------        # +-------------------------------------------------------------------
        print("Type before set default:", type(mv_tdata2))

        if dataenv.mp_data_data_label == 1:
            mv_tdata2 = mv_tdata2[[list(mp_data_custom_input_keyfeat_scaled)[0]]]2[[list(mp_data_custom_input_keyfeat_scaled)[0]]]
            obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

        elif dataenv.mp_data_data_label == 2:        elif dataenv.mp_data_data_label == 2:
            mv_tdata2 = mv_tdata2[[mv_tdata2.columns[0]] + [list(mp_data_custom_input_keyfeat_scaled)[0]]]st(mp_data_custom_input_keyfeat_scaled)[0]]]
            # Ensure the data is sorted by time
            mv_tdata2 = mv_tdata2.sort_index()
            obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left") "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

        elif dataenv.mp_data_data_label == 3:
            # Ensure the data is sorted by time use full datasetl dataset
            mv_tdata2 = mv_tdata2.sort_index()_index()
            print("Count of Tdata2:", len(mv_tdata2)))
            obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")            obj1_CDataProcess.run_mql_print(mv_tdata2, mp_data_tab_rows, mp_data_tab_width, "fancy_grid", floatfmt=".5f", numalign="left", stralign="left")

        # +-------------------------------------------------------------------
        # STEP: Generate X and y from the Time Series
        # +---------------------------------------------------------------------------------------
        # At thsi point the normalised data columns are split across the X and Y data        # At thsi point the normalised data columns are split across the X and Y data
        obj1_Mqlmlsetup = CMqlmlsetup()  # Create an instance of the classCreate an instance of the class
        # 1: 24 HOURS/24 HOURS prediction window
        print("1: MINUTES data entries per time frame: MINUTES:", MINUTES, "HOURS:", MINUTES * 60, "DAYS:", MINUTES * 60 * 24)
        timeval = MINUTES * 60  # hours        timeval = MINUTES * 60  # hours
        pasttimeperiods = 24
        futuretimeperiods = 24
        predtimeperiods = 1
        features_count = len(dataenv.mp_data_custom_input_keyfeat)  # Number of features in inputustom_input_keyfeat)  # Number of features in input
        labels_count = len(dataenv.mp_data_custom_output_label)  # Number of labels in output
        batch_size = tuneenv.batch_size        batch_size = tuneenv.batch_size

        print("timeval:", timeval, "pasttimeperiods:", pasttimeperiods, "futuretimeperiods:", futuretimeperiods, "predtimeperiods:", predtimeperiods)eriods, "futuretimeperiods:", futuretimeperiods, "predtimeperiods:", predtimeperiods)
        past_width = pasttimeperiods * timeval
        future_width = futuretimeperiods * timeval
        pred_width = predtimeperiods * timeval
        print("past_width:", past_width, "future_width:", future_width, "pred_width:", pred_width)        print("past_width:", past_width, "future_width:", future_width, "pred_width:", pred_width)

        # Create the input features (X) and label values (y)ues (y)
        print("list(mp_data_custom_input_keyfeat_scaled)", list(mp_data_custom_input_keyfeat_scaled))_input_keyfeat_scaled))

        # STEP: Create input (X) and label (Y) tensors Ensure consistent data shapeata shape
        # Create the input (X) and label (Y) tensors Close_scaled is the feature to predict and Close last entry in future the labelsors Close_scaled is the feature to predict and Close last entry in future the label
        mv_tdata2_X, mv_tdata2_y = obj1_Mqlmlsetup.create_Xy_time_windows3(mv_tdata2, past_width, future_width, target_column=list(mp_data_custom_input_keyfeat_scaled), feature_column=list(mp_data_custom_input_keyfeat))list(mp_data_custom_input_keyfeat_scaled), feature_column=list(mp_data_custom_input_keyfeat))
        print("mv_tdata2_X.shape", mv_tdata2_X.shape, "mv_tdata2_y.shape", mv_tdata2_y.shape)data2_X.shape, "mv_tdata2_y.shape", mv_tdata2_y.shape)

        # +-------------------------------------------------------------------------------------------------------------------
        # STEP: Normalize the Y datahe Y data
        # +-------------------------------------------------------------------
        # Scale the Y labels
        mv_tdata2_y = scaler.transform(mv_tdata2_y.reshape(-1, 1))  # Transform Y valuesmv_tdata2_y.reshape(-1, 1))  # Transform Y values

        # +-------------------------------------------------------------------
        # STEP: Split the data into training and test sets Fixed Partitioningnd test sets Fixed Partitioning
        # +-----------------------------------------------------------------------------------------------
        # Batch size alignment fit the number of rows as whole number divisible by the batch size to avoid float errorsof rows as whole number divisible by the batch size to avoid float errors
        batch_size = tuneenv.batch_size
        precountX = len(mv_tdata2_X)        precountX = len(mv_tdata2_X)
        precounty = len(mv_tdata2_y)
        mv_tdata2_X, mv_tdata2_y = obj1_Mqlmlsetup.align_to_batch_size(mv_tdata2_X, mv_tdata2_y, batch_size)h_size)
        print(f"Aligned data: X shape: {mv_tdata2_X.shape}, Y shape: {mv_tdata2_y.shape}")        print(f"Aligned data: X shape: {mv_tdata2_X.shape}, Y shape: {mv_tdata2_y.shape}")

        # Check the number of rows
        print("Batch size alignment: mv_tdata2_X shape:", mv_tdata2_X.shape, "Precount:", precountX, "Postcount:", len(mv_tdata2_X))
        print("Batch size alignment: mv_tdata2_y shape:", mv_tdata2_y.shape, "Precount:", precounty, "Postcount:", len(mv_tdata2_y))county, "Postcount:", len(mv_tdata2_y))

        # Split the data into training, validation, and test sets

        # STEP: Split data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(mv_tdata2_X, mv_tdata2_y, test_size=(tuneenv.validation_split + tuneenv.test_split), shuffle=False)rain, y_temp = train_test_split(mv_tdata2_X, mv_tdata2_y, test_size=(tuneenv.validation_split + tuneenv.test_split), shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(tuneenv.test_split / (tuneenv.validation_split + tuneenv.test_split)), shuffle=False)nv.test_split / (tuneenv.validation_split + tuneenv.test_split)), shuffle=False)

        print(f"Training set: X_train: {X_train.shape}, y_train: {y_train.shape}")e}")
        print(f"Validation set: X_val: {X_val.shape}, y_val: {y_val.shape}")
        print(f"Test set: X_test: {X_test.shape}, y_test: {y_test.shape}")
        # +-------------------------------------------------------------------
        # STEP: convert numpy arrays to TF datasets TF datasets
        # +-------------------------------------------------------------------------------------------------------------
        # initiate the object using a window generatorwindow is not  used in this model Parametersa window generatorwindow is not  used in this model Parameters
        tf_batch_size = tuneenv.batch_size

        train_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)        train_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_train, y_train, batch_size=tf_batch_size, shuffle=True)
        val_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)tup.create_tf_dataset(X_val, y_val, batch_size=tf_batch_size, shuffle=False)
        test_dataset = obj1_Mqlmlsetup.create_tf_dataset(X_test, y_test, batch_size=tf_batch_size, shuffle=False)
        print(f"TF Datasets created: Train: {tf.data.experimental.cardinality(train_dataset).numpy()}, Val: {tf.data.experimental.cardinality(val_dataset).numpy()}, Test: {tf.data.experimental.cardinality(test_dataset).numpy()}")rdinality(val_dataset).numpy()}, Test: {tf.data.experimental.cardinality(test_dataset).numpy()}")

        # +--------------------------------------------------------------------------------
        # STEP: add tensor values for model input        # STEP: add tensor values for model input
        # +------------------------------------------------------------------------------

        train_shape, val_shape, test_shape = None, None, None

        for dataset, name in zip([train_dataset, val_dataset, test_dataset], ['train', 'val', 'test']):in', 'val', 'test']):
            for spec in dataset.element_spec:
                if name == 'train':
                    train_shape = spec.shape
                elif name == 'val':
                    val_shape = spec.shape
                elif name == 'test':
                    test_shape = spec.shapee

        # +-------------------------------------------------------------------
        # STEP: Final shape summaries
        # +-------------------------------------------------------------------
        # Final summary of shapes
        # STEP: Confirm tensor shapes for the tuner        # STEP: Confirm tensor shapes for the tuner
        input_shape = X_train.shape[1:]  # Shape of a single sample (time_steps, features)s, features)
        label_shape = y_train.shape[1:]  # Shape of a single label (future_steps)of a single label (future_steps)
        # Example data: shape = (num_samples, time_steps, features) cnn_data = np.random.random((1000, 1440, 1))  # 1000 samples, 1440 timesteps, 1 feature np.random.random((1000, 1440, 1))  # 1000 samples, 1440 timesteps, 1 feature
        # Example data: labels = np.random.random((1000, 1))        # Example data: labels = np.random.random((1000, 1))
        print(f"Full Input shape for model: {X_train.shape}, Label shape for model: {y_train.shape}")Label shape for model: {y_train.shape}")
        print(f"No Batch Input shape for model: {input_shape}, Label shape for model: {label_shape}")        print(f"No Batch Input shape for model: {input_shape}, Label shape for model: {label_shape}")
        # batch components
        input_keras_batch = tuneenv.batch_sizee
        input_def_keras_batch = Nonee
        # Get the input shape for the model
        input_rows_X = len(X_train)
        input_rows_y = len(y_train)
        input_batch_size = tuneenv.batch_sizeatch_size
        input_batches = X_train.shape[0]
        input_timesteps = X_train.shape[1]        input_timesteps = X_train.shape[1]
        input_features = X_train.shape[2]
        # Get the output shape for the modele model
        output_label = y_train.shape[1]
        output_shape = y_train.shapeape
        output_features = y_train.shape[1]
        print(f"input_def_keras_batch  {input_def_keras_batch}, input_keras_batch: {input_keras_batch}")keras_batch}")
        print(f"Input rows X: {input_rows_X},Input rows y: {input_rows_y} , Input batch_size {input_batch_size}, Input batches: {input_batches}, Input timesteps: {input_timesteps}, Input steps or features: {input_features}") batch_size {input_batch_size}, Input batches: {input_batches}, Input timesteps: {input_timesteps}, Input steps or features: {input_features}")
        print(f"Output label: {output_label}, Output shape: {output_shape}, Output features: {output_features}")
        # pass in the data shape for the model

        input_shape = (input_timesteps, input_features)
        output_label_shape = (output_label, dataenv.mp_data_custom_output_label_count) = (output_label, dataenv.mp_data_custom_output_label_count)
        print(f"Input shape for model: {input_shape}, Output shape for model: {output_label_shape}")shape}, Output shape for model: {output_label_shape}")
        # +-------------------------------------------------------------------------------------------------------------
        # STEP: Tune best model Hyperparameter tuning and model setupter tuning and model setup
        # +--------------------------------------------------------------------------------------------------------------
        # Hyperparameter configurationion
        obj1_TunerParams = CMdtunerHyperModel((
            tuner_params=tuneenv,
            input_shape=input_shape,
            data_input_shape=input_shape,
            output_shape=output_label_shape,
            input_batch_size=batch_size,,
            tunemode=True,
            tunemodeepochs=True,
            model_summary=True,
            batch_size=batch_size,
            epochs=15
        )
        hypermodel_params = obj1_TunerParams.get_hypermodel_params()        hypermodel_params = obj1_TunerParams.get_hypermodel_params()
        print("Tuner Parameters:", hypermodel_params)  # Print the tuner parameters# Print the tuner parameters

        # instansiate tuner claa
        obj1_CMdtuner = CMdtuner(
            hypermodel_params=hypermodel_params,
            traindataset=train_dataset,
            valdataset=val_dataset,
            testdataset=test_dataset,
            castmode='float32',

        )
        # initialize tuner
        obj1_CMdtuner.initialize_tuner()

        # Check and load the modelel
        best_model = obj1_CMdtuner.check_and_load_model(mp_ml_mbase_path, ftype='tf')ner.check_and_load_model(mp_ml_mbase_path, ftype='tf')

        # If no model or a hard run is required, run the search or a hard run is required, run the search
        runtuner = Falseuntuner = False

        if best_model is None:
            print("Running the tuner search bo model")            print("Running the tuner search bo model")
            runtuner = obj1_CMdtuner.run_search()uner.run_search()
            obj1_CMdtuner.export_best_model(ftype='tf')best_model(ftype='tf')
        elif mp_ml_hard_run:
            print("Running the tuner search hard run")arch hard run")
            runtuner = obj1_CMdtuner.run_search()r.run_search()
            obj1_CMdtuner.export_best_model(ftype='tf')_model(ftype='tf')
        else:
            print("Best model loaded successfully")            print("Best model loaded successfully")
            runtuner = True   runtuner = True

        # +---------------------------------------------------------------------------------------------------------
        # STEP: Train and evaluate the best model        # STEP: Train and evaluate the best model
        # +---------------------------------------------------------------------------------------------------------------
        print("Model: Loading file from directory", mp_ml_mbase_path, "Model: filename", mp_ml_project_name)e", mp_ml_project_name)
        if (load_model := obj1_CMdtuner.check_and_load_model(mp_ml_mbase_path, ftype='tf')) is not None:        if (load_model := obj1_CMdtuner.check_and_load_model(mp_ml_mbase_path, ftype='tf')) is not None:
            best_model = load_model
            print("Model: Best model: ", best_model.name): Best model: ", best_model.name)
            print("best_model.summary()", best_model.summary())            print("best_model.summary()", best_model.summary())

            # Fit the label scaler on the training labelsels

            # Model Training
            tf.keras.backend.clear_session(free_memory=True).clear_session(free_memory=True)

            print("Training the best model...")
            best_model.fit(
                train_dataset,   train_dataset,
                validation_data=val_dataset,
                batch_size=batch_size,batch_size,
                epochs=mp_ml_tf_param_epochs                epochs=mp_ml_tf_param_epochs
            )
            print("Training completed.")

            # Model Evaluation
            print("Evaluating the model...")
            val_metrics = best_model.evaluate(val_dataset, verbose=1)l.evaluate(val_dataset, verbose=1)
            test_metrics = best_model.evaluate(test_dataset, verbose=1)et, verbose=1)

            print(f"Validation Metrics - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")            print(f"Validation Metrics - Loss: {val_metrics[0]:.4f}, Accuracy: {val_metrics[1]:.4f}")
            print(f"Test Metrics - Loss: {test_metrics[0]:.4f}, Accuracy: {test_metrics[1]:.4f}"):.4f}, Accuracy: {test_metrics[1]:.4f}")

            # Fit the label scaler on the training labelsscaler on the training labels
            label_scaler.fit(y_train.reshape(-1, 1))

            # Predictions and Scaling
            print("Running predictions and scaling...")predictions and scaling...")
            predicted_fx_price = best_model.predict(test_dataset) = best_model.predict(test_dataset)
            predicted_fx_price = label_scaler.inverse_transform(predicted_fx_price)r.inverse_transform(predicted_fx_price)

            real_fx_price = label_scaler.inverse_transform(y_test.reshape(-1, 1))erse_transform(y_test.reshape(-1, 1))
            print("Predictions and scaling completed.")rint("Predictions and scaling completed.")
            # +-------------------------------------------------------------------------------------------------------------
            # STEP: Performance Check            # STEP: Performance Check
            # +-----------------------------------------------------------------------------------------------------------------------
            # Evaluation and visualization
            # Mean Squared Error (MSE): It measures the average squared difference between the predicted and actual values.ed difference between the predicted and actual values.
            # The lower the MSE, the better the model.

            # Mean Absolute Error (MAE): It measures the average absolute difference between the predicted and actual values.icted and actual values.
            # Like MSE, lower values indicate better model performance.

            # R2 Score: Also known as the coefficient of determination, it measures the proportion of the variance in thedetermination, it measures the proportion of the variance in the
            # dependent variable that is predictable from the independent variable(s). An R2 score of 1 indicates a from the independent variable(s). An R2 score of 1 indicates a
            # perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the label            # perfect fit, while a score of 0 suggests that the model is no better than predicting the mean of the label
            # variable. Negative values indicate poor model performance.es indicate poor model performance.
            # Check for NaN values and handle them
            if np.isnan(real_fx_price).any() or np.isnan(predicted_fx_price).any():d_fx_price).any():
                print("Warning: NaN values found in input data. Handling NaNs by removing corresponding entries.")moving corresponding entries.")
                mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)                mask = ~np.isnan(real_fx_price) & ~np.isnan(predicted_fx_price)
                real_fx_price = real_fx_price[mask]
                predicted_fx_price = predicted_fx_price[mask][mask]

            mse = mean_squared_error(real_fx_price, predicted_fx_price)real_fx_price, predicted_fx_price)
            mae = mean_absolute_error(real_fx_price, predicted_fx_price)
            r2 = r2_score(real_fx_price, predicted_fx_price)redicted_fx_price)
            print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
            print(f"Mean Squared Error: The lower the MSE, the better the model: {mse}")MSE, the better the model: {mse}")
            print(f"Mean Absolute Error: The lower the MAE, the better the model: {mae}")            print(f"Mean Absolute Error: The lower the MAE, the better the model: {mae}")
            print(f"R2 Score: The closer to 1, the better the model: {r2}")

            plt.plot(real_fx_price, color='red', label='Real FX Price')            plt.plot(real_fx_price, color='red', label='Real FX Price')
            plt.plot(predicted_fx_price, color='blue', label='Predicted FX Price')
            plt.title('FX Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('FX Price')
            plt.legend()
            plt.savefig(mp_ml_base_path + '/' + 'plot.png')
            if mp_ml_show_plot:
                plt.show()
            print("Plot Model saved to ", mp_ml_base_path + '/' + 'plot.png')e_path + '/' + 'plot.png')

            if ONNX_save:            if ONNX_save:
                # Save the model to ONNX

                # Define the output path
                mp_output_path = mp_ml_data_path + f"model_{mp_symbol_primary}_{mp_ml_data_type}.onnx"odel_{mp_symbol_primary}_{mp_ml_data_type}.onnx"
                print(f"Output Path: {mp_output_path}")

                # Convert Keras model to ONNX
                opset_version = 17  # Choose an appropriate ONNX opset version                opset_version = 17  # Choose an appropriate ONNX opset version

                # Assuming your model has a single input
                spec = [tf.TensorSpec(input_shape, tf.float32, name="input")]shape, tf.float32, name="input")]
                onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)= tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=opset_version)

                # Save the ONNX modelhe ONNX model
                onnx.save_model(onnx_model, mp_output_path)
                print(f"Model saved to {mp_output_path}")aved to {mp_output_path}")

                # Verify the ONNX model
                checker.check_model(onnx_model)                checker.check_model(onnx_model)
                print("ONNX model is valid.")NX model is valid.")

                # Check ONNX Runtime version                # Check ONNX Runtime version
                print("ONNX Runtime version:", ort.__version__)ion:", ort.__version__)

                # finish
                mt5.shutdown()                mt5.shutdown()
                print("Finished")
        else:
            print("No data loaded")            print("No data loaded")
        mt5.shutdown()
        print("Finished")

"""


if __name__ == "__main__":
    main(logger)


