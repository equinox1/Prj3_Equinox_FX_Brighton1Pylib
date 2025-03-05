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
from tsMqlMLParams import CMqlEnvMLParams
from tsMqlMLProcess import CDMLProcess
from tsMqlOverrides import CMqlOverrides
from tsMqlUtilities import CUtilities


# Setup the logging and tensor platform dependencies
obj1_CMqlSetup = CMqlSetup(loglevel='INFO', warn='ignore', tfdebug=False)
strategy = obj1_CMqlSetup.get_computation_strategy()
# format values
mp_data_tab_rows = 5
mp_data_tab_width = 30
hrows=mp_data_tab_rows
hwidth=mp_data_tab_width
# +-------------------------------------------------------------------
# STEP: End of driving parameters
# +-------------------------------------------------------------------

def main(logger):
    with strategy.scope():

        # +----------------------------------------------------------
        # STEP: setup environment
        # +----------------------------------------------------------
        # Usage:env = EnvManager(custom_params={"ml": {"epochs": 20}}) ,logger.info("ML Epochs:", env.get_param("ml", "epochs"))
        # env.override_params({"base": {"learning_rate": 0.005}}) ,logger.info("Updated Learning Rate:", env.get_param("base", "learning_rate"))
        utils_config = CUtilities()
        env = CMqlEnvMgr()
        override_config = CMqlOverrides()
        # Fetch all parameters
        params = env.all_params()
        params.update(override_config.all_params())
        logger.info(f"All Parameters: {params}")

        # Ensure sections exist
        params_sections = params.keys()
        logger.info(f"PARAMS SECTIONS: {params_sections}")

        base_params = params.get("base", {})
        data_params = params.get("data", {})
        ml_params = params.get("ml", {})
        mltune_params = params.get("mltune", {})
        app_params = params.get("app", {})

        # Check if data_params exists
        if not data_params:
            logger.error("Missing 'data' section in params!")
            raise ValueError("Missing 'data' section in params.")

        # +-------------------------------------------------------------------
        # STEP: Load Reference class and time variables
        # +-------------------------------------------------------------------

        # File wrangling overrides
        mp_data_dropna = data_params.get('df4_mp_data_dropna', False)
        mp_data_merge = data_params.get('df4_mp_data_merge', False)
        mp_data_convert = data_params.get('df4_mp_data_convert', False)
        mp_data_drop = data_params.get('df4_mp_data_drop', False)
        logger.info(f"Dropna: {mp_data_dropna}, Merge: {mp_data_merge}, Convert: {mp_data_convert}, Drop: {mp_data_drop}")

        # Reference class
        lp_timeframe_name = data_params.get('mp_data_timeframe', 'H4')
        reference_config = CMqlRefConfig(loaded_data_type='MINUTE', required_data_type=lp_timeframe_name)
        #Symbol Constants
        PRIMARY_SYMBOL = app_params.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD'))
        SECONDARY_SYMBOL = app_params.get('lp_app_secondary_symbol', app_params.get('mp_app_primary_symbol','EURCHF'))
        # Time Constants
        UNIT=reference_config.TIME_CONSTANTS["UNIT"]["SECOND"]
        MINUTE = reference_config.get_timevalue('MINUTE')
        HOUR = reference_config.get_timevalue('HOUR')
        DAY = reference_config.get_timevalue('DAY')
        # Current Time Constants
        CURRENTDAY = reference_config.get_current_time()["CURRENTDAY"]
        CURRENTMONTH = reference_config.get_current_time()["CURRENTMONTH"]
        CURRENTYEAR = reference_config.get_current_time()["CURRENTYEAR"]
   
        # Mql Time Constants
        TIMEZONE = reference_config.get_current_time()["TIMEZONE"]
        TIMEFRAME = reference_config.get_current_time()["TIMEFRAME"]
        timeval = HOUR # hours # used in the window creation
        logger.info(f"Timezone: {TIMEZONE}")
        logger.info(f"Timeframe: {TIMEFRAME}") 

        # Fetch individual parameters safely
        lp_timeframe_name = TIMEFRAME
        rows =  data_params.get('mp_data_rows', 1000)
        rowcount = data_params.get('mp_data_rowcount', 10000)
        logger.info(f"Timeframe Name: {lp_timeframe_name}, Rows: {rows}, Rowcount: {rowcount}")
       
        # +-------------------------------------------------------------------
        # STEP: CBroker Login
        # +-------------------------------------------------------------------
        logger.info("PARAM HEADER: MP_APP_BROKER:", app_params.get('mp_app_broker'))
        broker_config = CMqlBrokerConfig(app_params.get('mp_app_broker'))
        mqqlobj = broker_config.run_mql_login()
        if mqqlobj is True:
            logger.info("Successfully logged in to MetaTrader 5.")
        else:
            logger.info(f"Failed to login. Error code: {mqqlobj}")
        

        import MetaTrader5 as mt5
        mt5.initialize()
        # set time zone to UTC
        timezone = pytz.timezone("Etc/UTC")
        # create 'datetime' object in UTC time zone to avoid the implementation of a local time zone offset
        utc_from = datetime(2020, 1, 10, tzinfo=timezone)
         
        lp_app_primary_symbol = "EURUSD"
        lp_utc_from = datetime(2020, 1, 10, tzinfo=timezone)
        lp_data_rows = 10
        lp_data_rowcount = 100
        lp_timeframe = "H4"

        #Ticks
        # get 10 EURUSD ticks starting from 01.10.2020 in UTC time zone
        ticks = mt5.copy_ticks_from("EURUSD", utc_from, 10, mt5.COPY_TICKS_ALL)
        print("Ticks received:",len(ticks))
        ticks1 = pd.DataFrame(ticks)
        ticks1.head(5)

        # Rates
        ##############################################################
        # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
        rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_H4, utc_from, 10)
        print("Rates received:",len(rates))
        rates1 =pd.DataFrame(rates)
        rates1.head(5)

   

        ## main

        #Ticks
        # get 10 EURUSD ticks starting from 01.10.2020 in UTC time zone
       
        ticks2=mt5.copy_ticks_from(lp_app_primary_symbol, lp_utc_from, lp_data_rows, mt5.COPY_TICKS_ALL)
        print("Ticks received:",len(ticks2))
        ticks2 = pd.DataFrame(ticks2)
        ticks2.head(5)

        # Rates
        ##############################################################
        # get 10 EURUSD H4 bars starting from 01.10.2020 in UTC time zone
        rates2 = mt5.copy_rates_from(lp_app_primary_symbol, mt5.TIMEFRAME_H4, lp_utc_from, lp_data_rows)
        print("Rates received:",len(rates2))
        rates2 =pd.DataFrame(rates2)
        rates2.head(5)

        mt5.shutdown()
      
        
if __name__ == "__main__":
    main(logger)