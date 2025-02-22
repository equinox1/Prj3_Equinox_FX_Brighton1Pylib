"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlDataLoader.py
File: tsPyPackages/tsMqlDataLoader/tsMqlDataLoader.py
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

import logging
import pandas as pd
import numpy as np
import pytz
from datetime import datetime

from tsMqlPlatform import (
    run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
)

# initialise platform
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")
if loadmql:
      import MetaTrader5 as mt5

# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr

class CDataLoader:
    """Class to manage and load market data parameters with override capability."""

    def __init__(self, **kwargs):
        if loadmql:
               import MetaTrader5 as mt5
               self.mt5 = mt5
        else:
            self.mt5 = None
            
         # global parameters
        self.env = CMqlEnvMgr()
        self.params= self.env.all_params()

        self.lp_utc_from=lp_utc_from
        self.lp_utc_to=lp_utc_to

        self.base_params = self.env.all_params()["base"]
        self.data_params = self.env.all_params()["data"]
        self.ml_params = self.env.all_params()["ml"]
        self.mltune_params = self.env.all_params()["mltune"]   
        self.app_params = self.env.all_params()["app"]
        

        # data parameters
        self.mp_app_primary_symbol = kwargs.get(' mp_app_primary_symbol', self.app_params.get(' mp_app_primary_symbol'))
        self.mp_app_rows = kwargs.get('mp_app_rows', self.app_params.get('mp_app_rows'))
        self.mp_app_rowcount = kwargs.get('mp_app_rowcount', self.app_params.get('mp_app_rowcount'))

        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.base_params.get('mp_glob_data_path'))
        self.mp_data_data_label = kwargs.get('mp_data_data_label', self.data_params.get('mp_data_data_label'))
        self.mp_data_history_size = kwargs.get('mp_data_history_size', self.data_params.get('mp_data_history_size'))
        self.mp_data_timeframe = kwargs.get('mp_data_timeframe', self.data_params.get('mp_data_timeframe'))
        self.mp_data_tab_rows = kwargs.get('mp_data_tab_rows', self.data_params.get('mp_data_tab_rows'))
        self.mp_data_tab_width = kwargs.get('mp_data_tab_width', self.data_params.get('mp_data_tab_width'))
        self.mp_data_rownumber = kwargs.get('mp_data_rownumber', self.data_params.get('mp_data_rownumber'))
        self.mp_data_show_dtype = kwargs.get('mp_data_show_dtype', self.data_params.get('mp_data_show_dtype'))
        self.mp_data_show_head = kwargs.get('mp_data_show_head', self.data_params.get('mp_data_show_head'))
        self.mp_data_command_ticks = kwargs.get('mp_data_command_ticks', self.data_params.get('mp_data_command_ticks'))
        self.mp_data_command_rates = kwargs.get('mp_data_command_rates', self.data_params.get('mp_data_command_rates'))
        self.mp_data_cfg_usedata = kwargs.get('mp_data_cfg_usedata', self.data_params.get('mp_data_cfg_usedata'))
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', self.data_params.get('mp_data_loadapiticks'))
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', self.data_params.get('mp_data_loadapirates'))
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', self.data_params.get('mp_data_loadfileticks'))
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', self.data_params.get('mp_data_loadfilerates'))
        self.mp_data_dfname1 = kwargs.get('mp_data_dfname1', self.data_params.get('mp_data_dfname1'))
        self.mp_data_dfname2 = kwargs.get('mp_data_dfname2', self.data_params.get('mp_data_dfname2'))
        self.mp_app_rows = kwargs.get('mp_app_rows', self.data_params.get('mp_app_rows'))
        self.mp_data_rowcount = kwargs.get('mp_data_rowcount', self.data_params.get('mp_data_rowcount'))
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', self.data_params.get('mp_data_filename1'))
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', self.data_params.get('mp_data_filename2'))
        self.mp_app_primary_symbol = kwargs.get(' mp_app_primary_symbol', self.data_params.get(' mp_app_primary_symbol'))
        self.mp_data_utc_from = kwargs.get('mp_data_utc_from', self.data_params.get('mp_data_utc_from'))
        self.mp_data_utc_to = kwargs.get('mp_data_utc_to', self.data_params.get('mp_data_utc_to'))
        self.mp_data_custom_input_keyfeat = kwargs.get('mp_data_custom_input_keyfeat', self.data_params.get('mp_data_custom_input_keyfeat'))
        self.mp_data_custom_output_label = kwargs.get('mp_data_custom_output_label', self.data_params.get('mp_data_custom_output_label'))

    def load_data(self, **kwargs):
        """Loads market data from API or files."""
        rates = {'api_ticks': pd.DataFrame(), 'api_rates': pd.DataFrame(), 'file_ticks': pd.DataFrame(), 'file_rates': pd.DataFrame()}
        self.lp_utc_from=lp_utc_from
        self.lp_utc_to=lp_utc_to

        if loadmql:
            if self.mp_data_loadapiticks:
                rates['api_ticks'] = self._fetch_api_ticks()
            if self.mp_data_loadapirates:
                rates['api_rates'] = self._fetch_api_rates()
        
        if self.mp_data_loadfileticks:
            rates['file_ticks'] = self._load_from_file(self.mp_data_filename1)
        if self.mp_data_loadfilerates:
            rates['file_rates'] = self._load_from_file(self.mp_data_filename2)
        
        return rates
    
    def _fetch_api_ticks(self):
        """Fetches tick data from MetaTrader5 API."""
        try:
            
            logger.info(f"Fetching rate data from MetaTrader5 API")
            logger.info(f"Symbol: {self.mp_app_primary_symbol}")
            logger.info(f"Timeframe: {valid_timeframe}")
            logger.info(f"UTC to: {self.mp_data_utc_to}")   
            logger.info(f"Rows: {self.mp_app_rows}")
            logger.info(f"Command: {self.mp_data_command_ticks}")


            data = mt5.copy_ticks_from(self. mp_app_primary_symbol, self.mp_data_utc_to, self.mp_app_rows, mt5.COPY_TICKS_ALL)
            df = pd.DataFrame(data)
            logger.info(f"API tick data received: {len(df)} rows" if not df.empty else "No tick data found")
            return df
        except Exception as e:
            logger.error(f"MT5 API tick data exception: {e}")
            return pd.DataFrame()
    
    def _fetch_api_rates(self):
        """Fetches rate data from MetaTrader5 API."""
        try:
            valid_timeframe = self._validate_timeframe(self.mp_data_timeframe)
            if valid_timeframe is None:
                raise ValueError(f"Invalid timeframe: {self.mp_data_timeframe}")
         
            logger.info(f"Fetching rate data from MetaTrader5 API")
            logger.info(f"Symbol: {self.mp_app_primary_symbol}")
            logger.info(f"Timeframe: {valid_timeframe}")
            logger.info(f"UTC to: {self.mp_data_utc_to}")   
            logger.info(f"Rows: {self.mp_app_rows}")

            data = mt5.copy_rates_from(self.mp_app_primary_symbol, valid_timeframe, self.mp_data_utc_to, self.mp_app_rows)
            df = pd.DataFrame(data)
            logger.info(f"API rate data received: {len(df)} rows" if not df.empty else "No rate data found")
            return df
        except Exception as e:
            logger.error(f"MT5 API rates exception: {e}")
            return pd.DataFrame()

    def _load_from_file(self, filename):
        """Loads data from a CSV file."""
        try:
            if not filename:
                return pd.DataFrame()
            filepath = f"{self.mp_data_path}/{filename}"
            df = pd.read_csv(filepath, sep=",", nrows=self.mp_data_rowcount, low_memory=False)
            if "vol3" in df.columns:
                df.drop("vol3", axis=1, inplace=True)
            logger.info(f"File data received: {len(df)} rows" if not df.empty else "No data found")
            return df
        except Exception as e:
            logger.error(f"File load exception: {e}")
            return pd.DataFrame()
    
    def _validate_timeframe(self, timeframe):
        """Validates the timeframe string."""
        logger.info(f"Validating timeframe: {timeframe}")
        timeframes = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1,
        }
        return timeframes.get(timeframe)

    def _validate_reverse_timeframe(self, timeframe):
        """Validates the timeframe string."""
        timeframes = {
            "mt5.TIMEFRAME_M1": "M1", "mt5.TIMEFRAME_M5": "M5", "mt5.TIMEFRAME_M15": "M15",
            "mt5.TIMEFRAME_M30": "M30", "mt5.TIMEFRAME_H1": "H1", "mt5.TIMEFRAME_H4": "H4",
            "mt5.TIMEFRAME_D1": "D1", "mt5.TIMEFRAME_W1": "W1", "mt5.TIMEFRAME_MN1": "MN1",
        }
        return timeframes.get(timeframe)
    
    def set_mql_timezone(self, year, month, day, timezone):
        """Converts a date into a timezone-aware datetime object."""
        try:
            return pytz.timezone(timezone).localize(datetime(year, month, day))
        except Exception as e:
            logger.error(f"Timezone conversion error: {e}")
            return None
