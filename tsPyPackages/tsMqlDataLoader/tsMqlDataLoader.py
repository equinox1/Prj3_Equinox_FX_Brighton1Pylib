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


class CDataLoader:
    """Class to manage and load market data parameters with override capability."""

    def __init__(self,params, **kwargs):
        if loadmql:
               import MetaTrader5 as mt5
               self.mt5 = mt5
        else:
            self.mt5 = None
            
        # global parameters
        self.params = params
        self.general_params = self.params['genparams']
        self.data_params = self.params['dataparams']
       
        self._initialise_general_params(kwargs)
        self._initialise_data_params(kwargs)
        self._file_params()

    def _initialise_general_params(self, kwargs):
        """initialises parameters with overrides if provided."""
        self.general_params = self.params['genparams']

        #global parameters
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.general_params.get('mp_glob_data_path'))
        logger.info(f"Data path: {self.mp_glob_data_path}")
        

    def _initialise_data_params(self, kwargs):
        """initialises data parameters with overrides if provided."""
        self.data_params = self.params['dataparams']

         # data parameters

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
        self.mp_data_rows = kwargs.get('mp_data_rows', self.data_params.get('mp_data_rows'))
        self.mp_data_rowcount = kwargs.get('mp_data_rowcount', self.data_params.get('mp_data_rowcount'))
        self.mp_data_filename1 = kwargs.get('mp_data_filename1', self.data_params.get('mp_data_filename1'))
        self.mp_data_filename2 = kwargs.get('mp_data_filename2', self.data_params.get('mp_data_filename2'))
        self.mp_data_symbol = kwargs.get('mp_data_symbol', self.data_params.get('mp_data_symbol'))
        self.mp_data_utc_from = kwargs.get('mp_data_utc_from', self.data_params.get('mp_data_utc_from'))
        self.mp_data_utc_to = kwargs.get('mp_data_utc_to', self.data_params.get('mp_data_utc_to'))
        self.mp_data_custom_input_keyfeat = kwargs.get('mp_data_custom_input_keyfeat', self.data_params.get('mp_data_custom_input_keyfeat'))
        self.mp_data_custom_output_label = kwargs.get('mp_data_custom_output_label', self.data_params.get('mp_data_custom_output_label'))


    def _file_params(self):
        """Logs initialised parameters."""
        logger.info(f"mp_data_path: {self.mp_glob_data_path}")
        logger.info(f"Data file value1: {self.mp_data_filename1}")
        logger.info(f"Data file value2: {self.mp_data_filename2}")

    def load_data(self, **kwargs):
        """Loads market data from API or files."""
        rates = {'api_ticks': pd.DataFrame(), 'api_rates': pd.DataFrame(), 'file_ticks': pd.DataFrame(), 'file_rates': pd.DataFrame()}
        
        if loadmql:
            if self.mp_data_loadapiticks:
                rates['api_ticks'] = self._fetch_api_ticks()
            if self.mp_data_loadapirates:
                rates['api_rates'] = self._fetch_api_rates()
        
        if self.mp_data_loadfileticks:
            rates['file_ticks'] = self._load_from_file(self.mp_data_file_value1)
        if self.mp_data_loadfilerates:
            rates['file_rates'] = self._load_from_file(self.mp_data_file_value2)
        
        return rates
    
    def _fetch_api_ticks(self):
        """Fetches tick data from MetaTrader5 API."""
        try:
            logger.info("Fetching tick data from MetaTrader5 API")
            data = mt5.copy_ticks_from(self.mp_data_symbol, self.mp_data_utc_to, self.mp_data_rows, mt5.COPY_TICKS_ALL)
            df = pd.DataFrame(data)
            logger.info(f"API tick data received: {len(df)} rows" if not df.empty else "No tick data found")
            return df
        except Exception as e:
            logger.error(f"MT5 API tick data exception: {e}")
            return pd.DataFrame()
    
    def _fetch_api_rates(self):
        """Fetches rate data from MetaTrader5 API."""
        try:
            valid_timeframe = self._validate_reverse_timeframe(self.mp_data_timeframe)
            if valid_timeframe is None:
                raise ValueError(f"Invalid timeframe: {self.mp_data_timeframe}")
            
            logger.info("Fetching rate data from MetaTrader5 API")
            logger.info(f"DataPath: {self.mp_glob_data_path}")
            logger.info(f"Symbol: {self.mp_data_symbol}")
            logger.info(f"Timeframe: {valid_timeframe}")
            logger.info(f"UTC to: {self.mp_data_utc_to}")
            logger.info(f"Rows: {self.mp_data_rows}")

            data = mt5.copy_rates_from(self.mp_data_symbol, valid_timeframe, self.mp_data_utc_to, self.mp_data_rows)
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
