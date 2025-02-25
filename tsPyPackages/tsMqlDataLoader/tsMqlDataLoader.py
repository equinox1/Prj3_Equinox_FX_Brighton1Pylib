#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: tsMqlDataLoader.py
# Description: Login to MetaTrader and manage market data loading.
# Author: Tony Shepherd - Xercescloud
# Date: 2025-01-24
# Version: 1.1
# License: MIT License (Optional)

import logging
import pandas as pd
import numpy as np
import pytz
import os
from datetime import datetime
from tsMqlPlatform import run_platform, platform_checker, logger, config
from tsMqlEnvMgr import CMqlEnvMgr

class CDataLoader:
    """Class to manage and load market data with override capability."""

    def __init__(self, **kwargs):
        self.env = CMqlEnvMgr()
        self.local_data_params = {}  
        self._initialize_mql()

        self.params = self.env.all_params()

        # Ensure default values before calling _set_global_parameters
        self.mp_data_filename1 = self.params.get('data', {}).get('mp_data_filename1', 'default_filename1.csv')
        self.mp_data_filename2 = self.params.get('data', {}).get('mp_data_filename2', 'default_filename2.csv')

        default_utc_from = datetime.utcnow()
        default_utc_to = datetime.utcnow()

        self.lp_utc_from = kwargs.get('lp_utc_from', default_utc_from)
        self.lp_utc_to = kwargs.get('lp_utc_to', default_utc_to)
        self.lp_app_primary_symbol = kwargs.get('lp_app_primary_symbol', self.params.get('app', {}).get('mp_app_primary_symbol', 'EURUSD'))
        self.lp_app_rows = kwargs.get('lp_app_rows', self.params.get('app', {}).get('mp_app_rows', 1000))
        self.lp_timeframe = kwargs.get('lp_timeframe', self.params.get('app', {}).get('mp_app_timeframe', 'H4'))

        self._set_global_parameters(kwargs)  # Now safe to call

        # Debugging logs
        logger.debug(f"kwargs: {kwargs}")
        logger.debug(f"self.local_data_params: {self.local_data_params}")

        logger.info(f"UTC from: {self.lp_utc_from}")
        logger.info(f"UTC to: {self.lp_utc_to}")
        logger.info(f"Timeframe: {self.lp_timeframe}")
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}")
        logger.info(f"Rows to fetch: {self.lp_app_rows}")

    def _initialize_mql(self):
        """Initialize MetaTrader5 module and check platform."""
        pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = pchk.check_mql_state()
        logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")

        if self.loadmql:
            try:
                global mt5
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MetaTrader5 module. Error: {mt5.last_error()}")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5 module: {e}")

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        param_sections = ["base", "data", "ml", "mltune", "app"]
        for section in param_sections:
            setattr(self, f"{section}_params", self.params.get(section, {}))

        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_data_path', 'Mql5Data'))

        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', self.params.get('data', {}).get('mp_data_loadapiticks', True))
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', self.params.get('data', {}).get('mp_data_loadapirates', True))
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', self.params.get('data', {}).get('mp_data_loadfileticks', True))
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', self.params.get('data', {}).get('mp_data_loadfilerates', True))

        logger.info(f"Data path: {self.mp_glob_data_path}")
        logger.info(f"Data filename1: {self.mp_data_filename1_merge}")
        logger.info(f"Data filename2: {self.mp_data_filename2_merge}")
        logger.info(f"Load API ticks: {self.mp_data_loadapiticks}")
        logger.info(f"Load API rates: {self.mp_data_loadapirates}")
        logger.info(f"Load file ticks: {self.mp_data_loadfileticks}")
        logger.info(f"Load file rates: {self.mp_data_loadfilerates}")

    def load_data(self, **kwargs):
        """Load market data from API or files and return all DataFrames."""
        
        df_api_ticks = self._fetch_api_data('ticks') if self.mp_data_loadapiticks else pd.DataFrame()
        df_api_rates = self._fetch_api_data('rates') if self.mp_data_loadapirates else pd.DataFrame()
        df_file_ticks = self._load_from_file(self.mp_data_filename1_merge) if self.mp_data_loadfileticks else pd.DataFrame()
        df_file_rates = self._load_from_file(self.mp_data_filename2_merge) if self.mp_data_loadfilerates else pd.DataFrame()
        
        logger.info("Api Tick Rows loaded ",self.row_count(df_api_ticks))
        logger.info("Api Rates Rows loaded ",self.row_count(df_api_rates))
        logger.info("File Tick Rows loaded ",self.row_count(df_file_ticks))
        logger.info("File Rates Rows loaded ",self.row_count(df_file_rates))


        if all(df.empty for df in [df_api_ticks, df_api_rates, df_file_ticks, df_file_rates]):
            logger.error("No data loaded from API or files.")

        return df_api_ticks, df_api_rates, df_file_ticks, df_file_rates

    def _fetch_api_data(self, apitype=''):
        """Fetch data from MetaTrader5 API."""
        try:
            if 'mt5' not in globals() or not mt5.initialize():
                logger.error("MetaTrader5 module not initialized.")
                return pd.DataFrame()

            valid_timeframe = getattr(mt5, f"TIMEFRAME_{self.lp_timeframe}", mt5.TIMEFRAME_M1)

            logger.info(f"Fetching {apitype} data from MetaTrader5 API")
            if apitype == 'ticks':
                data = mt5.copy_ticks_from(self.lp_app_primary_symbol, self.lp_utc_from, self.lp_app_rows, mt5.COPY_TICKS_ALL)
            elif apitype == 'rates':
                data = mt5.copy_rates_from(self.lp_app_primary_symbol, valid_timeframe, self.lp_utc_from, self.lp_app_rows)
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"MT5 API {apitype} exception: {e}")
            return pd.DataFrame()

    def _load_from_file(self, filename):
        """Load data from a CSV file."""
        filepath = os.path.join(self.mp_glob_data_path, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return pd.DataFrame()


    def set_mql_timezone(self, year, month, day, timezone):
            """Convert a date into a timezone-aware datetime object."""
            try:
                  return pytz.timezone(timezone).localize(datetime(year, month, day))
            except Exception as e:
                  logger.error(f"Timezone conversion error: {e}")
                  return None

    def row_count(self, df):
        """Return the number of rows in a DataFrame."""
        return len(df.index)
    