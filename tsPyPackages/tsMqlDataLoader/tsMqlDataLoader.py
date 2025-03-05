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
        self.lp_data_rows = kwargs.get('lp_data_rows', self.params.get('data', {}).get('"mp_data_rows', 1000))
        self.lp_data_rowcount = kwargs.get('lp_data_rowcount', self.params.get('data', {}).get('mp_data_rowcount', 10000))
        self.lp_timeframe = kwargs.get('lp_timeframe', self.params.get('app', {}).get('mp_app_timeframe', 'mt5.TIMEFRAME_H4'))
       

        logger.info(f"UTC from: {self.lp_utc_from}")
        logger.info(f"UTC to: {self.lp_utc_to}")
        logger.info(f"Timeframe: {self.lp_timeframe}")
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}")
        logger.info(f"Rows to fetch: {self.lp_data_rows}")
        logger.info(f"Row count: {self.lp_data_rowcount}")
        logger.info(f"Timeframe: {self.lp_timeframe}")

        self._set_global_parameters(kwargs)  # Now safe to call

        # Debugging logs
        logger.debug(f"kwargs: {kwargs}")
        logger.debug(f"self.local_data_params: {self.local_data_params}")

        logger.info(f"UTC from: {self.lp_utc_from}")
        logger.info(f"UTC to: {self.lp_utc_to}")
        logger.info(f"Timeframe: {self.lp_timeframe}")
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}")
        logger.info(f"Rows to fetch: {self.lp_data_rows}")

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
        self.df = kwargs.get('df', pd.DataFrame())
        self.df_name = kwargs.get('df_name', 'df_name')

        if self.df_name == 'df_api_ticks':
            self.df = self._fetch_api_data('ticks') if self.mp_data_loadapiticks else pd.DataFrame()
        elif self.df_name == 'df_api_rates':
            self.df = self._fetch_api_data('rates') if self.mp_data_loadapirates else pd.DataFrame()
        elif self.df_name == 'df_file_ticks':
            self.df = self._load_from_file(self.mp_data_filename1_merge) if self.mp_data_loadfileticks else pd.DataFrame()
        elif self.df_name == 'df_file_rates':
            self.df = self._load_from_file(self.mp_data_filename2_merge) if self.mp_data_loadfilerates else pd.DataFrame()
        return self.df

    def _fetch_api_data(self, apitype=''):
        """Fetch data from MetaTrader5 API."""
        try:
            if 'mt5' not in globals() or not mt5.initialize():
                logger.error("MetaTrader5 module not initialized.")
                return pd.DataFrame()

           
           
           
            logger.info(f"Fetching {apitype} data from MetaTrader5 API")
            if apitype == 'ticks':
                logger.info(f"Api ticks: Fetching Symbol {self.lp_app_primary_symbol} with rows {self.lp_data_rows} of ticks from {self.lp_utc_from} to {self.lp_utc_to}")
                logger.info(f"Api ticks: FetchingTimeframe {self.lp_timeframe} ")
                #Tested ok:  ticks2=mt5.copy_ticks_from(lp_app_primary_symbol, lp_utc_from, lp_data_rows, mt5.COPY_TICKS_ALL)
                data = mt5.copy_ticks_from(self.lp_app_primary_symbol, self.lp_utc_from, self.lp_data_rows, mt5.COPY_TICKS_ALL)
            elif apitype == 'rates':
                logger.info(f"Api rates: Fetching Symbol {self.lp_app_primary_symbol} with rows {self.lp_data_rows} of rates from {self.lp_utc_from} to {self.lp_utc_to}")
                logger.info(f"Api rates: FetchingTimeframe {self.lp_timeframe} ")
                #Tested ok:  rates2 = mt5.copy_rates_from(lp_app_primary_symbol, mt5.TIMEFRAME_H4, lp_utc_from, lp_data_rows)
                data = mt5.copy_rates_from(self.lp_app_primary_symbol,mt5.TIMEFRAME_H4, self.lp_utc_from, self.lp_data_rows)
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
            return pd.read_csv(filepath,nrows=self.lp_data_rowcount)
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

    def reduce_data(self, df):
        """Reduce data to a specific number of rows."""
        return df.head(self.lp_data_rows)


    