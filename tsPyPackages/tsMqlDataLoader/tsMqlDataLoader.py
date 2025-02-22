#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: tsMqlDataLoader.py
# Description: Login to Metatrader and manage market data loading.
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
        self.data_params = {}  
        self._initialize_mql()

        self.params = self.env.all_params()

        # Ensure default values before calling _set_global_parameters
        self.mp_data_filename1 =  self.params.get('data', {}).get('mp_data_filename1', 'default_filename1.csv')
        self.mp_data_filename2 =  self.params.get('data', {}).get('mp_data_filename2', 'default_filename2.csv')
        
        self._set_global_parameters(kwargs)  # Now call this method after initializing needed attributes

        # Debugging logs
        logger.debug(f"kwargs: {kwargs}")
        logger.debug(f"self.data_params: {self.data_params}")

        self.lp_utc_from = kwargs.get('lp_utc_from', self.data_params.get('mp_data_utc_from', None))
        self.lp_utc_to = kwargs.get('lp_utc_to', self.data_params.get('mp_data_utc_to', None))
            
        self.lp_timeframe = kwargs.get('lp_timeframe', self.data_params.get('mp_data_timeframe', None))
        self.lp_app_primary_symbol = kwargs.get('lp_app_primary_symbol', self.params.get('app', {}).get('mp_app_primary_symbol', 'Unknown'))
        self.lp_app_rows = kwargs.get('lp_app_rows', self.params.get('app', {}).get('mp_app_rows', 1000))

        logger.info(f"UTC from: {self.lp_utc_from}")
        logger.info(f"UTC to: {self.lp_utc_to}")
        logger.info(f"Timeframe: {self.lp_timeframe}")
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}")
        logger.info(f"Rows to fetch: {self.lp_app_rows}")
        logger.info(f"Data filename1: {self.mp_data_filename1}")
        logger.info(f"Data filename2: {self.mp_data_filename2}")

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
                    logger.error("Failed to initialize MetaTrader5 module.")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5 module: {e}")

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        param_sections = ["base", "data", "ml", "mltune", "app"]
        for section in param_sections:
            setattr(self, f"{section}_params", self.params.get(section, {}))

        # Application parameters
        self.lp_app_primary_symbol = kwargs.get('lp_app_primary_symbol', self.params.get('app', {}).get('mp_app_primary_symbol', 'Unknown'))
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_data_path', 'Mql5Data'))

        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

       
        # Data parameters
        for key, default in self.data_params.items():
            setattr(self, key, kwargs.get(key, default))

    def load_data(self, **kwargs):
        """Load market data from API or files."""
        loadfile = {'api_ticks': pd.DataFrame(), 'api_rates': pd.DataFrame(),
                    'file_ticks': pd.DataFrame(), 'file_rates': pd.DataFrame()}
        
        if self.loadmql:
            if getattr(self, "mp_data_loadapiticks", False):
                loadfile['api_ticks'] = self._fetch_api_ticks()
            if getattr(self, "mp_data_loadapirates", False):
                loadfile['api_rates'] = self._fetch_api_rates()
        
        if getattr(self, "mp_data_loadfileticks", False):
            logger.info(f"Loading filename: {self.mp_data_filename1_merge}")
            self.datasrc = self.mp_data_filename1_merge
            loadfile['file_ticks'] = self._load_from_file(self.datasrc)

        if getattr(self, "mp_data_loadfilerates", False):
            self.datasrc = self.mp_data_filename2_merge
            loadfile['file_rates'] = self._load_from_file(self.datasrc)
        
        return loadfile
    
    def _fetch_api_ticks(self):
        return self._fetch_api_data(apitype='ticks')
    
    def _fetch_api_rates(self):
        return self._fetch_api_data(apitype='rates')
    
    def _fetch_api_data(self, apitype='rates'):
        """Fetch data from MetaTrader5 API."""
        try:
            if 'mt5' not in globals():
                logger.error("MetaTrader5 module not loaded.")
                return pd.DataFrame()
            
            if not mt5.initialize():
                logger.error("Failed to initialize MetaTrader5 connection.")
                return pd.DataFrame()

            valid_timeframe = self.lp_timeframe if self.lp_timeframe else mt5.TIMEFRAME_M1

            logger.info(f"Fetching {apitype} data from MetaTrader5 API")
            if apitype == 'ticks':
               logger.info(f"Ticks: Fetching {apitype} data from MetaTrader5 API")
               logger.info(f"Ticks:Primary symbol: {self.lp_app_primary_symbol}")
               logger.info(f"Ticks:UTC from: {self.lp_utc_from}")
               logger.info(f"Ticks:Rows to fetch: {self.lp_app_rows}")
               logger.info(f"Ticks:Copy ticks all: {mt5.COPY_TICKS_ALL}")
               data = mt5.copy_ticks_from(self.lp_app_primary_symbol, self.lp_utc_from, self.lp_app_rows, mt5.COPY_TICKS_ALL)
            else:
               logger.info(f"Rates: Fetching {apitype} data from MetaTrader5 API")
               logger.info(f"Rates:Primary symbol: {self.lp_app_primary_symbol}")
               logger.info(f"Rates:Timeframe: {valid_timeframe}")
               logger.info(f"Rates:UTC from: {self.lp_utc_from}")
               logger.info(f"Rates:Rows to fetch: {self.lp_app_rows}")
               data = mt5.copy_rates_from(self.lp_app_primary_symbol, valid_timeframe, self.lp_utc_from, self.lp_app_rows)

            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No {apitype} data found for {self.lp_app_primary_symbol}.")
            else:
                logger.info(f"Fetched {len(df)} {apitype} rows from API.")
            return df
        except Exception as e:
            logger.error(f"MT5 API {apitype} exception: {e}")
            return pd.DataFrame()
    
    def _load_from_file(self, filename):
        """Load data from a CSV file."""
        if not filename:
            logger.warning("No filename provided for file loading.")
            return pd.DataFrame()
        
        filepath = os.path.join(self.mp_glob_data_path, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
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