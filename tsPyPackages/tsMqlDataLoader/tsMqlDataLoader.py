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
        self.data_params = {}
        self._initialize_mql()
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()
        self._set_global_parameters(kwargs)
        # ✅ Fetch values safely from kwargs or default to self.data_params
        self.lp_utc_from = kwargs.get('lp_utc_from', self.data_params.get('lp_utc_from'))
        self.lp_utc_to = kwargs.get('lp_utc_to', self.data_params.get('lp_utc_to'))

    
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
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5 module: {e}")

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        param_sections = ["base", "data", "ml", "mltune", "app"]
        for section in param_sections:
            setattr(self, f"{section}_params", self.params.get(section, {}))
        
        # Application parameters
        self.mp_app_primary_symbol = kwargs.get('mp_app_primary_symbol', self.app_params.get('mp_app_primary_symbol'))
        self.mp_app_rows = kwargs.get('mp_app_rows', self.app_params.get('mp_app_rows'))
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.base_params.get('mp_glob_data_path'))
        self.mp_data_filename1_merge =  self.mp_app_primary_symbol + "_" + self.data_params.get('mp_data_filename1') + ".csv"
        self.mp_data_filename2_merge = self.mp_app_primary_symbol + "_" + self.data_params.get('mp_data_filename2') + ".csv"

        logger.info(f"Data path: {self.mp_glob_data_path}")
        logger.info(f"Primary symbol: {self.mp_app_primary_symbol}")
        logger.info(f"Rows to fetch: {self.mp_app_rows}")
        logger.info(f"Filename1: {self.mp_data_filename1_merge}")
        logger.info(f"Filename2: {self.mp_data_filename2_merge}")

       
        
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
            self.datasrc= self.mp_data_filename1_merge
            logger.info(f"Loading data from file: {self.datasrc}")
            loadfile['file_ticks'] = self._load_from_file(self.datasrc)
        if getattr(self, "mp_data_loadfilerates", False):
            self.datasrc=self.mp_data_filename2_merge
            logger.info(f"Loading data from file: {self.datasrc}")
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
            
            valid_timeframe = self._validate_timeframe(self.mp_data_timeframe)
            if valid_timeframe is None:
                raise ValueError(f"Invalid timeframe: {self.mp_data_timeframe}")
            
            logger.info(f"Fetching {apitype} data from MetaTrader5 API")
            if apitype == 'ticks':
                data = mt5.copy_ticks_from(self.mp_app_primary_symbol, self.lp_utc_from, self.mp_app_rows, mt5.COPY_TICKS_ALL)
            else:
                data = mt5.copy_rates_from(self.mp_app_primary_symbol, valid_timeframe, self.lp_utc_from, self.mp_app_rows)
               
            
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"No {apitype} data found for {self.mp_app_primary_symbol}.")
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
        try:
            df = pd.read_csv(filepath, sep=",", nrows=getattr(self, "mp_data_rowcount", None), low_memory=False)
            if "vol3" in df.columns:
                df.drop("vol3", axis=1, inplace=True)
            if df.empty:
                logger.warning(f"No data found in file: {filename}")
            else:
                logger.info(f"File data received: {len(df)} rows from {filename}")
            return df
        except Exception as e:
            logger.error(f"File load exception: {e}")
            return pd.DataFrame()
    
    def _validate_timeframe(self, timeframe):
        """Validate and return a valid MetaTrader5 timeframe."""
        if 'mt5' not in globals():
            logger.error("MetaTrader5 module not loaded.")
            return None
        
        timeframes = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }
        return timeframes.get(timeframe)
    
    def set_mql_timezone(self, year, month, day, timezone):
        """Convert a date into a timezone-aware datetime object."""
        try:
            return pytz.timezone(timezone).localize(datetime(year, month, day))
        except Exception as e:
            logger.error(f"Timezone conversion error: {e}")
            return None
