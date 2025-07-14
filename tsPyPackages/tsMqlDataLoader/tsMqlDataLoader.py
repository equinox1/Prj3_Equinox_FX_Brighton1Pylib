#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Filename: tsMqlDataLoader.py
# Description: Login to MetaTrader and manage market data loading.
# Author: Tony Shepherd - Xercescloud
# Date: 2025-01-24
# Version: 1.2 (Refined DataLoader __init__ and _set_global_parameters to handle kwargs properly)
# License: MIT License (Optional)


import logging
import pandas as pd
import numpy as np
import pytz
import os
from datetime import datetime
from tsMqlPlatform import run_platform, platform_checker, get_config
from tsMqlEnvMgr import CMqlEnvMgr
# Removed duplicate import logging

# -- Set up global logging (from tsMqlSetup) --
# Removed: from tsMqlSetup import CMqlSetup # This import is not needed here

# Load configuration
from tsMqlOverrides import CMqlOverrides  # ✅ Add this line
# Removed: from tsMqlPlatform import run_platform, platform_checker, get_config # Already imported above
# Removed: from tsMqlEnvMgr import CMqlEnvMgr # Already imported above

mql_overrides = CMqlOverrides()
all_params = mql_overrides.env.all_params()
app_params = all_params.get("app", {})
tune_params = all_params.get('mltune', {})
base_params = all_params.get("base", {})

backend_for_log = os.environ.get('BACKEND', tune_params.get('backend', 'pytorch'))

# Corrected: Assign the logger instance to the 'logger' variable
logger = logging.getLogger(__name__)

gtuner_model = tune_params.get('tuner_type', 'hyperband')  # Default ,randomsearch, bayesian, hyperband
backend = tune_params.get('backend', 'tensorflow')  #tensorflow, pytorch
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

global_logdir = app_params.get('mp_glob_base_log_path', './Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

mt5 = None # Initialize mt5 as None, will be set globally if available

class CDataLoader:
    """Class to manage and load market data with override capability."""

    def __init__(self, symbol, timeframe, start_date_str, end_date_str, data_path, **kwargs):
        self.env = CMqlEnvMgr()
        self.local_data_params = {}  
        self._initialize_mql()

        self.params = self.env.all_params()
        data_config = self.params.get('data', {}) # Get the 'data' section of the config

        # Assign positional arguments directly
        self.lp_app_primary_symbol = symbol
        self.lp_timeframe = timeframe
        self.lp_start_date_str = start_date_str
        self.lp_end_date_str = end_date_str
        self.mp_glob_base_data_path = data_path # Using this directly for data_path

        # Parse start and end dates from strings into timezone-aware datetime objects
        # Assuming dates are in YYYY-MM-DD format and should be treated as UTC
        try:
            # Set lp_utc_from using the provided start_date_str
            self.lp_utc_from = pytz.utc.localize(datetime.strptime(self.lp_start_date_str, '%Y-%m-%d'))
            # lp_utc_to is mostly for logging purposes, as copy_ticks_from/copy_rates_from
            # use a start date and a count.
            self.lp_utc_to = pytz.utc.localize(datetime.strptime(self.lp_end_date_str, '%Y-%m-%d'))
        except ValueError as e:
            logger.error(f"Error parsing date strings '{self.lp_start_date_str}' or '{self.lp_end_date_str}': {e}. "
                         "Ensure dates are in YYYY-MM-DD format. Falling back to current UTC time for data fetch.")
            # Fallback to current UTC time if parsing fails
            self.lp_utc_from = datetime.utcnow().replace(tzinfo=pytz.utc)
            self.lp_utc_to = datetime.utcnow().replace(tzinfo=pytz.utc)

        # Set specific data loading parameters, prioritizing kwargs over loaded config
        self.lp_data_rows = kwargs.get('mp_data_rows', data_config.get('mp_data_rows', 1000))
        self.lp_data_rowcount = kwargs.get('mp_data_rowcount', data_config.get('mp_data_rowcount', 10000))
        self.mp_data_loadapiticks = kwargs.get('mp_data_loadapiticks', data_config.get('mp_data_loadapiticks', True))
        self.mp_data_loadapirates = kwargs.get('mp_data_loadapirates', data_config.get('mp_data_loadapirates', True))
        self.mp_data_loadfileticks = kwargs.get('mp_data_loadfileticks', data_config.get('mp_data_loadfileticks', True))
        self.mp_data_loadfilerates = kwargs.get('mp_data_loadfilerates', data_config.get('mp_data_loadfilerates', True))
        
        # Filenames for local data loading, also from config but not meant to be overridden by kwargs here
        self.mp_data_filename1 = data_config.get('mp_data_filename1', 'default_filename1.csv')
        self.mp_data_filename2 = data_config.get('mp_data_filename2', 'default_filename2.csv')

        self._set_global_parameters() # Call without kwargs, as kwargs are now handled directly here

        # Debugging logs
        logger.debug(f"CDataLoader initialized with kwargs: {kwargs}")
        logger.info(f"UTC from: {self.lp_utc_from}")
        logger.info(f"UTC to: {self.lp_utc_to}")
        logger.info(f"Timeframe: {self.lp_timeframe}")
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}")
        logger.info(f"Rows to fetch (API): {self.lp_data_rows}")
        logger.info(f"lp_timeframe: {self.lp_timeframe}") # This seems redundant with self.lp_timeframe, but keeping for now.
        logger.info(f"lp_data_rowcount (File): {self.lp_data_rowcount}")
        logger.info(f"mp_data_filename1: {self.mp_data_filename1}")
        logger.info(f"mp_data_filename2: {self.mp_data_filename2}")
        logger.info(f"Load API ticks: {self.mp_data_loadapiticks}")
        logger.info(f"Load API rates: {self.mp_data_loadapirates}")
        logger.info(f"Load file ticks: {self.mp_data_loadfileticks}")
        logger.info(f"Load file rates: {self.mp_data_loadfilerates}")


    def _initialize_mql(self):
        """Initialize MetaTrader5 module and check platform.
           This method assumes MetaTrader5 is already initialized by the calling script
           and only focuses on importing the module and checking platform state.
        """
        pchk = run_platform.RunPlatform()
        self.os_platform = platform_checker.get_platform()
        self.loadmql = pchk.check_mql_state()
        logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")

        if self.loadmql:
            try:
                global mt5
                import MetaTrader5 as mt5
                # Removed the mt5.initialize() call from here.
                # It is assumed that mt5.initialize() is called once at the
                # entry point of the application (e.g., in chief/worker scripts)
                logger.info("MetaTrader5 module expected to be initialized by caller.")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5 module: {e}. "
                             f"Ensure MetaTrader5 package is installed and accessible.")

    def _set_global_parameters(self):
        """Set configuration parameters from environment."""
        param_sections = ["base", "data", "ml", "mltune", "app"]
        for section in param_sections:
            setattr(self, f"{section}_params", self.params.get(section, {}))

        # These parameters are now derived from self.params directly
        # and not expected from kwargs via this method.
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"
        
        # Removed redundant assignments that were previously using kwargs.get() within this method
        # as they are now handled directly in __init__ with proper prioritization.

        logger.info(f"Data path (from base_params if not overridden in __init__): {self.mp_glob_base_data_path}")
        logger.info(f"Data filename1_merge: {self.mp_data_filename1_merge}")
        logger.info(f"Data filename2_merge: {self.mp_data_filename2_merge}")
        # The loadapiticks, etc., are now logged from __init__ where they are first determined.


    def load_data(self, **kwargs):
        """Load market data from API or files and return all DataFrames."""
        df = kwargs.get('df', pd.DataFrame()) # Use local df, not self.df
        df_name = kwargs.get('df_name', 'df_name')

        if df_name == 'df_api_ticks':
            df = self._fetch_api_data('ticks') if self.mp_data_loadapiticks else pd.DataFrame()
        elif df_name == 'df_api_rates':
            df = self._fetch_api_data('rates') if self.mp_data_loadapirates else pd.DataFrame()
        elif df_name == 'df_file_ticks':
            df = self._load_from_file(self.mp_data_filename1_merge) if self.mp_data_loadfileticks else pd.DataFrame()
        elif df_name == 'df_file_rates':
            df = self._load_from_file(self.mp_data_filename2_merge) if self.mp_data_loadfilerates else pd.DataFrame()
        return df # Return the local df

    def _fetch_api_data(self, apitype=''):
        """Fetch data from MetaTrader5 API."""
        try:
            # Check if mt5 is imported and available in globals before attempting to use it
            if 'mt5' not in globals() or mt5 is None:
                logger.error("MetaTrader5 module not imported or available in global scope.")
                return pd.DataFrame()
            
            # Since mt5.initialize() is now handled by the caller, we only check if it's available.
            # No need to call mt5.initialize() here again.
            
            logger.info(f"Fetching {apitype} data from MetaTrader5 API")
            if apitype == 'ticks':
                logger.info(f"Api ticks: Fetching Symbol {self.lp_app_primary_symbol} with rows {self.lp_data_rows} of ticks from {self.lp_utc_from} to {self.lp_utc_to}")
                logger.info(f"Api ticks: FetchingTimeframe {self.lp_timeframe} ")
                # Use self.lp_utc_from (parsed from start_date_str) for fetching
                logger.info(f"Api Running command: mt5.copy_ticks_from({self.lp_app_primary_symbol}, {self.lp_utc_from}, {self.lp_data_rows}, mt5.COPY_TICKS_ALL)")
                data = mt5.copy_ticks_from(self.lp_app_primary_symbol, self.lp_utc_from, self.lp_data_rows, mt5.COPY_TICKS_ALL)
            elif apitype == 'rates':
                logger.info(f"Api rates: Fetching Symbol {self.lp_app_primary_symbol} with rows {self.lp_data_rows} of rates from {self.lp_utc_from} to {self.lp_utc_to}")
                logger.info(f"Api rates: FetchingTimeframe {self.lp_timeframe} ")
                # Use self.lp_utc_from (parsed from start_date_str) for fetching
                logger.info(f"Api Running command: mt5.copy_rates_from({self.lp_app_primary_symbol}, {self.lp_timeframe}, {self.lp_utc_from}, {self.lp_data_rows})")
                data = mt5.copy_rates_from(self.lp_app_primary_symbol,self.lp_timeframe, self.lp_utc_from, self.lp_data_rows)
            
            if data is None:
                logger.warning(f"MT5 API {apitype} returned None for data. Check symbol, timeframe, and date range in MT5 terminal.")
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"MT5 API {apitype} exception: {e}")
            return pd.DataFrame()

    def _load_from_file(self, filename):
        """Load data from a CSV file."""
        # Ensure self.mp_glob_base_data_path is a string for os.path.join
        filepath = os.path.join(str(self.mp_glob_base_data_path), filename)
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

    def run_dataloader_services(self):
        """Run the data loader services and return a dictionary of DataFrames."""
        logger.info("Running data loader services...")
        
        dfs = {
            "df_api_ticks": pd.DataFrame(),
            "df_api_rates": pd.DataFrame(),
            "df_file_ticks": pd.DataFrame(),
            "df_file_rates": pd.DataFrame()
        }
        
        # Load data based on flags set in __init__
        if self.mp_data_loadapiticks:
            dfs["df_api_ticks"] = self.load_data(df_name="df_api_ticks")
            logger.info(f"df_api_ticks loaded. Shape: {dfs['df_api_ticks'].shape}")

        if self.mp_data_loadapirates:
            dfs["df_api_rates"] = self.load_data(df_name="df_api_rates")
            logger.info(f"df_api_rates loaded. Shape: {dfs['df_api_rates'].shape}")

        if self.mp_data_loadfileticks:
            dfs["df_file_ticks"] = self.load_data(df_name="df_file_ticks")
            logger.info(f"df_file_ticks loaded. Shape: {dfs['df_file_ticks'].shape}")

        if self.mp_data_loadfilerates:
            dfs["df_file_rates"] = self.load_data(df_name="df_file_rates")
            logger.info(f"df_file_rates loaded. Shape: {dfs['df_file_rates'].shape}")

        return dfs