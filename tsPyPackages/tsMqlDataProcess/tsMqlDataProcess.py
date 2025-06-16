"""
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlDataProcess/tsMqlDataProcess.py
Description: Simplified, optimized module for loading data, processing DataFrames, and interfacing with MetaTrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 2.4 (Added robust final cleanup for non-numeric columns in wrangling)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import os
import logging


# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides

# -- Load global parameters --
mql_overrides = CMqlOverrides() 
app_params = mql_overrides.env.all_params().get("app", {})
tune_params = mql_overrides.env.all_params().get("mltune", {})

from tsMqlLogService import CMLogServiceSetup
logger = CMLogServiceSetup.initialize_logging(
    role_hint=__name__, 
    loglevel='INFO',
    logfile='tsneuropredict_app.log'
)



gtuner_model = tune_params.get('tuner_type', 'hyperband')
backend = tune_params.get('backend', 'tensorflow')
xerces_servername = app_params.get('xerces_servername', "WINSVRXERCES01")
xerces_server = app_params.get('xerces_server', '192.168.1.103')
xerces_port = app_params.get('xerces_port', 9000)
xerces_logfile = app_params.get('xerces_logfile', 'tsneuropredict_app.log')

global_logdir = app_params.get('LOGDIR', 'Logdir')
global_logfile = app_params.get('LOGFILE', 'xerces_logfile')

# Initialize platform checker (global to avoid re-initialization if used across functions)
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state() # Ensure loadmql is defined
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")


class CDataProcess:
    def __init__(self, df: pd.DataFrame, **kwargs):
        """
        Initialize data processing class.
        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            **kwargs: Additional keyword arguments for configuration.
        """
        self.original_df = df.copy(deep=True) # Store the original DataFrame
        self.df = df # The DataFrame that will be modified during processing

        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)
        self.lp_utc_from = kwargs.get('lp_utc_from', datetime.utcnow())
        self.lp_utc_to = kwargs.get('lp_utc_to', datetime.utcnow())
        self.mp_unit = kwargs.get('UNIT', {})

        # Initialize MetaTrader and environment parameters
        self._initialize_mql() 
        self._set_envmgr_params(kwargs)
        self._set_global_parameters(kwargs) 

        self.lp_app_primary_symbol = kwargs.get(
            'lp_app_primary_symbol',
            self.app_params.get('mp_app_primary_symbol', 'EURUSD')
        )
        self.lp_timeframe = kwargs.get(
            'lp_timeframe',
            self.data_params.get('mp_data_timeframe', 'mt5.TIMEFRAME_M1')
        )
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}, Timeframe: {self.lp_timeframe}")

        self._set_ml_features(kwargs)

        # Mapping definitions for column conversions and reordering
        self.from_to_column_maps = {
            'ticks1': {'time': 'T1_Date', 'bid': 'T1_Bid_Price', 'ask': 'T1_Ask_Price',
                       'last': 'T1_Last Price', 'volume': 'T1_Volume', 'time_msc': 'T1_Time_Msc',
                       'flags': 'T1_Flags', 'volume_real': 'T1_Real_Volume'},
            # For rates1 (API Rates), the 'time' column becomes 'R1_Date' after renaming.
            # This 'R1_Date' is already a timestamp or datetime, no need to merge with itself.
            'rates1': {'time': 'R1_Date', 'open': 'R1_Open', 'high': 'R1_High', 'low': 'R1_Low',
                       'close': 'R1_Close', 'tick_volume': 'R1_Tick_Volume', 'spread': 'R1_spread',
                       'real_volume': 'R1_Real_Volume'},
            'ticks2': {'mDatetime': 'T2_mDatetime', 'Date': 'T2_Date', 'Timestamp': 'T2_Timestamp',
                       'Bid Price': 'T2_Bid_Price', 'Ask Price': 'T2_Ask_Price', 'Last Price': 'T2_Last_Price',
                       'Volume': 'T2_Volume'},
            # For rates2 (File Rates), 'Date' and 'Timestamp' are separate and need merging.
            'rates2': { # Ensure these match your actual file headers and desired names
                'Date': 'R2_Date', 'Timestamp': 'R2_Timestamp', 
                'Open': 'R2_Open', 'High': 'R2_High', 'Low': 'R2_Low', 'Close': 'R2_Close',
                'tick_volume': 'R2_Tick Volume', 'Volume': 'R2_Volume', 'vol2': 'R2_Vol1', 'vol3': 'R2_Vol3'
            }
        }

        # Date format for API data ('f' for from timestamp/unit)
        # Date format for file data ('e' for direct datetime string)
        self.date_columns = {
            # For API data, 'time' is usually a Unix timestamp (seconds).
            'ticks1': ('time', None, 's', 'f'), 
            'rates1': ('time', None, 's', 'f'), # 'time' column, in seconds, needs 'f' conversion
            # For file data, 'Date' is a string like '20230101'
            'ticks2': ('Date', '%Y%m%d', None, 'e'), 
            'rates2': ('Date', '%Y%m%d', None, 'e'),
        }

        # Time format for API data ('a' for direct datetime string)
        # Time format for file data ('a' for direct datetime string)
        self.time_columns = {
            'ticks1': ('time_msc', '%Y%m%d %H:%M:%S.%f', None, 'a'), # time_msc for API ticks
            'rates1': (None, None, None, None), # No separate time column for API rates to convert
            'ticks2': ('Timestamp', '%H:%M:%S', None, 'a'),
            'rates2': ('Timestamp', '%H:%M:%S', None, 'a'),
        }

        # Conversion columns: type 'b' means df.pop(column) and convert.
        # This is for the *result* of a merge or the main datetime column after initial renaming/conversion.
        self.conv_columns = {
            'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', None, 'b'), # T1_Date becomes the primary datetime
            'rates1': ('R1_Date', None, 's', 'b'), # R1_Date (from original 'time' timestamp) becomes primary datetime
            'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', None, 'b'), # T2_mDatetime (after merge) becomes primary
            'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', None, 'b'), # R2_mDatetime (after merge) becomes primary
        }

        # Columns to drop after processing/merging (conv_type 'g')
        self.drop_columns = {
            'ticks1': (None, None, None, 'g', 
                       ['T1_Time_Msc', 'T1_Flags', 'T1_Last Price', 'T1_Real_Volume', 'T1_Volume']),
            'rates1': (None, None, None, 'g', 
                       ['R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume']),
            'ticks2': (None, None, None, 'g', 
                       ['T2_Timestamp', 'T2_Volume', 'T2_Last_Price']),
            'rates2': (None, None, None, 'g', 
                       ['R2_Timestamp', 'R2_Volume', 'R2_Vol1', 'R2_Tick Volume', 'R2_Real_Volume', 'R2_spread']) # Added more for rates2
        }

        # Merge columns: (date_col, time_col, merged_col)
        # API rates (rates1) do NOT need merging as 'time' is a single timestamp.
        self.merge_columns = {
            'ticks1': ('T1_Date', 'T1_Time_Msc', 'T1_mDatetime'), 
            'rates1': (None, None, None), # Explicitly no merge for API rates (rates1)
            'ticks2': ('R2_Date', 'R2_Timestamp', 'R2_mDatetime'), 
            'rates2': ('R2_Date', 'R2_Timestamp', 'R2_mDatetime'),
        }


        self.first_columns = {
            'df_api_ticks': 'T1_mDatetime', # This should be the final merged datetime column
            'df_api_rates': 'R1_Date', # This should be the final timestamp/datetime column
            'df_file_ticks': 'T2_mDatetime', 
            'df_file_rates': 'R2_mDatetime', 
        }

        self.last_columns = {
            'df_api_ticks': ('Close', 'Close_scaled'), 
            'df_api_rates': ('Close', 'Close_scaled'),
            'df_file_ticks': ('Close', 'Close_scaled'),
            'df_file_rates': ('Close', 'Close_scaled'),
        }

        # COLUMN_PARAMS remains largely the same, but values will be applied to the correct columns
        self.COLUMN_PARAMS = {
            "df_api_ticks": {
                'bid_column': 'T1_Bid_Price', 'ask_column': 'T1_Ask_Price', 'column_in': 'T1_Bid_Price',
                'open_column': None, 'high_column': None, 'low_column': None, 'close_column': None,
                'column_out1': self.feature4, 'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods, 'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col, 'ma_col': self.ma_col, 'returns_col': self.returns_col,
                'shift_in': self.shift_in, 'create_label': self.create_label,
                'df1_filter_int': self.data_params.get('df1_filter_int', False),
                'df1_filter_flt': self.data_params.get('df1_filter_flt', False),
                'df1_filter_obj': self.data_params.get('df1_filter_obj', False),
                'df1_filter_dtmi': self.data_params.get('df1_filter_dtmi', False),
                'df1_filter_dtmf': self.data_params.get('df1_filter_dtmf', False),
                'df1_mp_dropna': self.data_params.get('df1_mp_dropna', True),
                'df1_mp_merge': self.data_params.get('df1_mp_merge', True),
                'df1_mp_convert': self.data_params.get('df1_mp_convert', True),
                'df1_mp_drop': self.data_params.get('df1_mp_drop', False)
            },
            "df_api_rates": {
                'bid_column': 'R1_Open', 'ask_column': 'R1_Close', 'column_in': 'R1_Close', # Using Close for main input
                'open_column': 'R1_Open', 'high_column': 'R1_High', 'low_column': 'R1_Low', 'close_column': 'R1_Close',
                'column_out1': self.feature4, 'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods, 'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col, 'ma_col': self.ma_col, 'returns_col': self.returns_col,
                'shift_in': self.shift_in, 'create_label': self.create_label,
                'df2_filter_int': self.data_params.get('df2_filter_int', False),
                'df2_filter_flt': self.data_params.get('df2_filter_flt', False),
                'df2_filter_obj': self.data_params.get('df2_filter_obj', False),
                'df2_filter_dtmi': self.data_params.get('df2_filter_dtmi', False),
                'df2_filter_dtmf': self.data_params.get('df2_filter_dtmf', False),
                'df2_mp_dropna': self.data_params.get('df2_mp_dropna', True),
                'df2_mp_merge': self.data_params.get('df2_mp_merge', True),
                'df2_mp_convert': self.data_params.get('df2_mp_convert', True),
                'df2_mp_drop': self.data_params.get('df2_mp_drop', False)
            },
            "df_file_ticks": {
                'bid_column': 'T2_Bid_Price', 'ask_column': 'T2_Ask_Price', 'column_in': 'T2_Bid_Price',
                'open_column': None, 'high_column': None, 'low_column': None, 'close_column': None,
                'column_out1': self.feature4, 'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods, 'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col, 'ma_col': self.ma_col, 'returns_col': self.returns_col,
                'shift_in': self.shift_in, 'create_label': self.create_label,
                'df3_filter_int': self.data_params.get('df3_filter_int', False),
                'df3_filter_flt': self.data_params.get('df3_filter_flt', False),
                'df3_filter_obj': self.data_params.get('df3_filter_obj', False),
                'df3_filter_dtmi': self.data_params.get('df3_filter_dtmi', False),
                'df3_filter_dtmf': self.data_params.get('df3_filter_dtmf', False),
                'df3_mp_dropna': self.data_params.get('df3_mp_dropna', True),
                'df3_mp_merge': self.data_params.get('df3_mp_merge', True),
                'df3_mp_convert': self.data_params.get('df3_mp_convert', True),
                'df3_mp_drop': self.data_params.get('df3_mp_drop', False)
            },
            "df_file_rates": {
                'bid_column': 'R2_Open', 'ask_column': 'R2_Close', 'column_in': 'R2_Close',
                'open_column': 'R2_Open', 'high_column': 'R2_High', 'low_column': 'R2_Low', 'close_column': 'R2_Close',
                'column_out1': self.feature4, 'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods, 'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col, 'ma_col': self.ma_col, 'returns_col': self.returns_col,
                'shift_in': self.shift_in, 'create_label': self.create_label,
                'df4_filter_int': self.data_params.get('df4_filter_int', False),
                'df4_filter_flt': self.data_params.get('df4_filter_flt', False),
                'df4_filter_obj': self.data_params.get('df4_filter_obj', False),
                'df4_filter_dtmi': self.data_params.get('df4_filter_dtmi', False),
                'df4_filter_dtmf': self.data_params.get('df4_filter_dtmf', False),
                'df4_mp_dropna': self.data_params.get('df4_mp_dropna', True),
                'df4_mp_merge': self.data_params.get('df4_mp_merge', True),
                'df4_mp_convert': self.data_params.get('df4_mp_convert', True),
                'df4_mp_drop': self.data_params.get('df4_mp_drop', False)
            }
        }

    def _initialize_mql(self):
        """Initialize MetaTrader5 if available.
           This method assumes MetaTrader5 is already initialized by the calling script
           and only focuses on importing the module if loadmql is True.
        """
        global mt5 # Declare mt5 as global to be accessible across modules
        # pchk, os_platform, loadmql are assumed to be initialized globally
        # or passed as args if not globally available.
        # Given they are defined outside the class, they should be accessible.
        logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")
        if loadmql: # Use the globally determined loadmql state
            try:
                import MetaTrader5 as mt5 # Import mt5
                # Removed the mt5.initialize() call from here.
                # It is assumed that mt5.initialize() is called once at the
                # entry point of the application (e.g., in chief/worker scripts)
                logger.info("MetaTrader5 module expected to be initialized by caller and imported.")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5: {e}")
        else:
            logger.info("MetaTrader5 module not to be loaded based on loadmql state.")


    def _set_envmgr_params(self, kwargs):
        """Extract environment parameters."""
        override_config = CMqlOverrides()
        self.params = override_config.env.all_params()
        logger.info("Loaded environment parameters.")
        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

    def _set_global_parameters(self, kwargs):
        """Placeholder for global parameter settings.
           Can be used to set any other parameters from kwargs that aren't
           explicitly handled by other _set methods.
        """
        pass

    def _set_ml_features(self, kwargs):
        """Extract and set machine learning features."""
        self.ml_features_config = self.ml_params.get('mp_features', {})
        self.feature4 = self.ml_params.get('feature4', self.ml_features_config.get('feature4', 'Close'))
        self.feature4_scaled = self.ml_params.get('feature4_scaled', self.ml_features_config.get('feature4_scaled', 'Close_Scaled'))
        self.label = self.ml_params.get('Label1', self.ml_features_config.get('Label1', 'Label'))
        logger.info(f"ML features configured: {self.feature4}, {self.feature4_scaled}, {self.label}")
        self.mp_ml_input_keyfeat = self.feature4
        self.mp_ml_input_keyfeat_scaled = self.feature4_scaled
        self.mp_ml_input_label = self.label

        # File parameters
        self.rownumber = self.ml_params.get('mp_rownumber', False)
        self.mp_data_filename1 = self.params.get('data', {}).get('mp_data_filename1', 'default1.csv')
        self.mp_data_filename2 = self.params.get('data', {}).get('mp_data_filename2', 'default2.csv')
        self.lookahead_periods = self.params.get('ml', {}).get('mp_lookahead_periods', 1)
        self.ma_window = self.params.get('ml', {}).get('mp_ml_tf_ma_windowin', 10)
        self.hl_avg_col = self.params.get('ml', {}).get('mp_ml_hl_avg_col', 'HL_Avg')
        self.ma_col = self.params.get('ml', {}).get('mp_ml_ma_col', 'MA')
        self.returns_col = self.params.get('ml', {}).get('mp_ml_returns_col', 'Returns')
        self.shift_in = self.params.get('ml', {}).get('mp_ml_tf_shiftin', 1)
        self.run_avg = self.params.get('ml', {}).get('mp_ml_run_avg', False)
        self.run_avg_scaled = self.params.get('ml', {}).get('mp_ml_run_avg_scaled', False)
        self.log_stationary = self.params.get('ml', {}).get('mp_ml_log_stationary', False)
        self.remove_zeros = self.params.get('ml', {}).get('mp_ml_remove_zeros', False)
        self.last_col = self.params.get('ml', {}).get('mp_ml_last_col', False)
        self.last_col_scaled = self.params.get('ml', {}).get('mp_ml_last_col_scaled', False)
        self.first_col = self.params.get('ml', {}).get('mp_ml_first_col', False)
        self.mp_ml_dropna = self.params.get('ml', {}).get('mp_ml_dropna', False)
        self.mp_ml_dropna_scaled = self.params.get('ml', {}).get('mp_ml_dropna_scaled', False)
        self.create_label = self.params.get('ml', {}).get('mp_ml_create_label', False)
        self.create_label_scaled = self.params.get('ml', {}).get('mp_ml_create_label_scaled', False)
        self.lp_data_rows = kwargs.get('lp_data_rows', self.params.get('data', {}).get('mp_data_rows', 1000))
        self.lp_data_rowcount = kwargs.get('lp_data_rowcount', self.params.get('data', {}).get('mp_data_rowcount', 10000))
        # Corrected to use mp_glob_base_data_path for consistency with CDataLoader
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_base_data_path', 'Mql5Data')) 
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

        logger.info("Machine learning features configured.")

    # --- Helper methods ---
    def _convert_datetime(self, df: pd.DataFrame, column: str, fmt: str = None,
                          unit: str = None, conv_type: str = None, drop_cols: list = None):
        """Generalized datetime conversion (or dropping columns) helper."""
        # Special handling for drop_cols as it doesn't require a 'column' parameter
        if conv_type == 'g' and drop_cols:
            cols_to_drop = [col for col in drop_cols if col in df.columns]
            if cols_to_drop:
                df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')
                logger.info(f"Dropped columns: {cols_to_drop}")
            else:
                logger.debug(f"No columns from {drop_cols} found to drop.") # Debug for no drop
            return df
        
        # For other conversion types, ensure the column exists
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame for conversion type '{conv_type}'. Skipping conversion.")
            return df

        try:
            if conv_type == 'a': # Direct parsing of date/time string, often with format
                # For timestamps that are HH:MM:SS, to convert them to datetime.time objects:
                # These are Python time objects, not directly numerical, should be dropped later if not merged
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce').dt.time
            elif conv_type == 'e': # Direct parsing of date string like 'YYYYMMDD'
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce').dt.date
            elif conv_type == 'b': # Used for popping/replacing a column to be the primary datetime index
                # Ensure the column exists before pop and is convertible
                temp_col = df.pop(column) # Pop the column to avoid SettingWithCopyWarning
                df[column] = pd.to_datetime(temp_col, format=fmt, unit=unit, errors='coerce', utc=True)
            elif conv_type == 'c': # Specific format for milliseconds (not used much now)
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                df[column] = pd.to_datetime(df[column].dt.strftime('%d/%m/%y %H:%M:%S.%f'),
                                            format='%d/%m/%y %H:%M:%S.%f', errors='coerce', utc=True)
            elif conv_type == 'd': # Convert to Unix timestamp
                df[column] = df[column].map(pd.Timestamp.timestamp)
            elif conv_type == 'f': # Convert from Unix timestamp with a unit (e.g., 's' for seconds)
                df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
            
        except Exception as e:
            logger.error(f"Error converting column '{column}' with type '{conv_type}' (format: {fmt}, unit: {unit}): {e}")
        return df

    def _merge_datetime(self, df: pd.DataFrame, col_date: str, col_time: str, merged_col: str):
        """Merge date and time columns (as date and time objects) into a single datetime column."""
        if not (col_date in df.columns and col_time in df.columns):
            logger.warning(f"Cannot merge: '{col_date}' or '{col_time}' not found in DataFrame. Skipping datetime merge.")
            return df
        
        try:
            # Ensure both columns are not NaN before combining
            combined_datetime_series = df.apply(
                lambda row: datetime.combine(row[col_date], row[col_time]) if pd.notna(row[col_date]) and pd.notna(row[col_time]) else pd.NaT,
                axis=1
            )
            df[merged_col] = pd.to_datetime(combined_datetime_series, errors='coerce', utc=True)
            
            # Drop original columns after successful merge
            df.drop(columns=[col_date, col_time], inplace=True, errors='ignore')
            df = self._reorder_columns(df, merged_col)
            logger.info(f"Successfully merged '{col_date}' and '{col_time}' into '{merged_col}'.")
        except Exception as e:
            logger.error(f"Error merging '{col_date}' and '{col_time}' into '{merged_col}': {e}")
        return df

    def _reorder_columns(self, df: pd.DataFrame, first_col: str):
        """Place a specific column as the first column in the DataFrame."""
        if first_col in df.columns:
            cols = [first_col] + [col for col in df.columns if col != first_col]
            logger.info(f"Column '{first_col}' moved to the start.")
            return df[cols]
        logger.warning(f"Column '{first_col}' not found for reordering to start. Skipping reorder.")
        return df

    # --- Data Wrangling Methods ---
    def run_wrangle_service(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Run the wrangling service on a DataFrame based on its name."""
        ldf = df.copy(deep=True) # Always work on a deep copy
        if ldf.empty:
            logger.warning("DataFrame is empty. Skipping wrangling.")
            return ldf

        config_key = df_name
        if config_key not in self.COLUMN_PARAMS:
            logger.warning(f"No configuration for DataFrame: {df_name}. Skipping wrangling.")
            return ldf

        # Dynamically set filter flags based on config_key from self.data_params
        suffix_map = {'df_api_ticks': '1', 'df_api_rates': '2', 'df_file_ticks': '3', 'df_file_rates': '4'}
        suffix = suffix_map.get(config_key, '')

        self.filter_int = self.data_params.get(f'df{suffix}_filter_int', False)
        self.filter_flt = self.data_params.get(f'df{suffix}_filter_flt', False)
        self.filter_obj = self.data_params.get(f'df{suffix}_filter_obj', False)
        self.filter_dtmi = self.data_params.get(f'df{suffix}_filter_dtmi', False)
        self.filter_dtmf = self.data_params.get(f'df{suffix}_filter_dtmf', False)
        self.mp_dropna = self.data_params.get(f'df{suffix}_mp_dropna', True)
        self.mp_merge = self.data_params.get(f'df{suffix}_mp_merge', True)
        self.mp_convert = self.data_params.get(f'df{suffix}_mp_convert', True)
        self.mp_drop = self.data_params.get(f'df{suffix}_mp_drop', False)

        logger.info(f"Wrangling {df_name} data with merge: {self.mp_merge} and convert: {self.mp_convert}")

        # Determine source_key for mapping lookups (e.g., 'rates1' for 'df_api_rates')
        source_key = 'ticks1' if 'api_ticks' in df_name else \
                     'rates1' if 'api_rates' in df_name else \
                     'ticks2' if 'file_ticks' in df_name else \
                     'rates2' if 'file_rates' in df_name else None
        
        if source_key is None:
            logger.error(f"Could not determine source_key for {df_name}. Skipping wrangling steps.")
            return ldf

        # --- Step 1: Rename columns early ---
        if source_key in self.from_to_column_maps:
            ldf.rename(columns=self.from_to_column_maps[source_key], inplace=True)
            logger.info(f"DW: 1.1 Renamed columns based on mapping for {df_name}.")
        
        # --- Step 2: Convert relevant date/time columns to appropriate types ---
        # This is for columns that will be used in merging or are primary datetimes.
        # Ensure column exists *after* renaming.
        if source_key in self.date_columns:
            col, fmt, unit, conv_type = self.date_columns[source_key]
            if col and col in ldf.columns: # Check if column exists after rename
                ldf = self._convert_datetime(ldf, col, fmt, unit, conv_type)
                logger.info(f"DW: 1.2 Converted Date column '{col}' for {df_name}.")
            else:
                logger.warning(f"Date column '{col}' not found after rename for {df_name}. Skipping date conversion.")

        if source_key in self.time_columns:
            col, fmt, unit, conv_type = self.time_columns[source_key]
            if col and col in ldf.columns: # Check if column exists after rename
                ldf = self._convert_datetime(ldf, col, fmt, unit, conv_type)
                logger.info(f"DW: 1.3 Converted Time column '{col}' for {df_name}.")
            else:
                logger.warning(f"Time column '{col}' not found after rename for {df_name}. Skipping time conversion.")

        # --- Step 3: Merge date and time columns (only if merge is specified and columns exist) ---
        merge_config = self.merge_columns.get(source_key)
        if merge_config and merge_config[0] is not None and self.mp_merge: # Check if merge is configured and enabled
            col_date_orig, col_time_orig, merged_col = merge_config
            # Only attempt merge if both components exist AFTER earlier renames/conversions
            if col_date_orig in ldf.columns and col_time_orig in ldf.columns:
                ldf = self._merge_datetime(ldf, col_date_orig, col_time_orig, merged_col)
                logger.info(f"DW: 1.4 Merged '{col_date_orig}' and '{col_time_orig}' into '{merged_col}' for {df_name}.")
            else:
                logger.warning(f"Cannot merge: '{col_date_orig}' or '{col_time_orig}' missing for {df_name}. Skipping merge.")
        elif merge_config and merge_config[0] is None:
            logger.info(f"DW: 1.4 Merge not configured for '{df_name}' (source key: {source_key}). Skipping merge step.")
        else:
            logger.info(f"DW: 1.4 Merge disabled via config (mp_merge={self.mp_merge}) for '{df_name}'. Skipping merge step.")


        # --- Step 4: Final conversion of the primary datetime column (if applicable) ---
        # This handles cases like R1_Date which is already the primary datetime column after initial 'time' rename and conversion.
        conv_config = self.conv_columns.get(source_key)
        if conv_config and conv_config[0] is not None and self.mp_convert: # Check if conversion is configured and enabled
            col, fmt, unit, conv_type = conv_config
            if col in ldf.columns: # Ensure the column exists at this stage
                ldf = self._convert_datetime(ldf, col, fmt, unit, conv_type)
                logger.info(f"DW: 1.5 Converted final primary datetime column '{col}' for {df_name}.")
            else:
                logger.warning(f"Primary conversion column '{col}' not found for {df_name}. Skipping final datetime conversion.")
        elif conv_config and conv_config[0] is None:
            logger.info(f"DW: 1.5 Final datetime conversion not configured for '{df_name}'. Skipping.")
        else:
            logger.info(f"DW: 1.5 Final datetime conversion disabled via config (mp_convert={self.mp_convert}) for '{df_name}'. Skipping.")

        # --- Step 5: Drop unnecessary columns ---
        drop_config = self.drop_columns.get(source_key)
        if drop_config and self.mp_drop: # Check if drop is configured and enabled
            _, _, _, conv_type_drop, drop_cols = drop_config
            ldf = self._convert_datetime(ldf, None, None, None, conv_type_drop, drop_cols) # Call with None for column
            logger.info(f"DW: 1.6 Dropped unnecessary columns for {df_name} if enabled.")
        else:
            logger.info(f"DW: 1.6 Column dropping disabled via config (mp_drop={self.mp_drop}) for '{df_name}'. Skipping.")

        # --- Step 6: Apply type filtering conversions (numerical, object) ---
        for dtype, flag in [('int64', self.filter_int), ('float64', self.filter_flt)]:
            if flag:
                for col in ldf.select_dtypes(include=[dtype]).columns:
                    ldf[col] = pd.to_numeric(ldf[col], errors='coerce')
                    logger.info(f"DW: 1.7 Converted {dtype} columns to numeric if enabled for {df_name}.")
        if self.filter_obj:
            for col in ldf.select_dtypes(include=['object']).columns:
                ldf[col] = pd.to_datetime(ldf[col], errors='coerce') # Attempt to convert objects to datetime
                logger.info(f"DW: 1.8 Converted object columns to datetime if enabled for {df_name}.")
        if self.filter_dtmi:
            for col in ldf.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                ldf[col] = pd.to_numeric(ldf[col].view('int64')) # Convert datetime to int (nanoseconds since epoch)
                logger.info(f"DW: 1.9 Converted datetime columns to int64 if enabled for {df_name}.")
        if self.filter_dtmf:
            for col in ldf.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                ldf[col] = pd.to_numeric(ldf[col].view('float64')) # Convert datetime to float
                logger.info(f"DW: 1.10 Converted datetime columns to float64 if enabled for {df_name}.")
        
        # Apply fillna AFTER all conversions that might introduce NaNs
        if self.mp_dropna:
            numeric_cols = ldf.select_dtypes(include=['number']).columns
            ldf[numeric_cols] = ldf[numeric_cols].fillna(0) 
            logger.info(f"DW: 1.11 Filled NaN values in numeric columns with 0 for {df_name}.")
            
        # Reorder columns (put primary datetime first if merged/converted)
        final_first_col = self.first_columns.get(df_name)
        if final_first_col and final_first_col in ldf.columns:
            ldf = self._reorder_columns(ldf, final_first_col)
        else:
            logger.warning(f"Final primary column '{final_first_col}' not found for reordering. Skipping final reorder for {df_name}.")

        # --- Final cleanup: Ensure only numeric columns (and the primary datetime index) remain. ---
        # This catches any non-numeric columns that might have slipped through or were intentionally
        # kept for intermediate steps but are not intended as final features for ML.
        cols_to_drop_final_cleanup = []
        for col in ldf.columns:
            # If the column is an object type and not the designated final datetime column
            if ldf[col].dtype == 'object' and (final_first_col is None or col != final_first_col):
                # Attempt to convert to numeric, coercing errors.
                # If it still contains non-numeric data (e.g., all values become NaN), drop it.
                temp_series = pd.to_numeric(ldf[col], errors='coerce')
                if temp_series.isnull().all() and not ldf[col].isnull().all(): # If it became all NaN, and wasn't already all NaN
                    cols_to_drop_final_cleanup.append(col)
                else:
                    # If some values were convertible, or it was already all NaN, keep it (but it might contain NaNs now)
                    ldf[col] = temp_series
            # If it's a datetime type and not the designated final datetime column, and not already converted to numeric
            elif pd.api.types.is_datetime64_any_dtype(ldf[col]) and (final_first_col is None or col != final_first_col):
                # Convert to Unix timestamp (numerical representation)
                try:
                    ldf[col] = ldf[col].astype(np.int64) // 10**9 # Convert to seconds
                    logger.info(f"DW: Final cleanup - Converted datetime column '{col}' to Unix timestamp (seconds).")
                except Exception as e:
                    logger.warning(f"DW: Final cleanup - Could not convert datetime column '{col}' to Unix timestamp: {e}. Dropping.")
                    cols_to_drop_final_cleanup.append(col)

        if cols_to_drop_final_cleanup:
            logger.warning(f"DW: Final cleanup - Dropping non-numeric/unconvertible columns: {cols_to_drop_final_cleanup} from {df_name}")
            ldf.drop(columns=cols_to_drop_final_cleanup, inplace=True, errors='ignore')

        return ldf

    def run_average_columns(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Compute moving average, log returns, and optionally log stationarity."""
        ldf = df.copy(deep=True)  # Always work on a copy
        try:
            config = self.COLUMN_PARAMS.get(df_name, {})
            col_in = config.get("column_in")
            
            if not col_in or col_in not in ldf.columns:
                logger.error(f"Column '{col_in}' not found in DataFrame '{df_name}' for averaging. Skipping run_average_columns.")
                return ldf

            # Ensure data type is numeric for calculations
            ldf[col_in] = pd.to_numeric(ldf[col_in], errors='coerce')
            if ldf[col_in].isnull().any():
                logger.warning(f"NaN values introduced in '{col_in}' after numeric conversion for '{df_name}'. Filling with 0 for calculations.")
                ldf[col_in].fillna(0, inplace=True)


            if self.ma_window and config.get("ma_col"):
                ldf.loc[:, config["ma_col"]] = ldf[col_in].rolling(window=self.ma_window, min_periods=1).mean().bfill()
                logger.info(f"Moving average calculated in column '{config['ma_col']}' for '{df_name}'.")

            if self.shift_in and config.get("returns_col"):
                ldf.loc[:, col_in] = ldf[col_in].ffill() # Forward fill any NaNs before returns calculation
                # Handle non-positive values for log returns
                if (ldf[col_in] <= 0).any():
                    logger.warning(f"Non-positive values found in '{col_in}' for '{df_name}'. Cannot compute true log returns. Using percentage change instead.")
                    ldf.loc[:, config["returns_col"]] = ldf[col_in].pct_change(periods=self.shift_in).fillna(0)
                else:
                    ldf.loc[:, config["returns_col"]] = np.log(ldf[col_in] / ldf[col_in].shift(self.shift_in)).dropna()
                logger.info(f"Log returns/Percentage change computed in column '{config['returns_col']}' for '{df_name}'.")

            if self.log_stationary and config.get("ma_col") in ldf.columns:
                if (ldf[config["ma_col"]] <= 0).any():
                    logger.warning(f"Non-positive values found in '{config['ma_col']}' for '{df_name}'. Skipping log stationary transformation.")
                else:
                    ldf.loc[:, config["ma_col"]] = np.log(ldf[config["ma_col"]]).diff().fillna(0)
                    logger.info(f"Log stationary transformation applied on '{config['ma_col']}' for '{df_name}'.")

            if self.remove_zeros and config.get("returns_col") in ldf.columns:
                initial_rows = len(ldf)
                ldf = ldf[ldf[config["returns_col"]] != 0].copy() 
                logger.info(f"Removed {initial_rows - len(ldf)} rows with zero returns for '{df_name}'.")

            return ldf

        except Exception as e:
            logger.error(f"Error in run_average_columns for '{df_name}': {e}. Returning original DataFrame state.")
            return df

    def add_line_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a row number column if enabled."""
        ldf = df.copy(deep=True)
        if self.rownumber and not ldf.empty:
            ldf['rownumber'] = range(1, len(ldf) + 1)
            logger.info("Row numbers added.")
        elif self.rownumber and ldf.empty:
            logger.warning("DataFrame is empty, cannot add row numbers.")
        return ldf

    def move_col_to_end(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Move a specified column to the end."""
        ldf = df.copy(deep=True)
        if col_name is None or col_name not in ldf.columns:
            logger.warning(f"Column '{col_name}' not found for moving to end. Returning current DataFrame.")
            return ldf
        cols = [col for col in ldf.columns if col != col_name] + [col_name]
        logger.info(f"Column '{col_name}' moved to end.")
        return ldf[cols]

    def move_col_to_start(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Move a specified column to the start."""
        ldf = df.copy(deep=True)
        if col_name not in ldf.columns:
            logger.warning(f"Column '{col_name}' not found for moving to start. Returning current DataFrame.")
            return ldf
        cols = [col_name] + [col for col in ldf.columns if col != col_name]
        logger.info(f"Column '{col_name}' moved to start.")
        return ldf[cols]

    def create_index_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the first column as index after cleaning."""
        ldf = df.copy(deep=True)
        if ldf.empty:
            logger.warning("DataFrame is empty, cannot create index column.")
            return ldf
            
        first_col = ldf.columns[0]
        try:
            # Convert the first column to datetime objects
            ldf[first_col] = pd.to_datetime(ldf[first_col], errors='coerce')
            # Drop rows where the datetime conversion resulted in NaT (Not a Time)
            ldf.dropna(subset=[first_col], inplace=True)

            if ldf.empty:
                logger.warning(f"DataFrame became empty after dropping rows with invalid datetimes in '{first_col}'.")
                return ldf

            # Set the datetime column as the DataFrame index
            ldf.set_index(first_col, inplace=True)
            ldf.sort_index(inplace=True)  # Optional: sort by the index if needed
            logger.info(f"Index column '{first_col}' created and set.")
        except Exception as e:
            logger.error(f"Error creating index column from '{first_col}': {e}. Returning DataFrame without index.")
        return ldf


    def establish_common_feat_col(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Establish a common feature column for tick (bid-ask average) or OHLC (close)."""
        ldf = df.copy(deep=True)  # Always work on a copy
        config = self.COLUMN_PARAMS.get(df_name, {})

        if df_name in ["df_api_ticks", "df_file_ticks"]:
            bid_col, ask_col, out_col = config.get("bid_column"), config.get("ask_column"), config.get("column_out1")
            
            if not (bid_col and ask_col and out_col and bid_col in ldf.columns and ask_col in ldf.columns):
                logger.error(f"Missing or invalid columns for tick data processing in '{df_name}'. Bid: '{bid_col}', Ask: '{ask_col}', Out: '{out_col}'. Skipping common feature column creation.")
                return ldf
            
            # Ensure columns are numeric before calculation
            ldf[bid_col] = pd.to_numeric(ldf[bid_col], errors='coerce').fillna(0)
            ldf[ask_col] = pd.to_numeric(ldf[ask_col], errors='coerce').fillna(0)

            ldf.loc[:, out_col] = (ldf[bid_col] + ldf[ask_col]) / 2  # Safe assignment
            if self.run_avg and config.get("hl_avg_col"): # Check if hl_avg_col is configured
                ldf.loc[:, config["hl_avg_col"]] = ldf[out_col]  # Safe assignment
            logger.info(f"Bid-ask average computed for tick data in '{df_name}'. New column: '{out_col}'.")

        elif df_name in ["df_api_rates", "df_file_rates"]:
            close_col = config.get("close_column")
            out_col = config.get("column_out1") # This is 'Close' based on self.feature4 default

            if not (close_col and out_col and close_col in ldf.columns):
                logger.error(f"Missing or invalid 'close_column' ('{close_col}') or 'column_out1' ('{out_col}') for OHLC data processing in '{df_name}'. Skipping common feature column creation.")
                return ldf
            
            # Ensure close column is numeric
            ldf[close_col] = pd.to_numeric(ldf[close_col], errors='coerce').fillna(0)

            # Assign the 'close_col' content to the 'column_out1' (which is 'Close')
            ldf.loc[:, out_col] = ldf[close_col] 
            logger.info(f"Common feature column established for OHLC data in '{df_name}'. Column '{close_col}' mapped to '{out_col}'.")
        else:
            logger.warning(f"Unknown DataFrame type '{df_name}' for establish_common_feat_col. No common feature column created.")

        return ldf


    def establish_common_feat_col_scaled(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Establish a scaled version of the common feature column."""
        ldf = df.copy(deep=True)  # Always work on a copy
        config = self.COLUMN_PARAMS.get(df_name, {})
        out_col, out_scaled_col = config.get("column_out1"), config.get("column_out2") # These are 'Close' and 'Close_Scaled' or similar
        
        if not (out_col and out_scaled_col and out_col in ldf.columns):
            logger.warning(f"Missing or invalid output column definitions ('{out_col}', '{out_scaled_col}') or '{out_col}' not in DataFrame '{df_name}' for scaled feature. Skipping scaled column creation.")
            return ldf

        # Ensure the source column is numeric before calculating percentage change
        ldf[out_col] = pd.to_numeric(ldf[out_col], errors='coerce').fillna(0)

        # Calculate percentage change for scaling. Handle potential division by zero.
        # Add a small epsilon to avoid division by zero if values can be zero.
        epsilon = 1e-9 # Define epsilon locally if not from config
        ldf.loc[:, out_scaled_col] = ldf[out_col].pct_change().fillna(0)
        
        logger.info(f"Scaled feature column '{out_scaled_col}' created for '{df_name}'.")
        return ldf

    def process_data(self) -> pd.DataFrame:
        """
        Main method to run the complete data processing workflow on the internally
        stored DataFrame. It dynamically determines the DataFrame name.
        """
        if self.df.empty:
            logger.error("No data available in CDataProcess for processing. Returning empty DataFrame.")
            return pd.DataFrame()

        # Dynamically determine df_name based on presence of specific columns
        # This assumes a certain structure from CDataLoader's output.
        inferred_df_name = None

        # Enhanced inference: Check for renamed columns first, as wrangling happens
        # It's better to rely on `mp_app_cfg_usedata` if it's consistently set.
        if self.app_params.get('mp_app_cfg_usedata'):
            inferred_df_name = self.app_params.get('mp_app_cfg_usedata')
            logger.info(f"Inferred df_name from 'mp_app_cfg_usedata': {inferred_df_name}")
        
        # Fallback heuristic if mp_app_cfg_usedata is not set or reliable
        if inferred_df_name is None:
            if 'R1_Open' in self.df.columns and 'R1_Close' in self.df.columns:
                # If API rates data has already been renamed from 'time' to 'R1_Date' etc.
                # Or if original API data came with these names
                inferred_df_name = "df_api_rates"
                logger.info(f"Inferred df_name heuristically: {inferred_df_name} (based on R1_Open/Close)")
            elif 'T1_Bid_Price' in self.df.columns and 'T1_Ask_Price' in self.df.columns:
                inferred_df_name = "df_api_ticks"
                logger.info(f"Inferred df_name heuristically: {inferred_df_name} (based on T1_Bid_Price/Ask_Price)")
            elif 'R2_Open' in self.df.columns and 'R2_Close' in self.df.columns:
                inferred_df_name = "df_file_rates"
                logger.info(f"Inferred df_name heuristically: {inferred_df_name} (based on R2_Open/Close)")
            elif 'T2_Bid_Price' in self.df.columns and 'T2_Ask_Price' in self.df.columns:
                inferred_df_name = "df_file_ticks"
                logger.info(f"Inferred df_name heuristically: {inferred_df_name} (based on T2_Bid_Price/Ask_Price)")
            else:
                logger.error("Could not reliably infer DataFrame type. Processing might fail.")
                return pd.DataFrame() 


        logger.info(f"Starting data processing for identified DataFrame type: {inferred_df_name} with initial shape {self.df.shape}")

        processed_df = self.run_dataprocess_services(df=self.df, df_name=inferred_df_name)
        return processed_df


    # --- Service Workflow ---
    def run_dataprocess_services(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Run the complete data processing workflow on a given DataFrame."""
        ldf = df.copy(deep=True) # Ensure we're working on a copy passed into this method
        if ldf.empty:
            logger.warning(f"DataFrame '{df_name}' is empty. Skipping data processing services.")
            return ldf

        logger.info(f"Starting data processing for {df_name} with shape {ldf.shape}")

        # DP:1.1 Wrangling Data
        logger.info(f"DP:1.1 Wrangling {df_name} data...")
        ldf = self.run_wrangle_service(df=ldf, df_name=df_name)
        if ldf.empty:
            logger.error(f"DataFrame '{df_name}' became empty after wrangling. Cannot proceed with further processing.")
            return ldf

        # DP:1.2 Averaging Columns (e.g., MA, Returns)
        logger.info(f"DP:1.2 Averaging columns for {df_name}...")
        ldf = self.run_average_columns(ldf, df_name)
        if ldf.empty:
            logger.error(f"DataFrame '{df_name}' became empty after averaging columns. Cannot proceed.")
            return ldf

        # DP:1.3 Establishing Common Feature Column (e.g., Close, BidAskAvg)
        logger.info(f"DP:1.3 Establishing common feature column for {df_name}...")
        ldf = self.establish_common_feat_col(ldf, df_name)
        if ldf.empty:
            logger.error(f"DataFrame '{df_name}' became empty after establishing common feature column. Cannot proceed.")
            return ldf

        # DP:1.4 Establishing Scaled Common Feature Column
        logger.info(f"DP:1.4 Establishing scaled common feature column for {df_name}...")
        ldf = self.establish_common_feat_col_scaled(ldf, df_name)
        if ldf.empty:
            logger.error(f"DataFrame '{df_name}' became empty after establishing scaled common feature column. Cannot proceed.")
            return ldf

        # DP:1.5 Move primary datetime column to start
        first_col_to_move = self.first_columns.get(df_name)
        if first_col_to_move and first_col_to_move in ldf.columns:
            logger.info(f"DP:1.5 Moving primary datetime column to start: {first_col_to_move}")
            ldf = self.move_col_to_start(ldf, first_col_to_move)
        else:
            logger.warning(f"No specific first column '{first_col_to_move}' found for {df_name} to move to start. Skipping.")

        # DP:1.6 Move specified last columns to end (if self.last_col is True)
        last_col_config = self.last_columns.get(df_name)
        if last_col_config and self.last_col: 
            last_col_name = last_col_config[0] # The column name to move
            if last_col_name in ldf.columns: # Check if the target column exists
                logger.info(f"DP:1.6 Moving specified column to end: {last_col_name}")
                ldf = self.move_col_to_end(ldf, last_col_name)
            else:
                logger.warning(f"Last column '{last_col_name}' not found for {df_name}. Skipping move_col_to_end.")
        else:
            logger.info(f"DP:1.6 Moving last column disabled via config (self.last_col={self.last_col}) or no config for '{df_name}'. Skipping move_col_to_end.")

        # DP:1.7 Add line Numbers
        logger.info(f"DP:1.7 Add line Numbers {df_name}...")
        ldf = self.add_line_numbers(ldf)
        
        # DP:1.8 Create Index for DataFrame
        logger.info(f"DP:1.8 Create Index for {df_name}...")
        ldf = self.create_index_column(ldf)

        logger.info(f"DP:1.9 Data processing completed for {df_name} with shape {ldf.shape}")
        return ldf


def main():
    # This main function is for testing tsMqlDataProcess.py in isolation.
    logger.info("Running tsMqlDataProcess.py in standalone mode for testing.")
    
    # Example dummy DataFrame for testing df_api_rates scenario
    # Mimicking MT5 API output after fetching, where 'time' is a timestamp
    data_api_rates = {
        'time': [1672531200, 1672531260, 1672617600, 1672617660, 1672704000], # Unix timestamps
        'open': [1.0, 1.0001, 1.0002, 1.0003, 1.0004],
        'high': [1.001, 1.0011, 1.0012, 1.0013, 1.0014],
        'low': [0.999, 0.9991, 0.9992, 0.9993, 0.9994],
        'close': [1.0005, 1.0006, 1.0007, 1.0008, 1.0009],
        'tick_volume': [100, 110, 120, 130, 140],
        'spread': [5, 5, 5, 5, 5],
        'real_volume': [1000, 1100, 1200, 1300, 1400]
    }
    test_df_api_rates = pd.DataFrame(data_api_rates)

    # Example dummy DataFrame for testing df_file_rates scenario
    data_file_rates = {
        'Date': ['20230101', '20230101', '20230102', '20230102', '20230103'],
        'Timestamp': ['00:00:00', '00:00:01', '00:00:00', '00:00:01', '00:00:00'],
        'Open': [1.0, 1.0001, 1.0002, 1.0003, 1.0004],
        'High': [1.001, 1.0011, 1.0012, 1.0013, 1.0014],
        'Low': [0.999, 0.9991, 0.9992, 0.9993, 0.9994],
        'Close': [1.0005, 1.0006, 1.0007, 1.0008, 1.0009],
        'Volume': [100, 110, 120, 130, 140],
        'tick_volume': [100, 110, 120, 130, 140],
        'spread': [5, 5, 5, 5, 5],
        'real_volume': [1000, 1100, 1200, 1300, 1400]
    }
    test_df_file_rates = pd.DataFrame(data_file_rates)

    # Test df_api_rates
    logger.info("\n--- Testing df_api_rates ---")
    data_process_api_rates = CDataProcess(test_df_api_rates)
    # Simulate setting mp_app_cfg_usedata for accurate inference in process_data
    data_process_api_rates.app_params['mp_app_cfg_usedata'] = 'df_api_rates'
    processed_api_rates = data_process_api_rates.process_data()
    if not processed_api_rates.empty:
        logger.info("\nProcessed df_api_rates Head:")
        logger.info(tabulate(processed_api_rates.head(), headers='keys', tablefmt='psql'))
        logger.info(f"Processed df_api_rates shape: {processed_api_rates.shape}")
    else:
        logger.warning("Processed df_api_rates is empty. Check logs for errors.")

    # Test df_file_rates
    logger.info("\n--- Testing df_file_rates ---")
    data_process_file_rates = CDataProcess(test_df_file_rates)
    # Simulate setting mp_app_cfg_usedata for accurate inference in process_data
    data_process_file_rates.app_params['mp_app_cfg_usedata'] = 'df_file_rates'
    processed_file_rates = data_process_file_rates.process_data()
    if not processed_file_rates.empty:
        logger.info("\nProcessed df_file_rates Head:")
        logger.info(tabulate(processed_file_rates.head(), headers='keys', tablefmt='psql'))
        logger.info(f"Processed df_file_rates shape: {processed_file_rates.shape}")
    else:
        logger.warning("Processed df_file_rates is empty. Check logs for errors.")


if __name__ == '__main__':
    main()
