"""
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlDataProcess/tsMqlDataProcess.py
Description: Simplified, optimized module for loading data, processing DataFrames, and interfacing with MetaTrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 2.0 (Refactored, simplified and optimized)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from tabulate import tabulate

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlOverrides import CMqlOverrides

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")


class CDataProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)
        self.lp_utc_from = kwargs.get('lp_utc_from', datetime.utcnow())
        self.lp_utc_to = kwargs.get('lp_utc_to', datetime.utcnow())
        self.mp_unit = kwargs.get('UNIT', {})

        # Initialize MetaTrader and environment parameters
        self._initialize_mql()
        self._set_envmgr_params(kwargs)
        self._set_global_parameters(kwargs)

        # Set primary symbol and timeframe with defaults.
        # This must be set before _set_ml_features because that method uses lp_app_primary_symbol.
        self.lp_app_primary_symbol = kwargs.get(
            'lp_app_primary_symbol',
            self.params.get('app', {}).get('mp_app_primary_symbol', 'EURUSD')
        )
        self.lp_timeframe = kwargs.get(
            'lp_timeframe',
            self.params.get('data', {}).get('mp_data_timeframe', 'H4')
        )
        logger.info(f"Primary symbol: {self.lp_app_primary_symbol}, Timeframe: {self.lp_timeframe}")

        # Set machine learning features, now that primary symbol is defined
        self._set_ml_features(kwargs)

        # Mapping definitions for column conversions and reordering
        self.from_to_column_maps = {
            'ticks1': {'time': 'T1_Date', 'bid': 'T1_Bid_Price', 'ask': 'T1_Ask_Price',
                       'last': 'T1_Last Price', 'volume': 'T1_Volume', 'time_msc': 'T1_Time_Msc',
                       'flags': 'T1_Flags', 'volume_real': 'T1_Real_Volume'},
            'rates1': {'time': 'R1_Date', 'open': 'R1_Open', 'high': 'R1_High', 'low': 'R1_Low',
                       'close': 'R1_Close', 'tick_volume': 'R1_Tick_Volume', 'spread': 'R1_spread',
                       'real_volume': 'R1_Real_Volume'},
            'ticks2': {'mDatetime': 'T2_mDatetime', 'Date': 'T2_Date', 'Timestamp': 'T2_Timestamp',
                       'Bid Price': 'T2_Bid_Price', 'Ask Price': 'T2_Ask_Price', 'Last Price': 'T2_Last_Price',
                       'Volume': 'T2_Volume'},
            'rates2': {'mDatetime': 'R2_mDatetime', 'Date': 'R2_Date', 'Timestamp': 'R2_Timestamp',
                       'Open': 'R2_Open', 'High': 'R2_High', 'Low': 'R2_Low', 'Close': 'R2_Close',
                       'tick_volume': 'R2_Tick Volume', 'Volume': 'R2_Volume', 'vol2': 'R2_Vol1', 'vol3': 'R2_Vol3'}
        }

        self.date_columns = {
            'ticks1': ('time', '%Y%m%d', 's', 'f'),
            'rates1': ('time', '%Y%m%d', 's', 'f'),
            'ticks2': ('Date', '%Y%m%d', 's', 'e'),
            'rates2': ('Date', '%Y%m%d', 's', 'e'),
        }

        self.time_columns = {
            'ticks1': ('time_msc', '%Y%m%d %H:%M:%S', 'ms', 'a'),
            'ticks2': ('Timestamp', '%H:%M:%S', 'ms', 'a'),
            'rates2': ('Timestamp', '%H:%M:%S', 's', 'a'),
        }

        self.conv_columns = {
            'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', 's', 'b'),
            'rates1': ('R1_Date', '%Y%m%d %H:%M:%S.%f', 's', 'b'),
            'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', 's', 'b'),
            'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', 's', 'b'),
        }

        self.drop_columns = {
            'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g',
                       ['T1_Time_Msc', 'T1_Flags', 'T1_Last Price', 'T1_Real_Volume', 'T1_Volume']),
            'rates1': ('R1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g',
                       ['R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume']),
            'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g',
                       ['T2_Timestamp', 'T2_Volume', 'T2_Last_Price']),
            'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g',
                       ['R2_Timestamp', 'R2_Volume', 'R2_Vol1'])
        }

        self.merge_columns = {
            'ticks1': ('T1_Date', 'T1_Timestamp', 'T1_mDatetime', '%Y%m%d %H:%M:%S', '%H:%M:%S'),
            'rates1': ('R1_Date', 'R1_Timestamp', 'R1_mDatetime', '%Y%m%d %H:%M:%S', '%H:%M:%S'),
            'ticks2': ('T2_Date', 'T2_Timestamp', 'T2_mDatetime', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M:%S'),
            'rates2': ('R2_Date', 'R2_Timestamp', 'R2_mDatetime', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M:%S'),
        }

        # Updated mapping with keys that match the expected DataFrame names
        self.first_columns = {
            'df_api_ticks': 'T1_Date',
            'df_api_rates': 'R1_Date',
            'df_file_ticks': 'T2_mDatetime',
            'df_file_rates': 'R2_mDatetime',
        }

        self.last_columns = {
            'df_api_ticks': ('Close', 'Close_scaled'),
            'df_api_rates': ('Close', 'Close_scaled'),
            'df_file_ticks': ('Close', 'Close_scaled'),
            'df_file_rates': ('Close', 'Close_scaled'),
        }

        # COLUMN_PARAMS holds all processing parameters for different DataFrame types.
        self.COLUMN_PARAMS = {
            "df_api_ticks": {
                'bid_column': 'T1_Bid_Price',
                'ask_column': 'T1_Ask_Price',
                'column_in': 'T1_Bid_Price',
                'column_out1': self.feature4,
                'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label,
                'df1_filter_int': self.params.get('data', {}).get('df1_filter_int', False),
                'df1_filter_flt': self.params.get('data', {}).get('df1_filter_flt', False),
                'df1_filter_obj': self.params.get('data', {}).get('df1_filter_obj', False),
                'df1_filter_dtmi': self.params.get('data', {}).get('df1_filter_dtmi', False),
                'df1_filter_dtmf': self.params.get('data', {}).get('df1_filter_dtmf', False),
                'df1_mp_dropna': self.params.get('data', {}).get('df1_mp_dropna', True),
                'df1_mp_merge': self.params.get('data', {}).get('df1_mp_merge', True),
                'df1_mp_convert': self.params.get('data', {}).get('df1_mp_convert', True),
                'df1_mp_drop': self.params.get('data', {}).get('df1_mp_drop', False)
            },
            "df_api_rates": {
                'bid_column': 'R1_Open',
                'ask_column': 'R1_Close',
                'column_in': 'R1_Open',
                'open_column': 'R1_Open',
                'high_column': 'R1_High',
                'low_column': 'R1_Low',
                'close_column': 'R1_Close',
                'column_out1': self.feature4,
                'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label,
                'df2_filter_int': self.params.get('data', {}).get('df2_filter_int', False),
                'df2_filter_flt': self.params.get('data', {}).get('df2_filter_flt', False),
                'df2_filter_obj': self.params.get('data', {}).get('df2_filter_obj', False),
                'df2_filter_dtmi': self.params.get('data', {}).get('df2_filter_dtmi', False),
                'df2_filter_dtmf': self.params.get('data', {}).get('df2_filter_dtmf', False),
                'df2_mp_dropna': self.params.get('data', {}).get('df2_mp_dropna', True),
                'df2_mp_merge': self.params.get('data', {}).get('df2_mp_merge', True),
                'df2_mp_convert': self.params.get('data', {}).get('df2_mp_convert', True),
                'df2_mp_drop': self.params.get('data', {}).get('df2_mp_drop', False)
            },
            "df_file_ticks": {
                'bid_column': 'T2_Bid_Price',
                'ask_column': 'T2_Ask_Price',
                'column_in': 'T2_Bid_Price',
                'column_out1': self.feature4,
                'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label,
                'df3_filter_int': self.params.get('data', {}).get('df3_filter_int', False),
                'df3_filter_flt': self.params.get('data', {}).get('df3_filter_flt', False),
                'df3_filter_obj': self.params.get('data', {}).get('df3_filter_obj', False),
                'df3_filter_dtmi': self.params.get('data', {}).get('df3_filter_dtmi', False),
                'df3_filter_dtmf': self.params.get('data', {}).get('df3_filter_dtmf', False),
                'df3_mp_dropna': self.params.get('data', {}).get('df3_mp_dropna', True),
                'df3_mp_merge': self.params.get('data', {}).get('df3_mp_merge', True),
                'df3_mp_convert': self.params.get('data', {}).get('df3_mp_convert', True),
                'df3_mp_drop': self.params.get('data', {}).get('df3_mp_drop', False)
            },
            "df_file_rates": {
                'bid_column': 'R2_Open',
                'ask_column': 'R2_Close',
                'column_in': 'R2_Open',
                'open_column': 'R2_Open',
                'high_column': 'R2_High',
                'low_column': 'R2_Low',
                'close_column': 'R2_Close',
                'column_out1': self.feature4,
                'column_out2': self.feature4_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label,
                'df4_filter_int': self.params.get('data', {}).get('df4_filter_int', False),
                'df4_filter_flt': self.params.get('data', {}).get('df4_filter_flt', False),
                'df4_filter_obj': self.params.get('data', {}).get('df4_filter_obj', False),
                'df4_filter_dtmi': self.params.get('data', {}).get('df4_filter_dtmi', False),
                'df4_filter_dtmf': self.params.get('data', {}).get('df4_filter_dtmf', False),
                'df4_mp_dropna': self.params.get('data', {}).get('df4_mp_dropna', True),
                'df4_mp_merge': self.params.get('data', {}).get('df4_mp_merge', True),
                'df4_mp_convert': self.params.get('data', {}).get('df4_mp_convert', True),
                'df4_mp_drop': self.params.get('data', {}).get('df4_mp_drop', False)
            }
        }

    def _initialize_mql(self):
        """Initialize MetaTrader5 if available."""
        self.os_platform = platform_checker.get_platform()
        self.loadmql = pchk.check_mql_state()
        logger.info(f"Running on: {self.os_platform}, loadmql state: {self.loadmql}")
        if self.loadmql:
            try:
                global mt5
                import MetaTrader5 as mt5
                if not mt5.initialize():
                    logger.error(f"Failed to initialize MetaTrader5. Error: {mt5.last_error()}")
            except ImportError as e:
                logger.error(f"Failed to import MetaTrader5: {e}")

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
        """Placeholder for global parameter settings."""
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
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_data_path', 'Mql5Data'))
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

        logger.info("Machine learning features configured.")

    # --- Helper methods ---
    def _convert_datetime(self, df: pd.DataFrame, column: str, fmt: str = None,
                          unit: str = None, conv_type: str = None, drop_cols: list = None):
        """Generalized datetime conversion (or dropping columns) helper."""
        try:
            if conv_type == 'a' or conv_type == 'e':
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
            elif conv_type == 'b':
                df[column] = pd.to_datetime(df.pop(column), format=fmt, errors='coerce')
            elif conv_type == 'c':
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                df[column] = pd.to_datetime(df[column].dt.strftime('%d/%m/%y %H:%M:%S.%f'),
                                            format='%d/%m/%y %H:%M:%S.%f', errors='coerce', utc=True)
            elif conv_type == 'd':
                df[column] = df[column].map(pd.Timestamp.timestamp)
            elif conv_type == 'f':
                df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
            elif conv_type == 'g' and drop_cols:
                cols_to_drop = [col for col in drop_cols if col in df.columns]
                if cols_to_drop:
                    df.drop(cols_to_drop, axis=1, inplace=True)
        except Exception as e:
            logger.error(f"Error converting column {column} with type {conv_type}: {e}")
        return df

    def _merge_datetime(self, df: pd.DataFrame, col_date: str, col_time: str, merged_col: str):
        """Merge date and time columns into a single datetime column."""
        try:
            if col_date in df.columns and col_time in df.columns:
                df[merged_col] = pd.to_datetime(
                    df[col_date].dt.strftime('%Y-%m-%d') + ' ' + df[col_time].dt.strftime('%H:%M:%S.%f'),
                    format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True
                )
                df.drop([col_date, col_time], axis=1, inplace=True)
                df = self._reorder_columns(df, merged_col)
        except Exception as e:
            logger.error(f"Error merging {col_date} and {col_time}: {e}")
        return df

    def _reorder_columns(self, df: pd.DataFrame, first_col: str):
        """Place a specific column as the first column in the DataFrame."""
        if first_col in df.columns:
            cols = [first_col] + [col for col in df.columns if col != first_col]
            return df[cols]
        return df

    # --- Data Wrangling Methods ---
    def run_wrangle_service(self, **kwargs) -> pd.DataFrame:
        """Run the wrangling service on a DataFrame based on its name."""
        self.df = kwargs.get('df', pd.DataFrame())
        self.df_name = kwargs.get('df_name')
        if self.df.empty:
            logger.warning("DataFrame is empty. Skipping wrangling.")
            return self.df

        # Map configuration based on DataFrame name
        config_key = self.df_name
        if config_key not in self.COLUMN_PARAMS:
            logger.warning(f"No configuration for DataFrame: {self.df_name}")
            return self.df

        config = self.COLUMN_PARAMS[config_key]
        # Set filtering and processing flags from kwargs or defaults
        self.filter_int = kwargs.get('filter_int', config.get(f"{config_key.split('_')[1]}_filter_int", False))
        self.filter_flt = kwargs.get('filter_flt', config.get(f"{config_key.split('_')[1]}_filter_flt", False))
        self.filter_obj = kwargs.get('filter_obj', config.get(f"{config_key.split('_')[1]}_filter_obj", False))
        self.filter_dtmi = kwargs.get('filter_dtmi', config.get(f"{config_key.split('_')[1]}_filter_dtmi", False))
        self.filter_dtmf = kwargs.get('filter_dtmf', config.get(f"{config_key.split('_')[1]}_filter_dtmf", False))
        self.mp_dropna = kwargs.get('mp_dropna', config.get(f"{config_key.split('_')[1]}_mp_dropna", True))
        self.mp_merge = kwargs.get('mp_merge', config.get(f"{config_key.split('_')[1]}_mp_merge", True))
        self.mp_convert = kwargs.get('mp_convert', config.get(f"{config_key.split('_')[1]}_mp_convert", True))
        self.mp_drop = kwargs.get('mp_drop', config.get(f"{config_key.split('_')[1]}_mp_drop", False))

        logger.info(f"Wrangling {self.df_name} data with merge: {self.mp_merge} and convert: {self.mp_convert}")

        # Process columns based on file source keys (use 'ticks1'/'rates1' for API and 'ticks2'/'rates2' for file)
        source_key = 'ticks1' if 'ticks' in self.df_name and 'api' in self.df_name else \
                     'rates1' if 'rates' in self.df_name and 'api' in self.df_name else \
                     'ticks2' if 'ticks' in self.df_name else 'rates2'
        
        # Apply datetime conversions if mapping are defined
        if source_key in self.date_columns:
            col, fmt, unit, conv_type = self.date_columns[source_key]
            self.df = self._convert_datetime(self.df, col, fmt, unit, conv_type)
            logger.info(f"DW: 1.1 Converted Date: {col} to datetime if found in mapping")
        if source_key in self.time_columns:
            col, fmt, unit, conv_type = self.time_columns[source_key]
            self.df = self._convert_datetime(self.df, col, fmt, unit, conv_type)
            logger.info(f"DW: 1.2 Converted Time: {col} to datetime if found in mapping")
        
        # Rename columns
        if source_key in self.from_to_column_maps:
            self.df.rename(columns=self.from_to_column_maps[source_key], inplace=True)
            logger.info(f"DW: 1.3 Renamed columns based on mapping")
        
        # Merge datetime columns if enabled
        if source_key in self.merge_columns and self.mp_merge:
            col_date, col_time, merged_col, _, _ = self.merge_columns[source_key]
            self.df = self._merge_datetime(self.df, col_date, col_time, merged_col)
            logger.info(f"DW: 1.4 Merged {col_date} and {col_time} into {merged_col} if enabled {self.mp_merge}")
        
        # Convert datetime in specified column if enabled
        if source_key in self.conv_columns and self.mp_convert:
            col, fmt, unit, conv_type = self.conv_columns[source_key]
            self.df = self._convert_datetime(self.df, col, fmt, unit, conv_type)
            logger.info(f"DW: 1.5 Converted specific datetime  column : {col} if enabled {self.mp_convert}")
        
        # Drop unnecessary columns if enabled
        if source_key in self.drop_columns and self.mp_drop:
            col, fmt, unit, conv_type, drop_cols = self.drop_columns[source_key]
            self.df = self._convert_datetime(self.df, col, fmt, unit, conv_type, drop_cols)
            logger.info(f"DW: 1.6 Dropped unnecessary columns if enabled {self.mp_drop}")
        
        # Apply type filtering conversions
        for dtype, flag in [('int64', self.filter_int), ('float64', self.filter_flt)]:
            if flag:
                for col in self.df.select_dtypes(include=[dtype]).columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    logger.info(f"DF: 1.1 Converted {dtype} columns to numeric if enabled {flag}")
        if self.filter_obj:
            for col in self.df.select_dtypes(include=['object']).columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                logger.info("DF: 1.2 Converted object columns to datetime if enabled")
        if self.filter_dtmi:
            for col in self.df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                self.df[col] = pd.to_numeric(self.df[col].view('int64'))
                logger.info("DF: 1.3 Converted datetime columns to int64 if enabled")
        if self.filter_dtmf:
            for col in self.df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                self.df[col] = pd.to_numeric(self.df[col].view('float64'))
                logger.info("DF: 1.4 Converted datetime columns to float64 if enabled")
        if self.mp_dropna:
            self.df.fillna(0, inplace=True)
            logger.info("DF: 1.5 Filled NaN values with 0")
            
        
        # Reorder columns if merged
        if source_key in self.merge_columns and self.mp_merge:
            _, _, merged_col, _, _ = self.merge_columns[source_key]
            self.df = self._reorder_columns(self.df, merged_col)
        
        return self.df

    def run_average_columns(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Compute moving average, log returns, future returns, and optionally log stationarity."""
        try:
            config = self.COLUMN_PARAMS.get(df_name, {})
            col_in = config.get("column_in")
            if col_in not in df.columns:
                logger.error(f"Column {col_in} not found in DataFrame.")
                return df

            # Moving average
            if self.ma_window and config.get("ma_col"):
                df[config["ma_col"]] = df[col_in].rolling(window=self.ma_window, min_periods=1).mean().fillna(method="bfill")
                logger.info(f"Moving average calculated in column {config['ma_col']}.")

            # Log returns (ensuring no non-positive values)
            if self.shift_in and config.get("returns_col"):
                df[col_in] = df[col_in].fillna(method='ffill')
                if (df[col_in] <= 0).any():
                    raise ValueError(f"Non-positive values found in {col_in}, cannot compute log returns.")
                df[config["returns_col"]] = np.log(df[col_in] / df[col_in].shift(self.shift_in)).dropna()
                logger.info(f"Log returns computed in column {config['returns_col']}.")

            # Log stationary transformation if enabled
            if self.log_stationary and config.get("ma_col") in df.columns:
                df[config["ma_col"]] = np.log(df[config["ma_col"]]).diff().fillna(0)
                logger.info(f"Log stationary transformation applied on {config['ma_col']}.")

            # Future returns (percentage change)
            if self.lookahead_periods and config.get("returns_col"):
                df[config["returns_col"]] = df[col_in].pct_change(periods=self.lookahead_periods).fillna(0)
                logger.info(f"Future returns computed in column {config['returns_col']}.")

            # Optionally remove rows with zero returns
            if self.remove_zeros and config.get("returns_col") in df.columns:
                df = df[df[config["returns_col"]] != 0]
                logger.info("Rows with zero returns removed.")

            return df
        except Exception as e:
            logger.error(f"Error in run_average_columns: {e}")
            return df

    def add_line_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a row number column if enabled."""
        if self.rownumber:
            df['rownumber'] = range(1, len(df) + 1)
            logger.info("Row numbers added.")
        return df

    def move_col_to_end(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Move a specified column to the end."""
        if col_name is None or col_name not in df.columns:
            logger.warning(f"Column {col_name} not found.")
            return df
        cols = [col for col in df.columns if col != col_name] + [col_name]
        logger.info(f"Column {col_name} moved to end.")
        return df[cols]

    def move_col_to_start(self, df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Move a specified column to the start."""
        if col_name not in df.columns:
            logger.warning(f"Column {col_name} not found.")
            return df
        cols = [col_name] + [col for col in df.columns if col != col_name]
        logger.info(f"Column {col_name} moved to start.")
        return df[cols]

    def create_index_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Set the first column as index after cleaning."""
        first_col = df.columns[0]
        # Convert the mDatetime' column to datetime objects if not already done
        df[first_col] = pd.to_datetime(df[first_col])
        # Set the datetime column as the DataFrame index
        df.set_index(first_col, inplace=True)
        df.sort_index(inplace=True)  # Optional: sort by the index if needed
        return df


    def establish_common_feat_col(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Establish a common feature column for tick (bid-ask average) or OHLC (close)."""
        config = self.COLUMN_PARAMS.get(df_name, {})
        if df_name in ["df_api_ticks", "df_file_ticks"]:
            bid, ask, out = config.get("bid_column"), config.get("ask_column"), config.get("column_out1")
            if not (bid and ask and out):
                raise ValueError(f"Missing definitions for {df_name}")
            df[out] = (df[bid] + df[ask]) / 2
            if self.run_avg and config.get("hl_avg_col"):
                df[config["hl_avg_col"]] = df[out]
            logger.info("Bid-ask average computed for tick data.")
        elif df_name in ["df_api_rates", "df_file_rates"]:
            close = config.get("close_column")
            if not close:
                raise ValueError("`close_column` must be provided for OHLC data.")
            df[config.get("column_out1")] = df[close]
            logger.info("Common feature column established for OHLC data.")
        return df

    def establish_common_feat_col_scaled(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Establish a scaled version of the common feature column."""
        config = self.COLUMN_PARAMS.get(df_name, {})
        out, out_scaled = config.get("column_out1"), config.get("column_out2")
        if not (out and out_scaled):
            raise ValueError(f"Missing output column definitions for {df_name}")
        if out in df.columns:
            df[out_scaled] = df[out].pct_change().fillna(0)
            logger.info(f"Scaled feature column {out_scaled} created.")
        return df

    # --- Service Workflow ---
    def run_dataprocess_services(self, **kwargs) -> pd.DataFrame:
        """Run the complete data processing workflow."""
        self.df = kwargs.get('df', pd.DataFrame())
        self.df_name = kwargs.get('df_name')
        logger.info(f"Starting data processing for {self.df_name} with shape {self.df.shape}")

        # Wrangle, average, establish features, reorder, and add index
        logger.info(f"DP:1.0 Running data processing for {self.df_name}...")
        logger.info(f"DP:1.1 Wrangling {self.df_name} data...")
        ldf = self.run_wrangle_service(df=self.df, df_name=self.df_name)
        logger.info(f"DP:1.2 Averaging columns for {self.df_name}...")
        ldf = self.run_average_columns(ldf, self.df_name)
        logger.info(f"DP:1.3 Establishing common feature column for {self.df_name}...")
        ldf = self.establish_common_feat_col(ldf, self.df_name)
        logger.info(f"DP:1.4 Moving columns to start {self.df_name} Column: {self.first_columns.get(self.df_name, ldf.columns[0])}")
        ldf = self.move_col_to_start(ldf, self.first_columns.get(self.df_name, ldf.columns[0]))
        logger.info(f"DP:1.5 Moving columns to End{self.df_name} Column: {self.last_columns.get(self.df_name, ())[0]}")
        ldf = self.move_col_to_end(ldf, self.last_columns.get(self.df_name, ())[0] if self.last_col else None)
        logger.info(f"DP:1.6 Add line Numbers {self.df_name}...")
        ldf = self.add_line_numbers(ldf)
        logger.info(f"DP:1.7 Create Index for {self.df_name}...")
        ldf = self.create_index_column(ldf)
        logger.info(ldf.index)
        logger.info(f"DP:1.8 Data processing completed for {self.df_name} with shape {ldf.shape}")
        return ldf


def main(logger):
    # Initialize data processing configuration with any required keyword arguments
    UNIT = {}  # Define or load UNIT as needed
    data_process = CDataProcess(mp_unit=UNIT)
    logger.info("Data processing configuration initialized.")
    # Further application logic can continue here...


if __name__ == '__main__':
    main(logger)
