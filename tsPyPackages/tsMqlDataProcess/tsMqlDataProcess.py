#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlDataProcess/tsMqlDataProcess.py
Description: Load and add files and data parameters, login to MetaTrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1 (Refactored)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import textwrap
from tabulate import tabulate

# Import platform dependencies
from tsMqlPlatform import run_platform, platform_checker, logger
from tsMqlEnvMgr import CMqlEnvMgr
from tsMqlUtilities import CUtilities

# Initialize platform checker
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform}, loadmql state: {loadmql}")


class CDataProcess:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        self.util_config = CUtilities()
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()
        
        # Extract parameter sections
        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)

        # Initialize run state parameters
        self._initialize_mql()
        self._set_global_parameters(kwargs)

        # Store column parameters for different DataFrame types
        self.COLUMN_PARAMS = {
            "df_api_ticks": {
                'bid_column': 'T1_Bid_Price',
                'ask_column': 'T1_Ask_Price',
                'column_in': 'T1_Bid_Price',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label
            },
            "df_api_rates": {
                'bid_column': 'R1_Open',
                'ask_column': 'R1_Close',
                'column_in': 'R1_Open',
                'open_column': 'R1_Open',
                'high_column': 'R1_High',
                'low_column': 'R1_Low',
                'close_column': 'R1_Close',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label
            }, 
            "df_file_ticks": {
                'bid_column': 'T2_Bid_Price',
                'ask_column': 'T2_Ask_Price',
                'column_in': 'T2_Bid_Price',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label
            },
            "df_file_rates": {
                'bid_column': 'R2_Open',
                'ask_column': 'R2_Close',
                'column_in': 'R2_Open',
                'open_column': 'R2_Open',
                'high_column': 'R2_High',
                'low_column': 'R2_Low',
                'close_column': 'R2_Close',
                'column_out1': self.mp_ml_input_keyfeat,
                'column_out2': self.mp_ml_input_keyfeat_scaled,
                'lookahead_periods': self.lookahead_periods,
                'ma_window': self.ma_window,
                'hl_avg_col': self.hl_avg_col,
                'ma_col': self.ma_col,
                'returns_col': self.returns_col,
                'shift_in': self.shift_in,
                'create_label': self.create_label
            }
        }

    def _initialize_mql(self):
        """Initialize MetaTrader5 module if available."""
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

    def _set_global_parameters(self, kwargs):
        """Set configuration parameters from environment or user input."""
        data_params = self.params.get('data', {})
        ml_params = self.params.get('ml', {})
        app_params = self.params.get('app', {})
        base_params = self.params.get('base', {})

        self.mp_data_filename1 = data_params.get('mp_data_filename1', 'default1.csv')
        self.mp_data_filename2 = data_params.get('mp_data_filename2', 'default2.csv')
        self.rownumber = ml_params.get('mp_rownumber', False)

        # Machine learning parameters
        self.mp_ml_input_keyfeat = ml_params.get('mp_ml_input_keyfeat', 'KeyFeature')
        self.mp_ml_input_keyfeat_scaled = ml_params.get('mp_ml_input_keyfeat_scaled', 'KeyFeature_Scaled')
        self.lookahead_periods = ml_params.get('mp_lookahead_periods', 1)
        self.ma_window = ml_params.get('mp_ma_window', 10)
        self.hl_avg_col = ml_params.get('mp_hl_avg_col', 'HL_Avg')
        self.ma_col = ml_params.get('mp_ma_col', 'MA')
        self.returns_col = ml_params.get('mp_returns_col', 'Returns')
        self.shift_in = ml_params.get('mp_shift_in', 1)
        self.create_label = ml_params.get('mp_create_label', False)
        self.create_label_scaled = ml_params.get('mp_create_label_scaled', False)

        # Run states
        self.run_avg = ml_params.get('mp_run_avg', False)
        self.run_avg_scaled = ml_params.get('mp_run_avg_scaled', False)
        self.log_stationary = ml_params.get('mp_log_stationary', False)
        self.remove_zeros = ml_params.get('mp_remove_zeros', False)
        self.last_col = ml_params.get('mp_last_col', False)

        # Application parameters
        self.lp_utc_from = kwargs.get('lp_utc_from', datetime.utcnow())
        self.lp_utc_to = kwargs.get('lp_utc_to', datetime.utcnow())
        self.lp_app_primary_symbol = kwargs.get('lp_app_primary_symbol', app_params.get('mp_app_primary_symbol', 'EURUSD'))
        self.lp_app_rows = kwargs.get('lp_app_rows', app_params.get('mp_app_rows', 1000))
        self.lp_timeframe = kwargs.get('lp_timeframe', app_params.get('mp_app_timeframe', 'H4'))
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', base_params.get('mp_glob_data_path', 'Mql5Data'))

        # Derived filenames
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

        # Filtering and conversion flags for data wrangling
        self.filter_int = kwargs.get('filter_int', False)
        self.filter_flt = kwargs.get('filter_flt', False)
        self.filter_obj = kwargs.get('filter_obj', False)
        self.filter_dtmi = kwargs.get('filter_dtmi', False)
        self.filter_dtmf = kwargs.get('filter_dtmf', False)
        self.mp_dropna = kwargs.get('mp_dropna', True)
        self.mp_convert = kwargs.get('mp_convert', True)
        self.mp_drop = kwargs.get('mp_drop', True)
        self.mp_merge = kwargs.get('mp_merge', True)
        self.mp_unit = kwargs.get('UNIT', {})


    def run_wrangle_service(self, **kwargs):
        """
        Run the data loader service for various DataFrame types.
        Returns:
            Tuple of processed DataFrames: (df_api_ticks, df_api_rates, df_file_ticks, df_file_rates)
        """
        self.df_api_ticks = kwargs.get('df_api_ticks', pd.DataFrame())
        self.df_api_rates = kwargs.get('df_api_rates', pd.DataFrame())
        self.df_file_ticks = kwargs.get('df_file_ticks', pd.DataFrame())
        self.df_file_rates = kwargs.get('df_file_rates', pd.DataFrame())

        # Wrangle each DataFrame if not empty
        if not self.df_api_ticks.empty and self.loadmql:
            self.df_api_ticks = self.wrangle_time(self.df_api_ticks, mp_filesrc="ticks1")
        if not self.df_api_rates.empty and self.loadmql:
            self.df_api_rates = self.wrangle_time(self.df_api_rates, mp_filesrc="rates1")
        if not self.df_file_ticks.empty:
            self.df_file_ticks = self.wrangle_time(self.df_file_ticks, mp_filesrc="ticks2")
        if not self.df_file_rates.empty:
            self.df_file_rates = self.wrangle_time(self.df_file_rates, mp_filesrc="rates2")

        return self.df_api_ticks, self.df_api_rates, self.df_file_ticks, self.df_file_rates

    def wrangle_time(self, df: pd.DataFrame, mp_filesrc: str) -> pd.DataFrame:
        """
        Wrangles time-related data in the DataFrame based on file source.
        """
        def rename_columns(df: pd.DataFrame, mapping: dict) -> None:
            valid_renames = {old: new for old, new in mapping.items() if old in df.columns}
            df.rename(columns=valid_renames, inplace=True)

        def merge_datetime(df: pd.DataFrame, col_date: str, col_time: str, merged_col: str, date_fmt: str, time_fmt: str) -> pd.DataFrame:
            if col_date in df.columns and col_time in df.columns:
                try:
                    logger.info(f"Merging columns {col_date} and {col_time} into {merged_col} for {mp_filesrc}")
                    df[merged_col] = pd.to_datetime(
                        df[col_date].dt.strftime('%Y-%m-%d') + ' ' + df[col_time].dt.strftime('%H:%M:%S.%f'),
                        format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
                    df.drop([col_date, col_time], axis=1, inplace=True)
                    # Reorder columns to place datetime first
                    datetime_col = merged_col if merged_col in df.columns else df.columns[0]
                    df = df[[datetime_col] + [col for col in df.columns if col != datetime_col]]
                except Exception as e:
                    logger.info(f"Error merging columns {col_date} and {col_time} for {mp_filesrc}: {e}")
            return df

        def resort_columns(df: pd.DataFrame, merged_col: str) -> pd.DataFrame:
            if merged_col in df.columns:
                df = df[[merged_col] + [col for col in df.columns if col != merged_col]]
                logger.info("Reordered columns to place datetime column first.")
            return df

        def convert_datetime(df: pd.DataFrame, column: str, fmt: str = None, unit: str = None, conv_type: str = None, drop_cols: list = None) -> None:
            try:
                if conv_type == 'a':
                    logger.info(f"Converting column {column} with type 'a' for {mp_filesrc} using format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%H:%M:%S.%f'), format='%H:%M:%S.%f', errors='coerce', utc=True)
                elif conv_type == 'b':
                    logger.info(f"Converting column {column} with type 'b' for {mp_filesrc} using format {fmt}")
                    df[column] = pd.to_datetime(df.pop(column), format=fmt, errors='coerce')
                elif conv_type == 'c':
                    logger.info(f"Converting column {column} with type 'c' for {mp_filesrc} using format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%d/%m/%y %H:%M:%S.%f'), format='%d/%m/%y %H:%M:%S.%f', errors='coerce', utc=True)
                elif conv_type == 'd':
                    logger.info(f"Converting column {column} with type 'd' for {mp_filesrc}")
                    df[column] = df[column].map(pd.Timestamp.timestamp)
                elif conv_type == 'e':
                    logger.info(f"Converting column {column} with type 'e' for {mp_filesrc} using format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                elif conv_type == 'f':
                    logger.info(f"Converting column {column} with type 'f' for {mp_filesrc} using unit {unit}")
                    df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
                elif conv_type == 'g':
                    logger.info(f"Dropping columns {drop_cols} for {mp_filesrc} if present")
                    if column in df.columns and drop_cols:
                        cols_to_drop = [col for col in drop_cols if col in df.columns]
                        if cols_to_drop:
                            df.drop(cols_to_drop, axis=1, inplace=True)
            except Exception as e:
                logger.info(f"Error converting column {column} for {mp_filesrc} with type {conv_type}: {e}")

        # Define mappings and configuration dictionaries
        mappings = {
            'ticks1': {
                'time': 'T1_Date',
                'bid': 'T1_Bid_Price',
                'ask': 'T1_Ask_Price',
                'last': 'T1_Last Price',
                'volume': 'T1_Volume',
                'time_msc': 'T1_Time_Msc',
                'flags': 'T1_Flags',
                'volume_real': 'T1_Real_Volume'
            },
            'rates1': {
                'time': 'R1_Date',
                'open': 'R1_Open',
                'high': 'R1_High',
                'low': 'R1_Low',
                'close': 'R1_Close',
                'tick_volume': 'R1_Tick_Volume',
                'spread': 'R1_spread',
                'real_volume': 'R1_Real_Volume'
            },
            'ticks2': {
                'mDatetime': 'T2_mDatetime',
                'Date': 'T2_Date',
                'Timestamp': 'T2_Timestamp',
                'Bid Price': 'T2_Bid_Price',
                'Ask Price': 'T2_Ask_Price',
                'Last Price': 'T2_Last_Price',
                'Volume': 'T2_Volume'
            },
            'rates2': {
                'mDatetime': 'R2_mDatetime',
                'Date': 'R2_Date',
                'Timestamp': 'R2_Timestamp',
                'Open': 'R2_Open',
                'High': 'R2_High',
                'Low': 'R2_Low',
                'Close': 'R2_Close',
                'tick_volume': 'R2_Tick Volume',
                'Volume': 'R2_Volume',
                'vol2': 'R2_Vol1',
                'vol3': 'R2_Vol3'
            }
        }

        date_columns = {
            'ticks1': ('time', '%Y%m%d', 's', 'f'),
            'rates1': ('time', '%Y%m%d', 's', 'f'),
            'ticks2': ('Date', '%Y%m%d', 's', 'e'),
            'rates2': ('Date', '%Y%m%d', 's', 'e'),
        }

        time_columns = {
            'ticks1': ('time_msc', '%Y%m%d %H:%M:%S', 'ms', 'a'),
            'ticks2': ('Timestamp', '%H:%M:%S', 'ms', 'a'),
            'rates2': ('Timestamp', '%H:%M:%S', 's', 'a'),
        }

        conv_columns = {
            'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', 's', 'b'),
            'rates1': ('R1_Date', '%Y%m%d %H:%M:%S.%f', 's', 'b'),
            'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', 's', 'b'),
            'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', 's', 'b'),
        }

        drop_columns = {
            'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g', ['T1_Time_Msc', 'T1_Flags', 'T1_Last Price', 'T1_Real_Volume', 'T1_Volume']),
            'rates1': ('R1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g', ['R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume']),
            'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g', ['T2_Timestamp', 'T2_Volume', 'T2_Last_Price']),
            'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g', ['R2_Timestamp', 'R2_Volume', 'R2_Vol1'])
        }

        merge_columns = {
            'ticks1': ('T1_Date', 'T1_Timestamp', 'T1_mDatetime', '%Y%m%d %H:%M:%S', '%H:%M:%S'),
            'rates1': ('R1_Date', 'R1_Timestamp', 'R1_mDatetime', '%Y%m%d %H:%M:%S', '%H:%M:%S'),
            'ticks2': ('T2_Date', 'T2_Timestamp', 'T2_mDatetime', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M:%S'),
            'rates2': ('R2_Date', 'R2_Timestamp', 'R2_mDatetime', '%Y%m%d %H:%M:%S', '%Y%m%d %H:%M:%S'),
        }

        if mp_filesrc in mappings:
            logger.info(f"Processing {mp_filesrc} data")
            # Convert date column if defined
            if mp_filesrc in date_columns:
                col, fmt, unit, conv_type = date_columns[mp_filesrc]
                convert_datetime(df, col, fmt=fmt, unit=unit, conv_type=conv_type)
            # Convert time column if defined
            if mp_filesrc in time_columns:
                col, fmt, unit, conv_type = time_columns[mp_filesrc]
                logger.info(f"Processing time column {col} for {mp_filesrc}")
                convert_datetime(df, col, fmt=fmt, unit=unit, conv_type=conv_type)
            # Rename columns
            rename_columns(df, mappings[mp_filesrc])
            # Merge datetime columns if required
            if mp_filesrc in merge_columns and self.mp_merge:
                col_date, col_time, merged_col, date_fmt, time_fmt = merge_columns[mp_filesrc]
                df = merge_datetime(df, col_date, col_time, merged_col, date_fmt, time_fmt)
            # Convert datetime columns if required
            if mp_filesrc in conv_columns and self.mp_convert:
                col, fmt, unit, conv_type = conv_columns[mp_filesrc]
                convert_datetime(df, col, fmt=fmt, unit=unit, conv_type=conv_type)
            # Drop unnecessary columns if required
            if mp_filesrc in drop_columns and self.mp_drop:
                col, fmt, unit, conv_type, drop_cols = drop_columns[mp_filesrc]
                logger.info(f"Dropping columns for {mp_filesrc}: {drop_cols}")
                convert_datetime(df, col, fmt=fmt, unit=unit, conv_type=conv_type, drop_cols=drop_cols)

            # Apply filtering conversions
            if self.filter_int:
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if self.filter_flt:
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if self.filter_obj:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            if self.filter_dtmi:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('int64'))
            if self.filter_dtmf:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('float64'))
            if self.mp_dropna:
                df.fillna(0, inplace=True)
                logger.info("Filled NaN values with 0")
            # Final column resorting if merged
            if mp_filesrc in merge_columns and self.mp_merge:
                _, _, merged_col, _, _ = merge_columns[mp_filesrc]
                df = resort_columns(df, merged_col)
                
            utils_config.run_mql_print(df, self.hrows, colwidth=self.colwidth, floatfmt='.5f', numalign='left', stralign='left')
            logger.info("Dataframe headers after wrangling printed successfully.")
        return df

    def establish_common_feat_col(self, df: pd.DataFrame, df_name: str):
        """
        Create common feature column for tick and OHLC data.
        """
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        
        bid_column = df_params.get("bid_column")
        ask_column = df_params.get("ask_column")
        column_out1 = df_params.get("column_out1")

        # For tick data, calculate average of bid and ask
        if df_name in ["df_api_ticks", "df_file_ticks"]:
            df[column_out1] = (df[bid_column] + df[ask_column]) / 2
            if self.run_avg:
                df[df_params.get("hl_avg_col")] = df[column_out1]
                logger.info("Calculated bid-ask average for tick data.")
        # For OHLC data, use close column
        elif df_name in ["df_api_rates", "df_file_rates"]:
            close_column = df_params.get("close_column")
            if close_column is None:
                raise ValueError("`close_column` must be provided for OHLC data.")
            df[column_out1] = df[close_column]
            utils_config.run_mql_print(df, self.hrows, colwidth=self.colwidth, floatfmt='.5f', numalign='left', stralign='left')
            logger.info("Established common feature column for OHLC data.")

    def establish_common_feat_col_scaled(self, df: pd.DataFrame, df_name: str):
        """Create a scaled column for the common feature column."""
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
         
        column_out1 = df_params.get("column_out1")
        column_out2 = df_params.get("column_out2")
   
        if column_out1 in df.columns:
            df[column_out2] = df[column_out1].pct_change().fillna(0)
            logger.info(f"Created scaled column: {column_out2}")

    def run_average_columns(self, df: pd.DataFrame, df_name: str) -> pd.DataFrame:
        """Compute moving average, log returns, future returns and other metrics."""
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        
        column_in = df_params.get("column_in")
        column_out1 = df_params.get("column_out1")
        column_out2 = df_params.get("column_out2")
        lookahead_periods = df_params.get("lookahead_periods")
        ma_window = df_params.get("ma_window")
        hl_avg_col = df_params.get("hl_avg_col")
        ma_col = df_params.get("ma_col")
        returns_col = df_params.get("returns_col")
        shift_in = df_params.get("shift_in")

        if column_in not in df.columns:
            logger.error(f"Column {column_in} not found in DataFrame.")
            return df

        # Calculate moving average
        if ma_window and ma_col:
            logger.info(f"Calculating moving average for {column_in} with window {ma_window}")
            df[ma_col] = self.calculate_moving_average(df, column_in, ma_window)
            logger.info(f"Moving average column {ma_col} created.")

        # Calculate log returns
        if shift_in and returns_col:
            logger.info(f"Calculating log returns for {column_in} with shift {shift_in}")
            df[returns_col] = self.calculate_log_returns(df, column_in, shift_in)
            logger.info(f"Log returns column {returns_col} created.")

        # Calculate log stationary if enabled
        if self.log_stationary and ma_col in df.columns:
            logger.info(f"Calculating log stationary for {ma_col}")
            df[ma_col] = self.calculate_log_stationary(df, ma_col)
            logger.info(f"Log stationary for {ma_col} computed.")
        
        # Calculate future returns
        if lookahead_periods and returns_col:
            logger.info(f"Calculating future returns for {column_in} with lookahead {lookahead_periods}")
            df[returns_col] = self.calculate_future_returns(df, column_in, lookahead_periods)
            logger.info(f"Future returns column {returns_col} created.")

        # Remove rows with zeros in returns column if required
        if self.remove_zeros and returns_col in df.columns:
            logger.info(f"Removing rows with zero in {returns_col}")
            df[returns_col] = self.run_remove_zeros(df, returns_col)
            logger.info(f"Rows with zeros removed from {returns_col}.")
        
        # Add row numbers if enabled
        if self.rownumber:
            self.set_row_numbers(df)
            logger.info("Row numbers added to DataFrame.")

        # Move specified column to end if last_col is provided
        if self.last_col:
            df = self.move_col_to_end(df, self.last_col)
            logger.info(f"Column {self.last_col} moved to the end.")

        # Create label scaled column if enabled
        if self.create_label_scaled and column_out2:
            df[column_out2] = df[column_in].shift(-lookahead_periods)
            df.dropna(inplace=True)
            logger.info(f"Label column {column_out2} created.")

        utils_config.run_mql_print(df, self.hrows, self.colwidth)
        return df

    def calculate_moving_average(self, df: pd.DataFrame, column: str, window: int):
        """Calculate moving average for a given column."""
        return df[column].rolling(window=window, min_periods=1).mean().fillna(method="bfill")
    
    def calculate_log_returns(self, df: pd.DataFrame, column: str, shift: int):
        """Calculate log returns for a given column."""
        df[column] = df[column].fillna(method='ffill')
        if (df[column] <= 0).sum() > 0:
            raise ValueError(f"Column '{column}' contains non-positive values, invalid for log returns.")
        return np.log(df[column] / df[column].shift(shift)).dropna()
    
    def calculate_log_stationary(self, df: pd.DataFrame, column: str):
        """Calculate log stationary transformation for a given column."""
        return np.log(df[column]).diff().fillna(0)
    
    def calculate_future_returns(self, df: pd.DataFrame, column: str, shift: int):
        """Calculate future returns for a given column."""
        return df[column].pct_change(periods=shift).fillna(0)
    
    def run_remove_zeros(self, df: pd.DataFrame, returns_col: str):
        """Remove rows where the returns column equals zero."""
        return df[df[returns_col] != 0]
    
    def set_row_numbers(self, df: pd.DataFrame):
        """Add a row number column to the DataFrame."""
        df['rownumber'] = range(1, len(df) + 1)
    
    def move_col_to_end(self, df: pd.DataFrame, last_col: str):
        """Move specified column to the end of the DataFrame."""
        if last_col not in df.columns:
            logger.warning(f"Column {last_col} not found. Skipping move_col_to_end.")
            return df
        cols = [col for col in df.columns if col != last_col] + [last_col]
        return df[cols]

    def create_index_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create an index column from the first column of the DataFrame.
        """
        first_col = df.columns[0]
        df[first_col] = df[first_col].fillna('Unknown').astype(str).str.strip()
        df.set_index(first_col, inplace=True)
        return df.dropna()
