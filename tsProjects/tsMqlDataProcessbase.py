"""
#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlDataProcess/tsMqlDataProcess.py
Description: Load and add files and data parameters.
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
import textwrap
from tabulate import tabulate
import textwrap
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
        self._logger = logging.getLogger(__name__)
        self.util_config = CUtilities()
        self.env = CMqlEnvMgr()
        self.local_data_params = {}  
        self.params = self.env.all_params()

        # Extract parameter sections
        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

        self.colwidth = kwargs.get('colwidth', 20)
        self.hrows = kwargs.get('hrows', 5)

        self._initialize_mql()
        self._set_global_parameters(kwargs)

       # Store column parameters after initialization
        self.COLUMN_PARAMS = {
            "df_api_ticks": {
                'bid_column': 'T1_Bid_Price',
                'ask_column':'T1_Ask_Price',
                'column_in':'T1_Bid_Price',
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
                'ask_column':'R1_Close',
                'column_in':'R1_Open',
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
                'ask_column':'T2_Ask_Price',
                'column_in':'T2_Bid_Price',
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
                'ask_column':'R2_Close',
                'column_in':'R2_Open',
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
        self.mp_data_filename1 = self.params.get('data', {}).get('mp_data_filename1', 'default1.csv')
        self.mp_data_filename2 = self.params.get('data', {}).get('mp_data_filename2', 'default2.csv')
        self.rownumber = self.params.get("mp_data_self.rownumber")

        # Machine learning parameters
        self.mp_ml_input_keyfeat = self.params.get('ml', {}).get('mp_ml_input_keyfeat', 'KeyFeature')
        self.mp_ml_input_keyfeat_scaled = self.params.get('ml', {}).get('mp_ml_input_keyfeat_scaled', 'KeyFeature_Scaled')
        self.lookahead_periods = self.params.get('ml', {}).get('mp_lookahead_periods', 1)
        self.ma_window = self.params.get('ml', {}).get('mp_ma_window', 10)
        self.hl_avg_col = self.params.get('ml', {}).get('mp_hl_avg_col', 'HL_Avg')
        self.ma_col = self.params.get('ml', {}).get('mp_ma_col', 'MA')
        self.returns_col = self.params.get('ml', {}).get('mp_returns_col', 'Returns')
        self.shift_in = self.params.get('ml', {}).get('mp_shift_in', 1)
        self.create_label = self.params.get('ml', {}).get('mp_create_label', False)
        #  Run states
        self. run_avg = self.params.get('ml', {}).get('mp_run_avg', False)
        self.run_avg_scaled = self.params.get('ml', {}).get('mp_run_avg_scaled', False)
        self.log_stationary = self.params.get('ml', {}).get('mp_log_stationary', False)
        self.rownumber = self.params.get('ml', {}).get('mp_rownumber', False)
        self.remove_zeros = self.params.get('ml', {}).get('mp_remove_zeros', False)
        self.last_col = self.params.get('ml', {}).get('mp_last_col', False)
        self.create_label_scaled = self.params.get('ml', {}).get('mp_create_label_scaled', False)
       

        self.lp_utc_from = kwargs.get('lp_utc_from', datetime.utcnow())
        self.lp_utc_to = kwargs.get('lp_utc_to', datetime.utcnow())
        self.lp_app_primary_symbol = kwargs.get('lp_app_primary_symbol', self.params.get('app', {}).get('mp_app_primary_symbol', 'EURUSD'))
        self.lp_app_rows = kwargs.get('lp_app_rows', self.params.get('app', {}).get('mp_app_rows', 1000))
        self.lp_timeframe = kwargs.get('lp_timeframe', self.params.get('app', {}).get('mp_app_timeframe', 'H4'))
        self.mp_glob_data_path = kwargs.get('mp_glob_data_path', self.params.get('base', {}).get('mp_glob_data_path', 'Mql5Data'))

        # Derived filenames
        self.mp_data_filename1_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename1}.csv"
        self.mp_data_filename2_merge = f"{self.lp_app_primary_symbol}_{self.mp_data_filename2}.csv"

   
    #---------------------------------------
    # create method  "run_wrangle_service".
    # class: CDataProcess      
    # usage: mql data
    # /param  var   
    # ------------------------------------ 

    def run_wrangle_service(self, **kwargs):
         """Run the data loader service."""
         self.df_api_ticks = kwargs.get('df_api_ticks', pd.DataFrame())
         self.df_api_rates = kwargs.get('df_api_rates', pd.DataFrame())
         self.df_file_ticks = kwargs.get('df_file_ticks', pd.DataFrame())
         self.df_file_rates = kwargs.get('df_file_rates', pd.DataFrame())

         self.mp_unit = kwargs.get('UNIT', {})
         self.loadmql = kwargs.get('loadmql', True)
         self.mp_filesrc = kwargs.get('mp_filesrc', 'ticks1')
         self.filter_int = kwargs.get('filter_int', False)
         self.filter_flt=self.filter_flt = kwargs.get('filter_flt=self.filter_flt', False)
         self.filter_obj=self.filter_obj = kwargs.get('filter_obj=self.filter_obj', False)
         self.filter_dtmi = kwargs.get('filter_dtmi', False)
         self.filter_dtmf = kwargs.get('filter_dtmf', False)
         self.mp_dropna = kwargs.get('mp_dropna',True)
         self.mp_convert = kwargs.get('mp_convert', True)
         self.mp_drop = kwargs.get('mp_drop', True)

         if len(self.df_api_ticks) > 0 and self.loadmql:
               self.df_api_ticks = self.wrangle_time(self.df_api_ticks, self.mp_unit, mp_filesrc="ticks1", filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_api_rates) > 0 and self.loadmql:
               self.df_api_rates = self.wrangle_time(self.df_api_rates,   self.mp_unit, mp_filesrc="rates1",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_file_ticks) > 0 and (self.loadmql == True or self.loadmql == False):
               self.df_file_ticks = self.wrangle_time(self.df_file_ticks,   self.mp_unit, mp_filesrc="ticks2",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_file_rates) > 0 and (self.loadmql == True or self.loadmql == False):
               self.df_file_rates = self.wrangle_time(self.df_file_rates,  self.mp_unit, mp_filesrc="rates2",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         
         return self.df_api_ticks, self.df_api_rates, self.df_file_ticks, self.df_file_rates


    def wrangle_time(self, df: pd.DataFrame, lp_unit: str, mp_filesrc: str, filter_int: bool, filter_flt: bool, filter_obj: bool, filter_dtmi: bool, filter_dtmf: bool, mp_dropna: bool, mp_merge: bool, mp_convert: bool, mp_drop: bool) -> pd.DataFrame:
        """
        Wrangles time-related data in the DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            lp_unit (str): Unit of time.
            mp_filesrc (str): Source of the file.
            filter_int (bool): Whether to filter integer columns.
            filter_flt (bool): Whether to filter float columns.
            filter_obj (bool): Whether to filter object columns.
            filter_dtmi (bool): Whether to filter datetime columns to int.
            filter_dtmf (bool): Whether to filter datetime columns to float.
            mp_dropna (bool): Whether to drop NaN values.

        Returns:
            pd.DataFrame: The wrangled DataFrame.
        """
        
        def rename_columns(df: pd.DataFrame, mapping: dict) -> None:
            valid_renames = {old: new for old, new in mapping.items() if old in df.columns}
            df.rename(columns=valid_renames, inplace=True)
       
        def merge_datetime(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc): 
            if col1 in df.columns and col2 in df.columns:
                try:
                    logger.info(f"Merging {mp_filesrc} {col1} and {col2} to {mcol}")
                    df[mcol] = pd.to_datetime(df[col1].dt.strftime('%Y-%m-%d') + ' ' + df[col2].dt.strftime('%H:%M:%S.%f'), format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
                    df.drop([col1, col2], axis=1, inplace=True)
                    # Reorder columns
                    logger.info("MDatetimeColumns: mcol",mcol,"col1",col1,"col2",col2,"filesrc",mp_filesrc)
                    datetime_col = mcol if mcol in df.columns else df.columns[0]
                    logger.info("Reordering columns with datetime first mcol:", mcol)
                    df = df[[datetime_col] + [col for col in df.columns if col != datetime_col]]
                    logger.info("Reordered columns with datetime first")
                    logger.info("Columns:", df.columns)
                except Exception as e:
                    logger.info(f"Error merging {mp_filesrc} {col1} and {col2} to {mcol}: {e}")
            return df

        def resort_columns(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc) -> pd.DataFrame:
            if mcol in df.columns:
                datetime_col = mcol if mcol in df.columns else df.columns[0]
                df = df[[datetime_col] + [col for col in df.columns if col != datetime_col]]
                logger.info("Reordered columns with datetime first")
                logger.info("Columns:", df.columns)
            return df

        def convert_datetime(df: pd.DataFrame, column: str, fmt: str = None, unit: str = None, type: str = None,dcol1 = None, dcol2 = None, dcol3 = None, dcol4= None,dcol5 = None) -> None:
            try:
                if type == 'a':
                    logger.info(f"Converting:a {mp_filesrc} {column} to datetime with stfttime hours string: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%H:%M:%S.%f'), format='%H:%M:%S.%f', errors='coerce', utc=True)
                elif type == 'b':
                    logger.info(f"Converting:b {mp_filesrc} {column} to datetime with tf date model: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df.pop(column), format=fmt, errors='coerce')
                elif type == 'c':
                    logger.info(f"Converting:c {mp_filesrc} {column} to datetime with stfttime years string: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%d/%m/%y %H:%M:%S.%f'), format='%d/%m/%y %H:%M:%S.%f', errors='coerce', utc=True)
                elif type == 'd':
                    logger.info(f"Converting:d {mp_filesrc} {column} to datetime with tf time: type {type} and format {fmt}")
                    df[column] = df[column].map(pd.Timestamp.timestamp)
                elif type == 'e':
                    logger.info(f"Converting:e {mp_filesrc} {column} to datetime with format {fmt}: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                elif type == 'f':
                    logger.info(f"Converting:f {mp_filesrc} {column} to datetime with unit {unit}: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
                elif type == 'g':
                    logger.info(f"Dropping:g {mp_filesrc} {column} with type {type}")
                    logger.info("Columns to drop:", dcol1, dcol2, dcol3, dcol4,dcol5)
                    logger.info("Columns:", df.columns)
                    if column in df.columns:
                        logger.info(f"Dropping column: {column} found in ", df.columns)
                        columns_to_drop = [col for col in [dcol1, dcol2, dcol3, dcol4,dcol5] if col in df.columns]
                        if columns_to_drop:
                            logger.info(f"Dropping columns: {columns_to_drop} from", df.columns)
                            df.drop(columns_to_drop, axis=1, inplace=True)
            except Exception as e:
                logger.info(f"Error converting {mp_filesrc} {column} {type}: {e}")

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
            'ticks1': ('time_msc', '%Y%m%d %H:%M:%S', 'ms', 'f'),
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
           'ticks1': ('T1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g','T1_Time_Msc', 'T1_Flags', 'T1_Last Price','T1_Real_Volume', 'T1_Volume' ),
           'rates1': ('R1_Date', '%Y%m%d %H:%M:%S', 'ms', 'g','R1_Tick_Volume', 'R1_spread', 'R1_Real_Volume', None,None),
           'ticks2': ('T2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g','T2_Timestamp', 'T2_Volume', 'T2_Last_Price', None, None),
           'rates2': ('R2_mDatetime', '%Y%m%d %H:%M:%S', 'ms', 'g','R2_Timestamp', 'R2_Volume', 'R2_Vol1', None, None),

        }

        # Columns are renamed before merging

        merge_columns = {
            'ticks1': ('T1_Date', 'T1_Timestamp', 'T1_mDatetime','%Y%m%d %H:%M:%S','%H:%M:%S'),
            'rates1': ('R1_Date', 'R1_Timestamp', 'R1_mDatetime','%Y%m%d %H:%M:%S','%H:%M:%S'),
            'ticks2': ('T2_Date', 'T2_Timestamp', 'T2_mDatetime','%Y%m%d %H:%M:%S','%Y%m%d %H:%M:%S'),
            'rates2': ('R2_Date', 'R2_Timestamp', 'R2_mDatetime','%Y%m%d %H:%M:%S','%Y%m%d %H:%M:%S'),
        }

        if mp_filesrc in mappings:
            logger.info(f"Processing {mp_filesrc} data")
            
            if mp_filesrc in date_columns:
                column, fmt, unit, type = date_columns[mp_filesrc]
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)

            if mp_filesrc in time_columns:
                column, fmt, unit, type = time_columns[mp_filesrc]
                logger.info(f"Columns Time: {column}, Format: {fmt}, Unit: {unit}, Type: {type}, Filesrc: {mp_filesrc}")

                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)

            # Rename columns
            rename_columns(df, mappings[mp_filesrc])
                
            # Merge datetime columns
            if mp_filesrc in merge_columns and mp_merge:
                col1, col2, mcol, mfmt1, mfmt2 = merge_columns[mp_filesrc]
                df = merge_datetime(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc)
                

            # Convert datetime columns with tf
            if mp_filesrc in conv_columns and mp_convert:
                column, fmt, unit, type = conv_columns[mp_filesrc]
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)

            # drop columns 
            if mp_filesrc in drop_columns and mp_drop:
                column, fmt, unit, type,col1,col2,col3,col4,col5 = drop_columns[mp_filesrc]
                logger.info("Columns Drop:", column, "Format:", fmt, "Unit:", unit, "Type:", type, "Filesrc:", mp_filesrc, "Columns:", col1, col2, col3, col4, col5)
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type,dcol1=col1,dcol2=col2,dcol3=col3,dcol4=col4,dcol5=col5)
                
            if filter_int:
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info("Columns Numeric:", col)
            if filter_flt:
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info("Columns Numeric:", col)
            if filter_obj:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info("Columns Object:", col)
            if filter_dtmi:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('int64'))
                    logger.info("Columns DateTime:", col)
            if filter_dtmf:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('float64'))
                    logger.info("Columns DateTime:", col)
            if mp_dropna:
                df.fillna(0, inplace=True)
                logger.info("Dropped NaN values")
            
             # final sort
            if mp_filesrc in merge_columns and mp_merge:
                col1, col2, mcol, mfmt1, mfmt2 = merge_columns[mp_filesrc]
                df = resort_columns(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc) 

            self.util_config.run_mql_print(df,self.hrows, colwidth=self.colwidth, floatfmt='.5f', numalign='left', stralign='left')
            logger.info("Dataframe headers after wrangle printed successfully.")
        return df

    def establish_common_feat_col(self, df, df_name):
       # create common Close colum across data types tick and ohlc
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        
        bid_column = df_params.get("bid_column")
        ask_column = df_params.get("ask_column")

        column_in = df_params.get("column_in")
        column_out1 = df_params.get("column_out1")
        column_out2 = df_params.get("column_out2")
        lookahead_periods = df_params.get("lookahead_periods")
        ma_window = df_params.get("ma_window")
        hl_avg_col = df_params.get("hl_avg_col")
        ma_col = df_params.get("ma_col")
        returns_col = df_params.get("returns_col")
        shift_in = df_params.get("shift_in")
        create_label = df_params.get("create_label")
       

        # Tick data
        if df_name == "df_api_ticks" or df_name == "df_file_ticks":
            df[column_out1] = (df[bid_column] + df[ask_column]) / 2
            if self.run_avg:
                df[hl_avg_col] = df[column_out1]
                logging.info("Bid-ask average calculated.")

        # OHLC data
        elif df_name == "df_api_rates" or df_name == "df_file_rates":
            open_column = df_params.get("open_column")
            high_column = df_params.get("high_column")
            low_column = df_params.get("low_column")
            close_column = df_params.get("close_column")

            if close_column is None:
                raise ValueError("`close_column` must be provided for run modes 2 or 4.")
            df[column_out1] = df[close_column]
            self.util_config.run_mql_print(df, self.hrows, colwidth=self.colwidth, floatfmt='.5f', numalign='left', stralign='left')
            logger.info("Dataframe headers after establish_common_feat_col printed successfully.")
           
    def establish_common_feat_col_scaled(self, df, df_name):
         """Create a scaled column for the common feature column."""
         df_params = self.COLUMN_PARAMS.get(df_name, {})
         logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
         
         column_out1 = df_params.get("column_out1")
         column_out2 = df_params.get("column_out2")
         lookahead_periods = df_params.get("lookahead_periods")
         ma_window = df_params.get("ma_window")
         hl_avg_col = df_params.get("hl_avg_col")
         ma_col = df_params.get("ma_col")
         returns_col = df_params.get("returns_col")
         shift_in = df_params.get("shift_in")
         create_label = df_params.get("create_label")
   
         # Create scaled column for the common feature column
         if column_out1 in df.columns:
               df[column_out2] = df[column_out1].pct_change().fillna(0)
               logger.info(f"Created scaled column: {column_out2}")

    # +-------------------------------------------------------------------
    # Run average methods
    # +-------------------------------------------------------------------
    def run_average_columns(self,df,df_name):
        """Compute high-low averages, moving averages, and log returns."""
        
        df_params = self.COLUMN_PARAMS.get(df_name, {})
        logger.info(f"Processing DataFrame: {df_name} with parameters: {df_params}")
        
        # read relevant columns from the parameters store
        column_in = df_params.get("column_in")
        column_out1 = df_params.get("column_out1")
        column_out2 = df_params.get("column_out2")
        lookahead_periods = df_params.get("lookahead_periods")
        ma_window = df_params.get("ma_window")
        hl_avg_col = df_params.get("hl_avg_col")
        ma_col = df_params.get("ma_col")
        returns_col = df_params.get("returns_col")
        shift_in = df_params.get("shift_in")
        create_label = df_params.get("create_label")

        if column_in not in df.columns:
            logger.error(f"Column {column_in} not found in DataFrame.")
            return df

        # 1: Calculate moving average
        if ma_window and ma_col:
            logger.info(f"Calculating moving average for column: {column_in} with window: {ma_window}")
            df[ma_col] = self.calculate_moving_average(df, column_in, ma_window)
            logger.info(f"Created moving average column: {ma_col}")

        # 2: Calculate log returns
        if shift_in and returns_col:
            logger.info(f"Calculating log returns for column: {column_in} with shift: {shift_in}")
            df[returns_col] = self.calculate_log_returns(df, column_in, shift_in)
            logger.info(f"Created log returns column: {returns_col}")

        # 3: Calculate log stationary
        if self.log_stationary and ma_col in df.columns:
            logger.info(f"Calculating log stationary for column: {ma_col}")
            df[ma_col] = self.calculate_log_stationary(df, ma_col)
            logger.info(f"Created log stationary column: {ma_col}")
        
        # 4: Calculate future returns
        if lookahead_periods and returns_col:
            logger.info(f"Calculating future returns for column: {column_in} with lookahead: {lookahead_periods}")
            df[returns_col] = self.calculate_future_returns(df, column_in, lookahead_periods)
            logger.info(f"Created future returns column: {returns_col}")

        # 5: Remove rows with zeros in the returns column if required
        if self.remove_zeros and returns_col in df.columns:
            logger.info(f"Removing rows with zeros in the returns column: {returns_col}")
            df[returns_col] = self.run_remove_zeros(df, returns_col)
            logger.info(f"Removed rows with zeros in the returns column: {returns_col}")
        
        # 6: Add row numbers to the DataFrame if required
        if self.rownumber:
            self.set_row_numbers(df)
            logger.info("Added row numbers to the DataFrame.")

        # 7: Move specified column to the end if it exists
        if self.last_col:
            df = self.move_col_to_end(df, last_col)
            logger.info(f"Moved column {last_col} to the end.")
        
        # 8: Create label scaled column
        if self.create_label_scaled and column_out2:
            df[column_out2] = df[column_in].shift(-lookahead_periods)
            df.dropna(inplace=True)
            logger.info(f"Created label column: {column_out2}")
        
        # 9: Print final DataFrame
        self.util_config.run_mql_print(df, self.hrows, self.colwidth)
        return df

    def calculate_moving_average(self, df, column, window):
        """Calculate moving average."""
        return df[column].rolling(window=window, min_periods=1).mean().fillna(method="bfill")
    
    def calculate_log_returns(self, df, column, shift):
        """Calculate log returns."""
        df[column] = df[column].fillna(method='ffill')
        if (df[column] <= 0).sum() > 0:
            raise ValueError(f"Column '{column}' contains non-positive values, which are invalid for log returns.")
        return np.log(df[column] / df[column].shift(shift)).dropna()
    
    def calculate_log_stationary(self, df, column):
        """Calculate log stationary."""
        return np.log(df[column]).diff().fillna(0)
    
    def calculate_future_returns(self, df, column, shift):
        """Calculate future returns."""
        return df[column].pct_change(periods=shift).fillna(0)
    
    def run_remove_zeros(self, df, returns_col):
        """Remove rows with zeros in the returns column."""
        return df[df[returns_col] != 0]
    
    def set_row_numbers(self, df):
        """Add row numbers to the DataFrame."""
        df['rownumber'] = range(1, len(df) + 1)
    
    def move_col_to_end(self, df, last_col):
        """Move specified column to the end if it exists."""
        if last_col not in df.columns:
            logger.warning(f"Column {last_col} not found in DataFrame. Skipping move_col_to_end.")
            return df
        cols = [col for col in df.columns if col != last_col] + [last_col]
        return df[cols]


    def create_index_column(self,df):
        """Select the data file based on the DataFrame name."""
        # Ensure no missing values, duplicates, or type issues
        first_column = df.columns[0]
        df[first_column] =df[first_column].fillna('Unknown')  # Handle NaNs
        df[first_column] =df[first_column].astype(str)  # Uniform type
        df[first_column] =df[first_column].str.strip()  # Remove whitespaces
        # Set the first column as index which is the datetime column
        df.set_index(first_column, inplace=True)
        df=df.dropna()
        return df