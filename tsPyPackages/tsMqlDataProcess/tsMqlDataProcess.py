"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlDataProcess.py
File: tsPyPackages/tsMqlDataProcess/tsMqlDataProcess.py
Description: Load and add files and data parameters.
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

# packages dependencies for this module
#
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")


import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import arrow
import pytz
from sklearn.preprocessing import MinMaxScaler
import logging
import tabulate
from tabulate import tabulate
import textwrap

# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr

#--------------------------------------------------------------------
# create class  "CDataProcess"
# usage: mql data services
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CDataProcess:
    def __init__(self, **kwargs):
        if loadmql:
               import MetaTrader5 as mt5
               self.mt5 = mt5
        else:
            self.mt5 = None
            
        self.env = CMqlEnvMgr()
        self.params = self.env.all_params()

        self.base_params = self.params.get("base", {})
        self.data_params = self.params.get("data", {})
        self.ml_params = self.params.get("ml", {})
        self.mltune_params = self.params.get("mltune", {})
        self.app_params = self.params.get("app", {})

        self._initialize_mql()
        

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
        logger.debug(f"self.data_params: {self.data_params}")

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
        return df

    def calculate_moving_average(self,df, column, window,min_periods=1):
        logger.info("Calculating moving average for column:", column, " with window:", window)
        # Calculate Moving Averages
        df['SMA'] = df[column].rolling(window=min_periods).mean()  # 14-day Simple Moving Average
        df["SMA"] = df["SMA"].fillna(method="bfill")  # or "ffill"
        df = df.dropna(subset=["SMA"])
        return df['SMA']

    def calculate_log_returns(self, df, column, shift):
        logger.info("Calculating log returns for column:", column, "with shift:", shift)
        # Check for valid values
        if df[column].isna().sum() > 0:
            logger.info("Warning: Column contains NaN values. Filling missing values.")
            df[column] = df[column].fillna(method='ffill')

        if (df[column] <= 0).sum() > 0:
            raise ValueError(f"Column '{column}' contains non-positive values, which are invalid for log returns.")

        # Ensure shift is applied before calculating log returns
        shifted_column = df[column].shift(shift)
        return np.log(df[column] / shifted_column).dropna()

    # create method  "run_shift_data1()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_shift_data1(self, df, lp_seconds, lp_unit):
        lv_seconds = lp_seconds
        lv_number_of_rows = lv_seconds
        df.style.set_properties(**{'text-align': 'left'})
        df['time'] = pd.to_datetime(df['time'], unit=lp_unit)
        df['close'] = (df['ask'] + df['bid']) / 2
        lv_empty_rows = pd.DataFrame(np.nan, index=range(lv_number_of_rows), columns=df.columns)
        df = pd.concat([df, lv_empty_rows])
        df['label'] = df['close'].shift(-lv_seconds)
        df = df.dropna()
        df.style.set_properties(**{'text-align': 'left'})
        logger.info("lpDf", df.tail(10))
        return df

    # create method  "create_dataset()".
    # class: cmqldatasetup      
    # usage: # Creating time steps for LSTM input
    # /param  var
    def create_dataset(self, data, lp_seconds):
        X_data, y_data = [], []
        for i in range(len(data) - lp_seconds - 1):
            X_data.append(data[i:(i + lp_seconds), 0])
            y_data.append(data[i + lp_seconds, 0])
        return np.array(X_data), np.array(y_data)

   
    # create method  "run_shift_data2()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_shift_data2(self, df, lp_seconds=60, lp_unit='s'):
        # Selecting the 'Close' column for prediction
        close_prices = df['close'].values.reshape(-1, 1)
        #Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        X, y = self.create_dataset(scaled_data, lp_seconds)
        # Reshaping data to the format required by LSTM
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y


    
    # create method  "run_mql_print".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var    
    def run_mql_print(self, df, hrows, colwidth, tablefmt="pretty", floatfmt=".5f", numalign="left", stralign="left"):
        logger.info("Type of df before run_mql_print:", type(df))

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a DataFrame, but got {type(df)}")

        if df.empty:
            logger.info("Warning: The DataFrame is empty. Nothing to display.")
            return

        # Ensure `hrows` is valid
        hrows = min(hrows, len(df))  # Prevent `hrows` from exceeding available rows

        # Wrap long text in columns
        def wrap_column_data(column, width):
            return column.apply(lambda x: '\n'.join(textwrap.wrap(str(x), width)) if pd.notnull(x) else "")

        df = df.apply(lambda col: wrap_column_data(col, colwidth))

        # Display the table
        logger.info(tabulate(
            df.head(hrows),
            showindex=False,
            headers=df.columns,
            tablefmt=tablefmt,
            numalign=numalign,
            stralign=stralign,
            floatfmt=floatfmt,
            maxcolwidths=[colwidth] * len(df.columns)  # Ensure max column width
        ))

    # create method  "move_col_to_end".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var    
    # Reorder columns to move the label column to the end
    def move_col_to_end(self,df, last_col):
        cols = [col for col in df.columns if col != last_col] + [last_col]
        return df[cols]


    # create method  "create_label_wrapper()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var           
    def create_label_wrapper(
        self,
        df,
        lookahead_periods,
        ma_window,
        bid_column,
        ask_column,
        column_in,
        column_out1,
        column_out2,
        open_column,
        high_column,
        low_column,
        close_column,
        run_mode,
        hl_avg_col,
        ma_col,
        returns_col,
        shift_in,
        rownumber,
        run_avg=True,
        run_ma=True,
        run_returns=True,
        run_future_returns=True,
        log_stationary=False,
        remove_zeros=True,
        create_label=False,
    ):
        """
        Wrapper function for `create_label` to handle the creation of label variables.

        Args:
            See `create_label` method for parameter details.

        Returns:
            The output of the `create_label` function.
        """
        params = {
            "df": df,
            "lookahead_periods": lookahead_periods,
            "ma_window": ma_window,
            "bid_column": bid_column,
            "ask_column": ask_column,
            "column_in": column_in,
            "column_out1": column_out1,
            "column_out2": column_out2,
            "open_column": open_column,
            "high_column": high_column,
            "low_column": low_column,
            "close_column": close_column,
            "run_mode": run_mode,
            "hl_avg_col": hl_avg_col,
            "ma_col": ma_col,
            "returns_col": returns_col,
            "shift_in": shift_in,
            "rownumber": rownumber,
            "run_avg": run_avg,
            "run_ma": run_ma,
            "run_returns": run_returns,
            "run_future_returns": run_future_returns,
            "log_stationary": log_stationary,
            "remove_zeros": remove_zeros,
        }

        return self.create_label(**params)

    def create_label(
        self,
        df,
        lookahead_periods,
        ma_window,
        bid_column,
        ask_column,
        column_in=None,
        column_out1=None,
        column_out2=None,
        open_column=None,
        high_column=None,
        low_column=None,
        close_column=None,
        hl_avg_col='HLAvg',
        ma_col='SMA',
        returns_col='Returns',
        shift_in=1,
        run_mode=1,
        run_avg=False,
        run_ma=False,
        log_stationary=False,
        run_returns=False,
        run_future_returns=False,
        remove_zeros=False,
        rownumber=False,
        createlabel=False,
    ):
        """
        Creates a label column in the DataFrame by calculating mid prices or shifting a specified column.

        Parameters:
            See docstring of `create_label_wrapper`.

        Returns:
            pd.DataFrame: DataFrame with the label column added.
        """
        if column_out1 is None:
            column_out1 = self.lp_features
        if column_out2 is None:
            column_out2 = self.lp_label

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input `df` must be a pandas DataFrame.")
        if not isinstance(lookahead_periods, int) or lookahead_periods <= 0:
            raise ValueError("`lookahead_periods` must be a positive integer.")
        if run_mode not in {1, 2, 3, 4}:
            raise ValueError("`run_mode` must be one of {1, 2, 3, 4}.")
        if run_mode in {1, 3} and column_in is None:
            raise ValueError("`column_in` must be provided for run modes 1 or 3.")
        if run_mode in {2, 4} and column_in is None:
            raise ValueError("`column_in` must be provided for run modes 2 or 4.")

        # Calculate base column based on run mode
        if run_mode in {1, 3}:  # Bid-ask average
            df[column_out1] = (df[bid_column] + df[ask_column]) / 2
            if run_avg:
                df[hl_avg_col] = df[column_out1]
                logging.info("Bid-ask average calculated.")

        elif run_mode in {2, 4}:  # High-low average
            if close_column is None:
                raise ValueError("`close_column` must be provided for run modes 2 or 4.")
            df[column_out1] = df[close_column]
            if run_avg:
                df[hl_avg_col] = (df[high_column] + df[low_column]) / 2
                logging.info("High-low average calculated.")

        # Apply moving average if required
        if run_ma:
            df[ma_col] = df[hl_avg_col].rolling(window=ma_window, min_periods=1).mean()
            logging.info("Moving averages calculated.")

        # Apply log stationary transformation if required
        if log_stationary:
            df[ma_col] = np.log(df[ma_col]).diff().fillna(0)
            logging.info("Log stationary transformation applied.")

        # Calculate returns if required
        if run_returns:
            df[returns_col] = df[column_out1].pct_change(periods=shift_in).fillna(0)
            logging.info("Returns calculated.")

        # Calculate future returns if required
        if run_future_returns:
            df[returns_col] = (df[column_out1].shift(-lookahead_periods) / df[column_out1]) - 1
            logging.info("Future Returns calculated.")

        # Set label column
        if createlabel:
            df[column_out2] = df[column_in].shift(-lookahead_periods)
            df.dropna(inplace=True)
            logging.info("Label column created.")
        
        # Remove rows with zeros in the returns column if required
        if remove_zeros and returns_col in df.columns:
            df = df[df[returns_col] != 0]

        # Add row numbers if required
        if rownumber:
            df['RowNumber'] = range(1, len(df) + 1)

        return df


    def prepare_data(self,tm, mp_symbol_primary, mp_unit, broker_config, mp_data_rows, mp_data_rowcount, TIMEFRAME):
        obj = CMqldatasetup(lp_features={'Close'}, lp_label={'Label'}, lp_label_count=1)
        CURRENTYEAR = datetime.now().year
        CURRENTDAYS = datetime.now().day
        CURRENTMONTH = datetime.now().month
        TIMEZONE = tm.TIME_CONSTANTS['TIMEZONES'][0]
        mv_data_utc_from = d1.set_mql_timezone(CURRENTYEAR - 5, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        mv_data_utc_to = d1.set_mql_timezone(CURRENTYEAR, CURRENTMONTH, CURRENTDAYS, TIMEZONE)
        mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates = d1.run_load_from_mql(
            True, True, True, True, "df_rates1", "df_rates2", mv_data_utc_from, mp_symbol_primary, mp_data_rows, mp_data_rowcount, mt5.COPY_TICKS_ALL, None, broker_config['MPDATAPATH'], broker_config['MPFILEVALUE1'], broker_config['MPFILEVALUE2'], TIMEFRAME
        )
        return mv_tdata1apiticks, mv_tdata1apirates, mv_tdata1loadticks, mv_tdata1loadrates

    def normalize_data(self,mv_tdata2, mp_ml_custom_input_keyfeat):
        scaler = MinMaxScaler()
        mp_ml_custom_input_keyfeat_list = list(mp_ml_custom_input_keyfeat)
        mp_ml_custom_input_keyfeat_scaled = [feat + '_Scaled' for feat in mp_ml_custom_input_keyfeat_list]
        mv_tdata2[mp_ml_custom_input_keyfeat_scaled] = scaler.fit_transform(mv_tdata2[mp_ml_custom_input_keyfeat_list])
        return mv_tdata2, scaler

    def split_data(self,mv_tdata2_X, mv_tdata2_y, mp_ml_validation_split, mp_ml_test_split, batch_size):
        X_train, X_temp, y_train, y_temp = train_test_split(mv_tdata2_X, mv_tdata2_y, test_size=(mp_ml_validation_split + mp_ml_test_split), shuffle=False)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(mp_ml_test_split / (mp_ml_validation_split + mp_ml_test_split)), shuffle=False)
        return X_train, X_val, X_test, y_train, y_val, y_test


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
         self.mp_dropna = kwargs.get('mp_dropna', False)
         self.mp_merge = kwargs.get('mp_merge', False)
         self.mp_convert = kwargs.get('mp_convert', False)
         self.mp_drop = kwargs.get('mp_drop', False)

         if len(self.df_api_ticks) > 0 and self.loadmql:
               self.df_api_ticks = self.wrangle_time(self.df_api_ticks, self.mp_unit, mp_filesrc="ticks1", filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_api_rates) > 0 and self.loadmql:
               self.df_api_rates = self.wrangle_time(self.df_api_rates,   self.mp_unit, mp_filesrc="rates1",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_file_ticks) > 0 and (self.loadmql == True or self.loadmql == False):
               self.df_file_ticks = self.wrangle_time(self.df_file_ticks,   self.mp_unit, mp_filesrc="ticks2",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         if len(self.df_file_rates) > 0 and (self.loadmql == True or self.loadmql == False):
               self.df_file_rates = self.wrangle_time(self.df_file_rates,  self.mp_unit, mp_filesrc="rates2",  filter_int=self.filter_int, filter_flt=self.filter_flt, filter_obj=self.filter_obj, filter_dtmi=self.filter_dtmi, filter_dtmf=self.filter_dtmf, mp_dropna=self.mp_dropna, mp_merge=self.mp_merge, mp_convert=self.mp_convert, mp_drop=self.mp_drop)
         
         return self.df_api_ticks, self.df_api_rates, self.df_file_ticks, self.df_file_rates


    def create_labels(self, df_ticks, df_rates, df_file_ticks, df_file_rates):
        """
        Function to create labels for different datasets.
        
        Returns:
        Tuple of updated DataFrames: (df_ticks, df_rates, df_file_ticks, df_file_rates)
        """
        lookahead_periods = self.params.get("mp_ml_cfg_period")
        ma_window = self.params.get("mp_ml_tf_ma_windowing")
        hl_avg_col = self.params.get("mp_ml_hl_avg_col")
        ma_col = self.params.get("mp_ml_ma_col")
        returns_col = self.params.get("mp_ml_returns_col")
        shift_in = self.params.get("mp_ml_tf_shiftin")

        def process_labels(df, bid_col, ask_col, col_in, run_mode):
            return self.create_label_wrapper(
                df=df,
                bid_column=bid_col,
                ask_column=ask_col,
                column_in=col_in,
                column_out1='Close',
                column_out2='Close_Scaled',
                open_column=f"R{run_mode}_Open",
                high_column=f"R{run_mode}_High",
                low_column=f"R{run_mode}_Low",
                close_column=f"R{run_mode}_Close",
                run_mode=run_mode,
                lookahead_periods=lookahead_periods,
                ma_window=ma_window,
                hl_avg_col=hl_avg_col,
                ma_col=ma_col,
                returns_col=returns_col,
                shift_in=shift_in,
                rownumber=self.params.get("mp_data_rownumber"),
                create_label=True
            )

        df_ticks = process_labels(df_ticks, "T1_Bid_Price", "T1_Ask_Price", "T1_Bid_Price", 1)
        df_rates = process_labels(df_rates, "R1_Bid_Price", "R1_Ask_Price", "R1_Close", 2)
        df_file_ticks = process_labels(df_file_ticks, "T2_Bid_Price", "T2_Ask_Price", "T2_Bid_Price", 3)
        df_file_rates = process_labels(df_file_rates, "R2_Bid_Price", "R2_Ask_Price", "R2_Close", 4)
        
        return df_ticks, df_rates, df_file_ticks, df_file_rates
