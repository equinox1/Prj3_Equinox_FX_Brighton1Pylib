#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
# packages dependencies for this module
from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
#+-------------------------------------------------------------------
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
#--------------------------------------------------------------------
# create class  "CMqldatasetup"
# usage: mql data services
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqldatasetuplegacy:
    def __init__(self, **kwargs):
        self.data = kwargs.get('data', None)
        self.mv_loadapiticks = kwargs.get('mv_loadApi', False)
        self.mv_loadapirates = kwargs.get('mv_loadApiRates', False)
        self.mv_loadFile = kwargs.get('mv_loadFile', False)
        self.mp_dfName = kwargs.get('mp_dfName', None)
        self.mv_utc_from = kwargs.get('mv_utc_from', None)
        self.mp_symbol_primary = kwargs.get('mp_symbol_primary', None)
        self.mp_rows = kwargs.get('mp_rows', None)
        self.mp_rowcount = kwargs.get('mp_rowcount', None)
        self.mp_command = kwargs.get('mp_command', None)
        self.mp_path = kwargs.get('mp_path', None)
        self.mp_filename = kwargs.get('mp_filename', None)
        self.mp_seconds = kwargs.get('mp_seconds', 60)
        self.mp_unit = kwargs.get('mp_unit', 's')
        self.lp_year = kwargs.get('lp_year', '2023')
        self.lp_month = kwargs.get('lp_month', '01')
        self.lp_day = kwargs.get('lp_day', '01')
        self.lp_timezone = kwargs.get('lp_timezone', 'etc/utc')
        self.lp_rates1 = kwargs.get('lp_rates1','lp_rates1')
        self.lp_utc_from = kwargs.get('lp_utc_from', '2023-01-01 00:00:00+00:00')
        self.lp_symbol = kwargs.get('lp_symbol', 'EURUSD')
        self.lp_rows = kwargs.get('lp_rows', 1000)
        self.lp_rowcount = kwargs.get('lp_rowcount', 1000)
        self.lp_flag = kwargs.get('lp_flag', 'mt5.COPY_TICKS_ALL')
        self.lp_path = kwargs.get('lp_path', '.')
        self.lp_filename = kwargs.get('lp_filename', 'tickdata1')
        self.lp_seconds = kwargs.get('lp_seconds', None)
        self.lp_timeframe = kwargs.get('lp_timeframe', 'mt5.TIMEFRAME_M1')
        self.lp_run = kwargs.get('lp_run', 1)
        self.lp_features = kwargs.get('lp_features', 'Close')
        self.lp_label = kwargs.get('lp_label', 'Label')
        self.lp_arch = kwargs.get('lp_arch', 'x86_64')
        self.lp_os = kwargs.get('lp_os', 'win64') # win64, linux64, macos
        

       
    # create method  "setmql_timezone()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def set_mql_timezone(self, lp_year, lp_month, lp_day, lp_timezone):
        lv_timezone = pytz.timezone(lp_timezone)  # Set the timezone
        native_dt = datetime(lp_year, lp_month, lp_day)  # Create a native datetime object
        return lv_timezone.localize(native_dt)

    # create method  "run_load_from_mql()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_load_from_mql(self,lp_os, lp_loadapiticks, lp_loadapirates, lp_loadfileticks, lp_loadfilerates, lp_rates1, lp_rates2, lp_utc_from, lp_symbol, lp_rows, lp_rowcount, lp_command_ticks,lp_command_rates, lp_path, lp_filename1, lp_filename2, lp_timeframe):
        if lp_os == 'windows': # windows ,linux, macos
            import MetaTrader5 as mt5

        #Reset the dataframes
        lp_rates1 = pd.DataFrame()
        lp_rates2 = pd.DataFrame()
        lp_rates3 = pd.DataFrame()
        lp_rates4 = pd.DataFrame()

        print("mp_unit", self.mp_unit, "mp_seconds", self.mp_seconds)
        if lp_loadapiticks and self.os == 'windows':
            try:
                print("Running Tick load from Mql")
                print("===========================")
                print("lp_symbol", lp_symbol)
                print("lp_utc_from", lp_utc_from)
                print("lp_rows", lp_rows)
                print("lp_command", lp_command_ticks)

                lp_rates1 = mt5.copy_ticks_from(lp_symbol, lp_utc_from, lp_rows, lp_command_ticks)
                lp_rates1 = pd.DataFrame(lp_rates1)
                
                if lp_rates1.empty:
                    print("1:No tick data found")  
                else:
                    print("Api tick data received:", len(lp_rates1))
            except Exception as e:
                print(f"Mt5 api ticks exception: {e}")
    

        if lp_loadapirates  and lp_os == 'windows':
            try:
                print("Running Rates load from Mql")
                print("===========================")
                print("lp_symbol", lp_symbol)
                print("lp_timeframe", lp_timeframe)
                print("lp_utc_from", lp_utc_from)
                print("lp_rows", lp_rows)
                
                lp_rates2 = mt5.copy_rates_from(lp_symbol, eval(lp_timeframe), lp_utc_from, lp_rows)
                lp_rates2 = pd.DataFrame(lp_rates2)
                
                if lp_rates2.empty:
                    print("1:No rate data found")  
                else:
                    print("Api rates data received:", len(lp_rates2))   
            except Exception as e:
                print(f"Mt5 api rates exception: {e}")

                

        if lp_loadfileticks:    
            lpmergepath = lp_path + "//" + lp_filename1
            try:
                lp_rates3 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount,low_memory=False)
               
                if lp_rates3.empty:
                    print("1:No tick data found")
                else:
                    print("File tick data received:", len(lp_rates3))
            except Exception as e:
                print(f"Fileload Tick exception: {e}")

        if lp_loadfilerates:    
            lpmergepath = lp_path + "//" + lp_filename2
            
            try:
                lp_rates4 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount, low_memory=False)
                lp_rates4.drop('vol3', axis=1, inplace=True)
                if lp_rates4.empty:
                    print("1:No rate data found")
                else:
                    print("File rate data received:", len(lp_rates4))
            except Exception as e:
                print(f"Fileload rates exception: {e}")
                

        return lp_rates1 , lp_rates2, lp_rates3, lp_rates4


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
                    print(f"Merging {mp_filesrc} {col1} and {col2} to {mcol}")
                    df[mcol] = pd.to_datetime(df[col1].dt.strftime('%Y-%m-%d') + ' ' + df[col2].dt.strftime('%H:%M:%S.%f'), format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
                    df.drop([col1, col2], axis=1, inplace=True)
                    # Reorder columns
                    print("MDatetimeColumns: mcol",mcol,"col1",col1,"col2",col2,"filesrc",mp_filesrc)
                    datetime_col = mcol if mcol in df.columns else df.columns[0]
                    print("Reordering columns with datetime first mcol:", mcol)
                    df = df[[datetime_col] + [col for col in df.columns if col != datetime_col]]
                    print("Reordered columns with datetime first")
                    print("Columns:", df.columns)
                except Exception as e:
                    print(f"Error merging {mp_filesrc} {col1} and {col2} to {mcol}: {e}")
            return df

        def resort_columns(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc) -> pd.DataFrame:
            if mcol in df.columns:
                datetime_col = mcol if mcol in df.columns else df.columns[0]
                df = df[[datetime_col] + [col for col in df.columns if col != datetime_col]]
                print("Reordered columns with datetime first")
                print("Columns:", df.columns)
            return df

        def convert_datetime(df: pd.DataFrame, column: str, fmt: str = None, unit: str = None, type: str = None,dcol1 = None, dcol2 = None, dcol3 = None, dcol4= None,dcol5 = None) -> None:
            try:
                if type == 'a':
                    print(f"Converting:a {mp_filesrc} {column} to datetime with stfttime hours string: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%H:%M:%S.%f'), format='%H:%M:%S.%f', errors='coerce', utc=True)
                elif type == 'b':
                    print(f"Converting:b {mp_filesrc} {column} to datetime with tf date model: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df.pop(column), format=fmt, errors='coerce')
                elif type == 'c':
                    print(f"Converting:c {mp_filesrc} {column} to datetime with stfttime years string: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                    df[column] = pd.to_datetime(df[column].dt.strftime('%d/%m/%y %H:%M:%S.%f'), format='%d/%m/%y %H:%M:%S.%f', errors='coerce', utc=True)
                elif type == 'd':
                    print(f"Converting:d {mp_filesrc} {column} to datetime with tf time: type {type} and format {fmt}")
                    df[column] = df[column].map(pd.Timestamp.timestamp)
                elif type == 'e':
                    print(f"Converting:e {mp_filesrc} {column} to datetime with format {fmt}: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce', utc=True)
                elif type == 'f':
                    print(f"Converting:f {mp_filesrc} {column} to datetime with unit {unit}: type {type} and format {fmt}")
                    df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce', utc=True)
                elif type == 'g':
                    print(f"Dropping:g {mp_filesrc} {column} with type {type}")
                    print("Columns to drop:", dcol1, dcol2, dcol3, dcol4,dcol5)
                    print("Columns:", df.columns)
                    if column in df.columns:
                        print(f"Dropping column: {column} found in ", df.columns)
                        columns_to_drop = [col for col in [dcol1, dcol2, dcol3, dcol4,dcol5] if col in df.columns]
                        if columns_to_drop:
                            print(f"Dropping columns: {columns_to_drop} from", df.columns)
                            df.drop(columns_to_drop, axis=1, inplace=True)
            except Exception as e:
                print(f"Error converting {mp_filesrc} {column} {type}: {e}")

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
            print(f"Processing {mp_filesrc} data")
            
            if mp_filesrc in date_columns:
                column, fmt, unit, type = date_columns[mp_filesrc]
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)

            if mp_filesrc in time_columns:
                column, fmt, unit, type = time_columns[mp_filesrc]
                print("Columns Time:", column, "Format:", fmt, "Unit:", unit, "Type:", type, "Filesrc:", mp_filesrc)
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
                print("Columns Drop:", column, "Format:", fmt, "Unit:", unit, "Type:", type, "Filesrc:", mp_filesrc, "Columns:", col1, col2, col3, col4, col5)
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type,dcol1=col1,dcol2=col2,dcol3=col3,dcol4=col4,dcol5=col5)
                
            if filter_int:
                for col in df.select_dtypes(include=['int64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print("Columns Numeric:", col)
            if filter_flt:
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print("Columns Numeric:", col)
            if filter_obj:
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print("Columns Object:", col)
            if filter_dtmi:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('int64'))
                    print("Columns DateTime:", col)
            if filter_dtmf:
                for col in df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64']).columns:
                    df[col] = pd.to_numeric(df[col].view('float64'))
                    print("Columns DateTime:", col)
            if mp_dropna:
                df.fillna(0, inplace=True)
                print("Dropped NaN values")
            
             # final sort
            if mp_filesrc in merge_columns and mp_merge:
                col1, col2, mcol, mfmt1, mfmt2 = merge_columns[mp_filesrc]
                df = resort_columns(df, col1, col2, mcol, mfmt1, mfmt2, mp_filesrc) 
        return df

    def calculate_moving_average(self,df, column, window,min_periods=1):
        print("Calculating moving average for column:", column, " with window:", window)
        # Calculate Moving Averages
        df['SMA'] = df[column].rolling(window=min_periods).mean()  # 14-day Simple Moving Average
        df["SMA"] = df["SMA"].fillna(method="bfill")  # or "ffill"
        df = df.dropna(subset=["SMA"])
        return df['SMA']

    def calculate_log_returns(self, df, column, shift):
        print("Calculating log returns for column:", column, "with shift:", shift)
        # Check for valid values
        if df[column].isna().sum() > 0:
            print("Warning: Column contains NaN values. Filling missing values.")
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
        print("lpDf", df.tail(10))
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


    #Time series data TEMPORIAN
    # create method  "run_load_from_mql()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def temporian_load_from_mql(self, lp_loadapiticks, lp_loadapirates, lp_loadfileticks, lp_loadfilerates, lp_rates1, lp_rates2, lp_utc_from, lp_symbol, lp_rows, lp_rowcount, lp_command, lp_path, lp_filename1, lp_filename2, lp_timeframe):
        
        #Reset the dataframes
        lp_rates1 = pd.DataFrame()
        lp_rates2 = pd.DataFrame()
        lp_rates3 = pd.DataFrame()
        lp_rates4 = pd.DataFrame()
        print("mp_unit", self.mp_unit, "mp_seconds", self.mp_seconds)
        if lp_loadapiticks:
            try:
                print("Running Tick load from Mql")
                print("===========================")
                print("lp_utc_from", lp_utc_from)
                print("lp_rows", lp_rows)
                print("lp_symbol", lp_symbol)
                print("lp_command", lp_command)
                lp_rates1 = mt5.copy_ticks_from(lp_symbol, lp_utc_from, lp_rows, lp_command)
                lp_rates1 = pd.DataFrame(lp_rates1)
                
                if lp_rates1.empty:
                    print("1:No tick data found")  
                else:
                    print("Api tick data received:", len(lp_rates1))
            except Exception as e:
                print(f"Mt5 api ticks exception: {e}")

        if lp_loadapirates:
            try:
                print("Running Rates load from Mql")
                print("===========================")
                print("lp_symbol", lp_symbol)
                print("lp_timeframe", lp_timeframe)
                print("lp_utc_from", lp_utc_from)
                print("lp_rows", lp_rows)
                lp_rates2 = mt5.copy_rates_from(lp_symbol,lp_timeframe ,lp_utc_from, lp_rows)
                lp_rates2 = pd.DataFrame(lp_rates2)
                
                if lp_rates2.empty:
                    print("1:No rate data found")  
                else:
                    print("Api rates data received:", len(lp_rates2))   
            except Exception as e:
                print(f"Mt5 api rates exception: {e}")

        if lp_loadfileticks:    
            lpmergepath = lp_path + "//" + lp_filename1
            try:
                lp_rates3 = tp.from_csv(lpmergepath)
               
                if lp_rates3.empty:
                    print("1:No tick data found")
                else:
                    print("File tick data received:", len(lp_rates3))
            except Exception as e:
                print(f"Fileload Tick exception: {e}")

        if lp_loadfilerates:    
            lpmergepath = lp_path + "//" + lp_filename2
            
            try:
                lp_rates4 = tp.from_csv(lpmergepath)
                lp_rates4.drop('vol3', axis=1, inplace=True)
                if lp_rates4.empty:
                    print("1:No rate data found")
                else:
                    print("File rate data received:", len(lp_rates4))
            except Exception as e:
                print(f"Fileload rates exception: {e}")

        return lp_rates1 , lp_rates2, lp_rates3, lp_rates4


    # create method  "run_mql_print".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var    
    def run_mql_print(self, df, hrows,colwidth,tablefmt = "pretty",floatfmt = ".5f",numalign = "left",stralign = "left"):
        print("Start First few rows of the data: Count", len(df))
        
        # Wrap long text in columns
        def wrap_column_data(column, width):
            return column.apply(lambda x: '\n'.join(textwrap.wrap(str(x), width)))
        
        df = df.apply(lambda col: wrap_column_data(col, colwidth))  # Adjust width as needed

        print(tabulate(
            df.head(hrows), 
            showindex=False,
            headers=df.columns,
            tablefmt="pretty", # "plain", "grid", "pipe", "orgtbl", "jira", "presto", "pretty", "psql", "rst", "mediawiki", "moinmoin", "youtrack", "html", "unsafehtml", "latex", "latex_raw", "latex_booktabs", "textile"
            numalign="left",
            stralign="left", 
            maxcolwidths=[colwidth] * len(df.columns),  # Limit column widths to 10
            floatfmt=".5f"
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