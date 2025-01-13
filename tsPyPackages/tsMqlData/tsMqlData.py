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
#
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime
import arrow
import pytz
from sklearn.preprocessing import MinMaxScaler
import logging
from tabulate import tabulate
#--------------------------------------------------------------------
# create class  "CMqldatasetup"
# usage: mql data services
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMqldatasetup:
    def __init__(self, **kwargs):
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
       
       
    logging.basicConfig(level=logging.INFO)
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
    def run_load_from_mql(self, lp_loadapiticks, lp_loadapirates, lp_loadfileticks, lp_loadfilerates, lp_rates1, lp_rates2, lp_utc_from, lp_symbol, lp_rows, lp_rowcount, lp_command, lp_path, lp_filename1, lp_filename2, lp_timeframe):
        
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


    def wrangle_time(self, df: pd.DataFrame, lp_unit: str, mp_filesrc: str, filter_int: bool, filter_flt: bool, filter_obj: bool, filter_dtmi: bool, filter_dtmf: bool, mp_dropna: bool,mp_merge: bool, mp_convert: bool):
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
       
        def merge_datetime(df, col1, col2, mcol, mfmt1,mfmt2,mp_filesrc): 
            if col1 in df.columns and col2 in df.columns:
                try:
                    print(f"Merging {mp_filesrc} {col1} and {col2} to {mcol}")
                    df[mcol] = pd.to_datetime(df[col1].dt.strftime('%Y-%m-%d') + ' ' + df[col2].dt.strftime('%H:%M:%S.%f'), format='%Y-%m-%d %H:%M:%S.%f', errors='coerce', utc=True)
                    print("Sort Merged datetime columns")
                    # Reorder columns
                    print("Sorting merged datetime column")
                    df = df[[mcol] + [col for col in df.columns if col not in {col1, col2, mcol}]]
                    print("Columns:", df.columns)
                except Exception as e:
                    print(f"Error merging {mp_filesrc} {col1} and {col2} to {mcol}: {e}")
            return df


        def convert_datetime(df: pd.DataFrame, column: str, fmt: str = None, unit: str = None, type: str = None) -> None:
           # print("Converting: datetime columns:", column," with format:", fmt, " and unit:", unit, " and type:",type)
            if column in df.columns:
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
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)

            # Rename columns
            rename_columns(df, mappings[mp_filesrc])
                
            # Merge datetime columns
            if mp_filesrc in merge_columns and mp_merge:
                col1, col2, mcol,mfmt1,mfmt2 = merge_columns[mp_filesrc]
                df = merge_datetime(df, col1, col2, mcol,mfmt1,mfmt2, mp_filesrc)

            # Convert datetime columns wiyh tf
            if mp_filesrc in conv_columns and mp_convert:
                column, fmt, unit, type = conv_columns[mp_filesrc]
                convert_datetime(df, column, fmt=fmt, unit=unit, type=type)
                
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
            
        return df



    def calculate_moving_average(self,df, column, window,min_periods=1):
        print("Calculating moving average for column:", column, " with window:", window)
        # Calculate Moving Averages
        df['SMA'] = df[column].rolling(window=min_periods).mean()  # 14-day Simple Moving Average
        df["SMA"] = df["SMA"].fillna(method="bfill")  # or "ffill"
        df = df.dropna(subset=["SMA"])

        #df['EMA'] = df[column].ewm(span=min_periods).mean()        # Exponential Moving Average with span=14
        #df['CMA'] = df[column].expanding().mean()        # Cumulative Moving Average 

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


    def create_target(
        self,
        df,
        lookahead_periods,
        ma_window,
        bid_column,
        ask_column,
        column_in=None,
        column_out1='close',
        column_out2='target',
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
        run_future_returns=False
    ):
        """
        Creates a target column in the DataFrame by calculating mid prices or shifting a specified column.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing market data.
            lookahead_periods (int): Number of seconds to shift for the target.
            ma_window (int): Number of periods for the moving average.
            bid_column (str): Name of the column with bid prices.
            ask_column (str): Name of the column with ask prices.
            open_column (str): Name of the column with open prices.
            high_column (str): Name of the column with high prices.
            low_column (str): Name of the column with low prices.
            close_column (str): Name of the column with close prices.
            column_in (str, optional): Column to use for mid-price calculation (optional).
            column_out1 (str): Name of the output column for the close price (default: 'close').
            column_out2 (str): Name of the output column for the target (default: 'target').
            hl_avg_col (str): Column name for high-low average (default: 'HLAvg').
            ma_col (str): Column name for moving average (default: 'MA').
            returns_col (str): Column name for returns (default: 'Returns').
            shift_in (int): Number of periods to shift for returns calculation (default: 1).
            run_mode (int): Specifies the operation mode (1, 2, 3, or 4).
            run_avg (bool): Whether to calculate the average of bid and ask prices or high and low prices.
            run_ma (bool): Whether to calculate the moving average of the input column.
            log_stationary (bool): Whether to calculate log returns for stationarity.
            run_returns (bool): Whether to calculate returns based on the input column.
            run_future_returns (bool): Whether to calculate future returns based on the input column.

        Returns:
            pd.DataFrame: DataFrame with the target column added.

        Raises:
            ValueError: If `column_in` is not provided for run modes 1/3 or 2/4.
            ValueError: If `run_mode` is not in {1, 2, 3, 4}.
        """

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input `data` must be a pandas DataFrame.")
        if not isinstance(lookahead_periods, int) or lookahead_periods <= 0:
            raise ValueError("The `lookahead_periods` must be a positive integer.")

        if run_mode not in {1, 2, 3, 4}:
            raise ValueError("`run_mode` must be one of {1, 2, 3, 4}.")

        if run_mode in {1, 3} and column_in is None:
            raise ValueError("`column_in` must be provided for run modes 1 or 3.")

        if run_mode in {2, 4} and column_in is None:
            raise ValueError("`column_in` must be provided for run modes 2 or 4.")

        # Calculate base column based on run mode
        if run_mode in {1, 3}:  # Bid-ask average
            if run_avg:
                df[column_out1] =(df[bid_column] + df[ask_column]) / 2
                df[hl_avg_col] = (df[bid_column] + df[ask_column]) / 2
                logging.info("Bid-ask average calculated.")
            else:
                df[column_out1] = (df[bid_column] + df[ask_column]) / 2

        elif run_mode in {2, 4}:  # High-low average
            if run_avg:
                df[column_out1] = (df[high_column] + df[low_column]) / 2
                df[hl_avg_col] = (df[high_column] + df[low_column]) / 2
                logging.info("High-low average calculated.")
            else:
                df[column_out1] = df[close_column]

        # Apply moving average if required
        if run_ma:
            df['SMA']= self.calculate_moving_average(df, hl_avg_col, ma_window, min_periods=14)
            logging.info("Moving averages calculated: SMA")

        # Apply log stationary transformation if required
        if log_stationary:
            df[ma_col] = self.calculate_log_returns(df, ma_col, 1)
            logging.info("Log stationary transformation applied.")

        # Calculate returns if required
        if run_returns:
            df[returns_col] = self.calculate_log_returns(df, ma_col, shift_in)
            logging.info("Returns calculated.")

        # Calculate future returns if required
        if run_future_returns:
            df[ma_col] = df[ma_col].mul(np.exp(df[returns_col].shift(-1))).shift(1)
            logging.info("Future Returns calculated.")

        

        # Set output columns
        df[column_out1] = df[column_out1] 
        df[column_out2] = df[column_in].shift(-lookahead_periods)
        logging.info("Target column created.")

        return df



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
        df['target'] = df['close'].shift(-lv_seconds)
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


    def run_mql_print(self, df, hrows):
        print("Start First few rows of the  data:Count",len(df))
        print(tabulate(df.head(hrows), showindex=False, headers=df.columns, tablefmt="pretty", numalign="left", stralign="left", floatfmt=".4f"))