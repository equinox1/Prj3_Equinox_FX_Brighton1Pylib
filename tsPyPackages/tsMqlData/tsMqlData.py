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
from datetime import datetime

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from sklearn.preprocessing import MinMaxScaler

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
                lp_rates3 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount)
               
                if lp_rates3.empty:
                    print("1:No tick data found")
                else:
                    print("File tick data received:", len(lp_rates3))
            except Exception as e:
                print(f"Fileload Tick exception: {e}")

        if lp_loadfilerates:    
            lpmergepath = lp_path + "//" + lp_filename2
            
            try:
                lp_rates4 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount)
                if lp_rates4.empty:
                    print("1:No rate data found")
                else:
                    print("File rate data received:", len(lp_rates4))
            except Exception as e:
                print(f"Fileload rates exception: {e}")

        return lp_rates1 , lp_rates2, lp_rates3, lp_rates4


    def wrangle_time(self, lp_df, lp_unit, lp_filesrc):
        def rename_columns(df, mapping):
            valid_renames = {old: new for old, new in mapping.items() if old in df.columns}
            df.rename(columns=valid_renames, inplace=True)
            print(f"Renamed columns: {valid_renames}")

        def convert_datetime(df, column,  fmt=None, unit=None):
            print(f"Converting {column} to datetime")
            if column in df.columns:
                try:
                    print(f"Converting {column} to datetime with unit {unit}")
                    df[column] = pd.to_datetime(df[column], unit=unit, errors='coerce')
                    print(df[column].head(3))  # Print the first few rows to verify conversion
                    print(f"Converting {column} to datetime with format {fmt}")
                    df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce')
                    print(df[column].head(3))  # Print the first few rows to verify conversion
                except Exception as e:
                    print(f"Error converting {column}: {e}")

        mappings = {
            'ticks1': {
                'time': 'T1_Date',
                'bid': 'T1_Bid Price',
                'ask': 'T1_Ask Price',
                'last': 'T1_Last Price',
                'volume': 'T1_Volume',
                'time_msc': 'T1_Time_Msc',
                'flags': 'T1_Flags',
                'volume_real': 'T1_Real Volume'
            },
            'rates1': {
                'time': 'R1_Date',
                'open': 'R1_Open',
                'high': 'R1_High',
                'low': 'R1_Low',
                'close': 'R1_Close',
                'tick_volume': 'R1_Tick Volume',
                'spread': 'R1_spread',
                'real_volume': 'R1_Real Volume'
            },
            'ticks2': {
                'Date': 'T2_Date',
                'Timestamp': 'T2_Timestamp',
                'Bid Price': 'T2_Bid Price',
                'Ask Price': 'T2_Ask Price',
                'Last Price': 'Last Price',
                'Volume': 'T2_Volume'
            },
            'rates2': {
                'Date': 'R2_Date',
                'Timestamp': 'R2_Timestamp',
                'Open': 'R2_Open',
                'High': 'R2_High',
                'Low': 'R2_Low',
                'Close': 'R2_Close',
                'tick_volume': 'R2_Tick Volume',
                'Volume': 'R2_Volume'
            }
        }

        date_columns = {
            'ticks1': ('time', '%Y%m%d', 's'),
            'rates1': ('time', '%Y%m%d', 's'),
            'ticks2': ('Date', '%Y%m%d', 'ms'),#Date 20240101
            'rates2': ('Date', '%Y%m%d,%H:%M:%S', 'ms'),#Date 20030505,00:00:00
        }

        time_columns = {
            'ticks1': ('time_msc', '%H:%M:%S', 'ms'),
            'ticks2': ('Timestamp', '%H:%M:%S', 's'), #Timestamp 22:00:12 
            'rates2': ('Timestamp', '%H:%M:%S', 's'), #Timestamp 17:49:00
        }

        if lp_filesrc in mappings:
            print(f"Processing {lp_filesrc} data")
            
            if lp_filesrc in date_columns:
                column, fmt, unit = date_columns[lp_filesrc]
                convert_datetime(lp_df, column,  fmt=fmt, unit=unit)
            

            if lp_filesrc in time_columns:
                column, fmt, unit = time_columns[lp_filesrc]
                convert_datetime(lp_df, column, fmt=fmt, unit=unit)
               

            rename_columns(lp_df, mappings[lp_filesrc])
            
            # Handle missing values
            lp_df.fillna(method='ffill', inplace=True)  # Forward fill
            lp_df.fillna(method='bfill', inplace=True)  # Backward fill

            # Remove duplicates
            lp_df.drop_duplicates(inplace=True)

            # Convert data types
            for col in lp_df.select_dtypes(include=['object']).columns:
                lp_df[col] = lp_df[col].astype(str)
            for col in lp_df.select_dtypes(include=['int64', 'float64']).columns:
                lp_df[col] = pd.to_numeric(lp_df[col], errors='coerce')

            # Remove outliers (example: removing rows where volume is extremely high)
            if 'volume' in lp_df.columns:
                lp_df = lp_df[lp_df['volume'] < lp_df['volume'].quantile(0.99)]

            # Standardize data (example: converting all string columns to lowercase)
            for col in lp_df.select_dtypes(include=['object']).columns:
                lp_df[col] = lp_df[col].str.lower()

            #lp_df.dropna(inplace=True)

        return lp_df


    # create method  "run_shift_data1()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_shift_data1(self, lp_df, lp_seconds, lp_unit):
        lv_seconds = lp_seconds
        lv_number_of_rows = lv_seconds
        lp_df.style.set_properties(**{'text-align': 'left'})
        lp_df['time'] = pd.to_datetime(lp_df['time'], unit=lp_unit)
        lp_df['close'] = (lp_df['ask'] + lp_df['bid']) / 2
        lv_empty_rows = pd.DataFrame(np.nan, index=range(lv_number_of_rows), columns=lp_df.columns)
        lp_df = pd.concat([lp_df, lv_empty_rows], ignore_index=True)
        lp_df['target'] = lp_df['close'].shift(-lv_seconds)
        lp_df = lp_df.dropna()
        lp_df.style.set_properties(**{'text-align': 'left'})
        print("lpDf", lp_df.tail(10))
        return lp_df

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