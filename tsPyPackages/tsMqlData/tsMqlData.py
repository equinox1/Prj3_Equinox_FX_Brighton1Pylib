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
    def __init__(self):
        pass

    # create method  "setmql_timezone()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def set_mql_timezone(self, lp_year=2024, lp_month=1, lp_day=1, lp_timezone="etc/UTC"):
        lv_timezone = pytz.timezone(lp_timezone)  # Set the timezone
        native_dt = datetime(lp_year, lp_month, lp_day)  # Create a native datetime object
        return lv_timezone.localize(native_dt)

    # create method  "run_load_from_mql()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_load_from_mql(self, lp_manual, lp_rates1="rates", lp_utc_from="2023-01-01 00:00:00+00:00", lp_symbol="JPYUSD", lp_rows=1000, lp_rowcount=1000, lp_flag=mt5.COPY_TICKS_ALL, lp_path=".", lp_filename="tickdata"):
        # request 100 000 eurusd ticks starting from lp_year, lp_month, lp_day in utc time zone
        if lp_manual:
            print("lp_manual:", lp_manual)
            print("lp_rates1:", lp_rates1)
            print("lp_utc_from:", lp_utc_from)
            print("lp_symbol:", lp_symbol)
            print("lp_rows:", lp_rows)
            print("lp_flag:", lp_flag)

        lp_rates1 = pd.DataFrame()
        lp_rates1 = lp_rates1.drop(index=lp_rates1.index)

        print("mt5 version:", mt5.version())
        print("mt5 info:", mt5.terminal_info())

        try:
            lp_rates1 = mt5.copy_ticks_from(lp_symbol, lp_utc_from, lp_rows, lp_flag)

            if lp_rates1 is None or len(lp_rates1) == 0:
                print("1:No tick data found")
            elif lp_rates1 is None or len(lp_rates1) == 0 and lp_manual:
                # Convert to DataFrame
                lp_rates1 = pd.DataFrame(lp_rates1)
                print("2:lp_rates1 loaded via api:", lp_rates1.head())
            else:
                lpmergepath = lp_path + "//" + lp_filename
                print("3:Manual load_filename:", lpmergepath)
                lp_rates1 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount)
                print("3:ticks received:", len(lp_rates1))
                lp_rates1 = lp_rates1.rename(columns={'Date': 'time'})
                lp_rates1 = lp_rates1.rename(columns={'Timestamp': 'time_msc'})
                lp_rates1 = lp_rates1.rename(columns={'Bid Price': 'bid'})
                lp_rates1 = lp_rates1.rename(columns={'Ask Price': 'ask'})
                lp_rates1 = lp_rates1.rename(columns={'Last Price': 'close'})
                lp_rates1 = lp_rates1.rename(columns={'Volume': 'volume'})
                lp_rates1 = lp_rates1[['time', 'bid', 'ask', 'close', 'time_msc', 'volume']]

        except Exception as e:
            e = mt5.last_error()
            print(f"Mt5 result: {e}")
        return lp_rates1

    # create method  "run_shift_data1()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def run_shift_data1(self, lp_df, lp_seconds=60, lp_unit='s'):
        lv_seconds = lp_seconds
        lv_number_of_rows = lv_seconds
        lp_df.style.set_properties(**{'text-align': 'left'})
        lp_df['time'] = pd.to_datetime(lp_df['time'], unit=lp_unit)
        lp_df['close'] = (lp_df['ask'] + lp_df['bid']) / 2
        lv_empty_rows = pd.DataFrame(np.nan, index=range(lv_number_of_rows), columns=lp_df.columns)
        lp_df = lp_df._append(lv_empty_rows, ignore_index=True)
        lp_df['target'] = lp_df['close'].shift(-lv_seconds)
        lp_df = lp_df.dropna()
        lp_df.style.set_properties(**{'text-align': 'left'})
        print("lpDf", lp_df.tail(10))
        return lp_df

    # create method  "create_dataset()".
    # class: cmqldatasetup      
    # usage: # Creating time steps for LSTM input
    # /param  var
    def create_dataset(self, data, lp_seconds=60):
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
        
   
    
