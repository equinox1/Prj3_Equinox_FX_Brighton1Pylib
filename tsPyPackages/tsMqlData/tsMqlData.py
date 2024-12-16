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
        self.mv_loadApi = kwargs.get('mv_loadApi', False)
        self.mv_loadFile = kwargs.get('mv_loadFile', False)
        self.mp_dfName = kwargs.get('mp_dfName', None)
        self.mv_utc_from = kwargs.get('mv_utc_from', None)
        self.mp_symbol_primary = kwargs.get('mp_symbol_primary', None)
        self.mp_rows = kwargs.get('mp_rows', None)
        self.mp_rowcount = kwargs.get('mp_rowcount', None)
        self.mp_command = kwargs.get('mp_command', None)
        self.mp_path = kwargs.get('mp_path', None)
        self.mp_filename = kwargs.get('mp_filename', None)
        self.mp_seconds = kwargs.get('mp_seconds', None)
        self.mp_unit = kwargs.get('mp_unit', None)
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
        self.svar = kwargs.get('svar', 0.0)



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
    def run_load_from_mql(self, lp_loadapi,lp_loadfile, lp_rates1,lp_rates2, lp_utc_from, lp_symbol, lp_rows, lp_rowcount, lp_command, lp_path, lp_filename):
        # request 100 000 eurusd ticks starting from lp_year, lp_month, lp_day in utc time zone
        print("lp_loadapi:", lp_loadapi)
        print("lp_loadfile:", lp_loadfile)
        print("lp_rates1:", lp_rates1)
        print("lp_utc_from:", lp_utc_from)
        print("lp_symbol:", lp_symbol)
        print("lp_rows:", lp_rows)
        print("lp_command:", lp_command)
        lp_rates1 = pd.DataFrame()
        lp_rates1 = lp_rates1.drop(index=lp_rates1.index)
        lp_rates2 = pd.DataFrame()
        lp_rates2 = lp_rates2.drop(index=lp_rates2.index)

        if lp_loadapi:
            print("mt5 version:", mt5.version())
            print("mt5 info:", mt5.terminal_info())
            try:
                lp_rates1 = mt5.copy_ticks_from(lp_symbol, lp_utc_from, lp_rows, lp_flag)
                lp_rates1 = pd.DataFrame(lp_rates1)
                print("2:lp_rates1 loaded via api:", lp_rates1.head())
                if lp_rates1 is None or len(lp_rates1) == 0:
                    print("1:No tick data found")  
            except Exception as e:
                e = mt5.last_error()
                print(f"Mt5 result: {e}")

        if lp_loadfile:    
            lpmergepath = lp_path + "//" + lp_filename
            print("3:Manual load_filename:", lpmergepath)
            
            try:
                lp_rates2 = pd.read_csv(lpmergepath, sep=',', nrows=lp_rowcount)
                print("3:ticks received:", len(lp_rates2))
            except Exception as e:
                e = mt5.last_error()
                print(f"Fileload result: {e}")
        return lp_rates1 , lp_rates2


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