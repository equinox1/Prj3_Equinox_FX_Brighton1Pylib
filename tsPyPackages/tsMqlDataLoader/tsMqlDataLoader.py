#+------------------------------------------------------------------+
#|                                                    tsmqlmoobj.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

# packages dependencies for this module
#

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
# create class  "DataLoader"
# usage: mql data services
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CDataLoader:
    """Class to store and manage market data parameters."""
    
    def __init__(self, **kwargs):
        self.api_ticks= kwargs.get('api_ticks', True)
        self.api_rates= kwargs.get('api_rates', True)
        self.file_ticks= kwargs.get('file_ticks', True)
        self.file_rates= kwargs.get('file_rates', True)
        self.dfname1= kwargs.get('dfname1', "df_rates1")
        self.dfname2= kwargs.get('dfname2', "df_rates2")
        self.utc_from= kwargs.get('utc_from', None)
        self.symbol_primary= kwargs.get('symbol_primary', None)
        self.rows= kwargs.get('rows', 1000)
        self.rowcount= kwargs.get('rowcount', 10000)
                
        self.data_path= kwargs.get('data_path', None)
        self.file_value1= kwargs.get('file_value1', None)
        self.file_value2= kwargs.get('file_value2', None)
        self.timeframe= kwargs.get('timeframe', None)
        self.os = kwargs.get('os', 'windows') # windows or linux or macos

        if self.os == 'windows':
            import MetaTrader5 as mt5
            self.command_ticks= kwargs.get('command_ticks', mt5.COPY_TICKS_ALL)
            self.command_rates= kwargs.get('command_rates', None)

            self.api_params = {
                "ticks": self.api_ticks,
                "rates": self.api_rates
            }
                
            self.file_params = {
                "ticks": self.file_ticks,
                "rates": self.file_rates
            }
                
            self.data_params = {
                "dfname1": self.dfname1,
                "dfname2": self.dfname2,
                "utc_from": self.utc_from,
                "symbol_primary": self.symbol_primary,
                "rows": self.rows,
                "rowcount": self.rowcount
            }
                
            self.command_params = {
                "ticks": self.command_ticks,
                "rates": self.command_rates
                }
                
            self.file_settings = {
                "data_path": self.data_path,
                "file_value1": self.file_value1,
                "file_value2": self.file_value2
            }
                
            timeframe = self.timeframe
                

    def get_params(self):
        return {
            "loadapiticks": self.api_params["ticks"],
            "loadapirates": self.api_params["rates"],
            "loadfileticks": self.file_params["ticks"],
            "loadfilerates": self.file_params["rates"],
            "rates1": None,  
            "rates2": None,
            "utc_from": self.data_params["utc_from"],
            "symbol": self.data_params["symbol_primary"],
            "rows": self.data_params["rows"],
            "rowcount": self.data_params["rowcount"],
            "command_ticks": self.command_params["ticks"],
            "command_rates": self.command_params["rates"],
            "path": self.file_settings["data_path"],
            "filename1": self.file_settings["file_value1"],
            "filename2": self.file_settings["file_value2"],
            "timeframe": self.timeframe,
            "command_rates": self.command_params["rates"],
            "command_ticks":self.command_params["ticks"]
        }


    def load_market_data(self,obj, params_obj):
        """
        Function to load market data using parameters from DataLoader.
        
        Parameters:
            obj (object): The instance that contains the 'run_load_from_mql' method.
            params_obj (DataLoader): Instance of DataLoader class.

        Returns:
            tuple: API ticks, API rates, file ticks, and file rates.
        """
        
        return self.load_data_from_mql(**params_obj.get_params())


    # create method  "load_data_from_mql".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var  
    #  def load_data_from_mql(self, loadapiticks, loadapirates, loadfileticks, loadfilerates, rates1, rates2, utc_from, symbol, rows, rowcount, command_ticks,command_rates, path, filename1, filename2, timeframe):
    def load_data_from_mql(self, **kwargs):
        loadapiticks = kwargs.get("loadapiticks")
        loadapirates = kwargs.get("loadapirates")
        loadfileticks = kwargs.get("loadfileticks")
        loadfilerates = kwargs.get("loadfilerates")
        rates1 = kwargs.get("rates1")
        rates2 = kwargs.get("rates2")
        utc_from = kwargs.get("utc_from")
        symbol = kwargs.get("symbol")
        rows = kwargs.get("rows")
        rowcount = kwargs.get("rowcount")
        command_ticks = kwargs.get("command_ticks")
        command_rates = kwargs.get("command_rates")
        path = kwargs.get("path")
        filename1 = kwargs.get("filename1")
        filename2 = kwargs.get("filename2")
        timeframe = kwargs.get("timeframe")
       

        #Reset the dataframes
        rates1 = pd.DataFrame()
        rates2 = pd.DataFrame()
        rates3 = pd.DataFrame()
        rates4 = pd.DataFrame()

        if self.os == 'windows':
            import MetaTrader5 as mt5

        if loadapiticks and self.os == 'windows':
            try:
                print("Running Tick load from Mql")
                print("===========================")
                print("symbol", symbol)
                print("utc_from", utc_from)
                print("rows", rows)
                print("command", command_ticks)
                
                rates1 = mt5.copy_ticks_from(symbol, utc_from, rows,command_ticks)
                rates1 = pd.DataFrame(rates1)
                
                if rates1.empty:
                    print("1:No tick data found")  
                else:
                    print("Api tick data received:", len(rates1))
            except Exception as e:
                print(f"Mt5 api ticks exception: {e}")
    

        if loadapirates and self.os == 'windows':
            try:
                print("Running Rates load from Mql")
                print("===========================")
                print("symbol", symbol)
                print("timeframe", timeframe)
                print("utc_from", utc_from)
                print("rows", rows)
                
                rates2 = mt5.copy_rates_from(symbol, eval(timeframe), utc_from, rows)
                rates2 = pd.DataFrame(rates2)
                
                if rates2.empty:
                    print("1:No rate data found")  
                else:
                    print("Api rates data received:", len(rates2))   
            except Exception as e:
                print(f"Mt5 api rates exception: {e}")

                

        if loadfileticks:    
            lpmergepath = path + "//" + filename1
            try:
                rates3 = pd.read_csv(lpmergepath, sep=',', nrows=rowcount,low_memory=False)
               
                if rates3.empty:
                    print("1:No tick data found")
                else:
                    print("File tick data received:", len(rates3))
            except Exception as e:
                print(f"Fileload Tick exception: {e}")

        if loadfilerates:    
            lpmergepath = path + "//" + filename2
            
            try:
                rates4 = pd.read_csv(lpmergepath, sep=',', nrows=rowcount, low_memory=False)
                rates4.drop('vol3', axis=1, inplace=True)
                if rates4.empty:
                    print("1:No rate data found")
                else:
                    print("File rate data received:", len(rates4))
            except Exception as e:
                print(f"Fileload rates exception: {e}")
                

        return rates1 , rates2, rates3, rates4


    # create method  "setmql_timezone()".
    # class: cmqldatasetup      
    # usage: mql data
    # /param  var                          
    def set_mql_timezone(self, year, month, day, timezone):
        lv_timezone = pytz.timezone(timezone)  # Set the timezone
        native_dt = datetime(year, month, day)  # Create a native datetime object
        return lv_timezone.localize(native_dt)

