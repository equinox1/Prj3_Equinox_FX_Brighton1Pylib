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
import MetaTrader5 as mt5
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
# create class  "MarketDataParams"
# usage: mql data services
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
class CMMarketDataParams:
    """Class to store and manage market data parameters."""
    
    def __init__(self, **kwargs):
                self.api_ticks= kwargs.get('api_ticks', True),
                self.api_rates= kwargs.get('api_rates', True),
                self.file_ticks= kwargs.get('file_ticks', True),
                self.file_rates= kwargs.get('file_rates', True),
                self.dfname1= kwargs.get('dfname1', "df_rates1"),
                self.dfname2= kwargs.get('dfname2', "df_rates2"),
                self.utc_from= kwargs.get('utc_from', None),
                self.symbol_primary= kwargs.get('symbol_primary', None),
                self.rows= kwargs.get('rows', 1000),
                self.rowcount= kwargs.get('rowcount', 10000),
                self.command_ticks= kwargs.get('command_ticks', mt5.COPY_TICKS_ALL),
                self.command_rates= kwargs.get('command_rates', None),
                self.data_path= kwargs.get('data_path', None),
                self.file_value1= kwargs.get('file_value1', None),
                self.file_value2= kwargs.get('file_value2', None),
                self.timeframe= kwargs.get('timeframe', None)

                self.api_params = {
                    "ticks": api_ticks,
                    "rates": api_rates
                }
                
                self.file_params = {
                    "ticks": file_ticks,
                    "rates": file_rates
                }
                
                self.data_params = {
                    "dfname1": dfname1,
                    "dfname2": dfname2,
                    "utc_from": utc_from,
                    "symbol_primary": symbol_primary,
                    "rows": rows,
                    "rowcount": rowcount
                }
                
                self.command_params = {
                    "ticks": command_ticks,
                    "rates": command_rates
                }
                
                self.file_settings = {
                    "data_path": data_path,
                    "file_value1": file_value1,
                    "file_value2": file_value2
                }
                
                self.timeframe = timeframe
                

    def get_params(self):
        """Return all parameters as a dictionary for function call."""
        return {
            **self.api_params,
            **self.file_params,
            **self.data_params,
            **self.command_params,
            **self.file_settings,
            "timeframe": self.timeframe
        }


    def load_market_data(self,obj, params_obj):
        """
        Function to load market data using parameters from MarketDataParams.
        
        Parameters:
            obj (object): The instance that contains the 'run_load_from_mql' method.
            params_obj (MarketDataParams): Instance of MarketDataParams class.

        Returns:
            tuple: API ticks, API rates, file ticks, and file rates.
        """
        return self.run_load_data_from_mql(**params_obj.get_params())


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
    #  def load_data_from_mql(self, lp_loadapiticks, lp_loadapirates, lp_loadfileticks, lp_loadfilerates, lp_rates1, lp_rates2, lp_utc_from, lp_symbol, lp_rows, lp_rowcount, lp_command_ticks,lp_command_rates, lp_path, lp_filename1, lp_filename2, lp_timeframe):
                         
    def load_data_from_mql(self, lp_obj1,lp_obj2):
        
        lp_loadapiticks = self.lp_loadapiticks
        lp_loadapirates = self.lp_loadapirates
        lp_loadfileticks = self.lp_loadfileticks
        lp_loadfilerates = self.lp_loadfilerates
        lp_rates1 = self.lp_rates1
        lp_rates2 = self.lp_rates2
        lp_utc_from = self.lp_utc_from
        lp_symbol = self.lp_symbol
        lp_rows = self.lp_rows
        lp_rowcount = self.lp_rowcount
        lp_command_ticks = self.lp_command_ticks
        lp_command_rates = self.lp_command_rates
        lp_path = self.lp_path
        lp_filename1 = self.lp_filename1
        lp_filename2 = self.lp_filename2
        lp_timeframe = self.lp_timeframe

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
    

        if lp_loadapirates:
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


   