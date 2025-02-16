#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
from tsMqlPlatform import run_platform, platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql=pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")



import numpy as np
import pandas as pd
import logging
import pytz
from datetime import datetime
from tabulate import tabulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CDataLoader:
    """Class to store and manage market data parameters."""

    def __init__(self, globalenv, datenv,mlenv ,**kwargs):
         #input of all environment settings
         self.globalenv = globalenv
         self.dataenv = datenv
         self.mlenv = mlenv
         
         #file and path parameters
         self.mp_data_path = self.globalenv.get_params().get('mp_data_path', None)
         self.mp_data_file_value1 = self.dataenv.get_params().get('mp_data_filename1', None)
         self.mp_data_file_value2 = self.dataenv.get_params().get('mp_data_filename2', None)
         

         logger.info(f"mp_data_path: {self.mp_data_path}")
         logger.info(f"Data file value1: {self.mp_data_file_value1}")
         logger.info(f"Data file value2: {self.mp_data_file_value2}")

         # load data params
         self.mp_data_loadapiticks = self.dataenv.get_params().get('mp_data_loadapiticks', True)
         self.mp_data_loadapirates = self.dataenv.get_params().get('mp_data_loadapirates', True)
         self.mp_data_loadfileticks = self.dataenv.get_params().get('mp_data_loadfileticks', True)
         self.mp_data_loadfilerates = self.dataenv.get_params().get('mp_data_loadfilerates', True)
         self.mp_data_cfg_usedata = self.dataenv.get_params().get('mp_data_cfg_usedata', 'loadfilerates')
         self.mp_data_show_dtype = self.dataenv.get_params().get('mp_data_show_dtype', False)
         self.mp_data_show_head = self.dataenv.get_params().get('mp_data_show_head', False)
                  
         # datafile parameters
         self.mv_data_dfname1 = self.dataenv.get_params().get('mv_data_dfname1', "df_rates1")
         self.mv_data_dfname2 = self.dataenv.get_params().get('mv_data_dfname2', "df_rates2")
         self.mp_data_rows = self.dataenv.get_params().get('mp_data_rows', 1000)
         self.mp_data_rowcount = self.dataenv.get_params().get('mp_data_rowcount', 10000)
         self.mp_data_rownumber = self.dataenv.get_params().get('mp_data_rownumber', False)
         self.mp_data_history_size = self.dataenv.get_params().get('mp_data_history_size', 5)
         self.mp_data_timeframe = self.dataenv.get_params().get('mp_data_timeframe', None)
         self.mp_data_tab_rows = self.dataenv.get_params().get('mp_data_tab_rows', 10)
         self.mp_data_tab_width = self.dataenv.get_params().get('mp_data_tab_width', 30)
         self.mp_data_symbol = self.dataenv.get_params().get('mp_data_symbol', 'EURUSD')
         self.mp_data_utc_from = self.dataenv.get_params().get('mp_data_utc_from', self.set_mql_timezone(2021, 1, 1, 'UTC'))
     
         #feature and label parameters
         self.mp_data_custom_input_keyfeat = self.dataenv.get_params().get('mp_data_custom_input_keyfeat', {'Close'})
         self.mp_data_custom_output_label = self.dataenv.get_params().get('mp_data_custom_output_label', self.mp_data_custom_input_keyfeat)
         self.mp_data_custom_input_keyfeat_scaled = {feat + '_Scaled' for feat in self.mp_data_custom_input_keyfeat}  # the feature to predict
         self.mp_data_custom_output_label_scaled = {targ + '_Scaled' for targ in self.mp_data_custom_output_label}  # the label shifted to predict
         self.mp_data_custom_output_label_count=len(self.mp_data_custom_output_label)


         #data averaging parameters
         self.mp_hl_avg_col = self.mlenv.get_params().get('mp_hl_avg_col', 'HLAvg')
         self.mp_ma_col = self.mlenv.get_params().get('mp_ma_col', 'SMA')
         self.mp_returns_col = self.mlenv.get_params().get('mp_returns_col', 'LogReturns')
         self.mp_returns_col_scaled = self.mlenv.get_params().get('mp_returns_col_scaled', 'LogReturns_scaled')
         self.mp_create_label = self.mlenv.get_params().get('mp_create_label', False)
         self.mp_create_label_scaled = self.mlenv.get_params().get('mp_create_label_scaled', False)
         self.mp_data_data_label = self.mlenv.get_params().get('mp_data_data_label', 3)


         #Api Command values
         self.mp_data_command_ticks = self.dataenv.get_params().get('mp_data_command_ticks', None)
         self.mp_data_command_rates = self.dataenv.get_params().get('mp_data_command_rates', None)

         # load of datafile parameters
         self.load_params ={
            "loadapiticks": self.mp_data_loadapiticks,
            "loadapirates": self.mp_data_loadapirates,
            "loadfileticks": self.mp_data_loadfileticks,
            "loadfilerates": self.mp_data_loadfilerates,
            "cfg_usedata": self.mp_data_cfg_usedata,
            "show_dtype": self.mp_data_show_dtype,
            "show_head": self.mp_data_show_head,         
         }

         self.datafile_params = {
            "dfname1": self.mv_data_dfname1,
            "dfname2": self.mv_data_dfname2,
            "rows": self.mp_data_rows,
            "rowcount": self.mp_data_rowcount,
            "rownumber": self.mp_data_rownumber,
            "history_size": self.mp_data_history_size,
            "timeframe": self.mp_data_timeframe,
            "tab_rows": self.mp_data_tab_rows,
            "tab_width": self.mp_data_tab_width,
            "symbol": self.mp_data_symbol,
            "utc_from": self.mp_data_utc_from, 
         }
 
         # feature and label parameters
         self.feature_params = {
            "input_keyfeat": self.mp_data_custom_input_keyfeat,
            "output_label": self.mp_data_custom_output_label,
            "input_keyfeat_scaled": self.mp_data_custom_input_keyfeat_scaled,
            "output_label_scaled": self.mp_data_custom_output_label_scaled,
            "output_label_count": self.mp_data_custom_output_label_count,
         }


         self.avg_params = {
            "hl_avg_col": self.mp_hl_avg_col,
            "ma_col": self.mp_ma_col,
            "returns_col": self.mp_returns_col,
            "returns_col_scaled": self.mp_returns_col_scaled,
            "create_label": self.mp_create_label,
            "create_label_scaled": self.mp_create_label_scaled,
            "data_label": self.mp_data_data_label,
         }
   
        

         # Store API command parameters
         self.command_params = {
             "ticks": self.mp_data_command_ticks,
             "rates": self.mp_data_command_rates
         }
        
         # Initialize the MT5 api commands if supported on platform
         if loadmql:
            import MetaTrader5 as mt5
            self.command_params = {
                "ticks": mt5.COPY_TICKS_ALL,
                "rates": None
            }
         else:
            self.command_params = {
                "ticks": None,
                "rates": None
            }

         self.files_params = {
            "datapath": self.mp_data_path,
            "filename1": self.mp_data_file_value1,
            "filename2": self.mp_data_file_value2,
         }
        
    def get_params(self):
        """Returns a dictionary of all set parameters."""
        return {
            "load_params": self.load_params,
            "datafile_params": self.datafile_params,
            "feature_params": self.feature_params,
            "avg_params": self.avg_params,
            "command_params": self.command_params,
            "files_params": self.files_params,
         }


    def load_market_data(self, obj, params_obj):
        """Loads market data using the provided parameters."""
        return self.load_data_from_mql(**params_obj.get_params())

    def load_data_from_mql(self, **kwargs):
        """Loads market data from API or files."""
        loadapiticks = kwargs.get("loadapiticks")
        loadapirates = kwargs.get("loadapirates")
        loadfileticks = kwargs.get("loadfileticks")
        loadfilerates = kwargs.get("loadfilerates")
        symbol = kwargs.get("symbol")
        utc_from = kwargs.get("utc_from")
        rows = kwargs.get("rows")
        rowcount = kwargs.get("rowcount")
        command_ticks = kwargs.get("command_ticks")
        command_rates = kwargs.get("command_rates")
        path = kwargs.get("path")
        filename1 = kwargs.get("filename1")
        filename2 = kwargs.get("filename2")
        timeframe = kwargs.get("timeframe")

        # Reset DataFrames
        rates1, rates2, rates3, rates4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Ensure MetaTrader5 is available if using Windows
        

        if loadmql :
            if loadapiticks:
                try:
                    logging.info("Fetching tick data from MetaTrader5 API")
                    rates1 = mt5.copy_ticks_from(symbol, utc_from, rows, command_ticks)
                    rates1 = pd.DataFrame(rates1)
                    logging.info(f"API tick data received: {len(rates1)} rows" if not rates1.empty else "No tick data found")
                except Exception as e:
                    logging.error(f"MT5 API tick data exception: {e}")

            if loadapirates:
                try:
                    logging.info("Fetching rate data from MetaTrader5 API")
                    valid_timeframe = eval(timeframe)
                    if valid_timeframe is None:
                        raise ValueError(f"Invalid timeframe: {timeframe}")
                    rates2 = mt5.copy_rates_from(symbol, valid_timeframe, utc_from, rows)
                    rates2 = pd.DataFrame(rates2)
                    logging.info(f"API rate data received: {len(rates2)} rows" if not rates2.empty else "No rate data found")
                except Exception as e:
                    logging.error(f"MT5 API rates exception: {e}")

        if loadfileticks and path and filename1:
            try:
                lpmergepath = f"{path}/{filename1}"
                rates3 = pd.read_csv(lpmergepath, sep=",", nrows=rowcount, low_memory=False)
                logging.info(f"File tick data received: {len(rates3)} rows" if not rates3.empty else "No tick data found")
            except Exception as e:
                logging.error(f"File load Tick exception: {e}")

        if loadfilerates and path and filename2:
            try:
                lpmergepath = f"{path}/{filename2}"
                rates4 = pd.read_csv(lpmergepath, sep=",", nrows=rowcount, low_memory=False)
                if "vol3" in rates4.columns:
                    rates4.drop("vol3", axis=1, inplace=True)
                logging.info(f"File rate data received: {len(rates4)} rows" if not rates4.empty else "No rate data found")
            except Exception as e:
                logging.error(f"File load Rates exception: {e}")

        return rates1, rates2, rates3, rates4

    def set_mql_timezone(self, year, month, day, timezone):
        """Converts a given date into a timezone-aware datetime object."""
        try:
            lv_timezone = pytz.timezone(timezone)
            native_dt = datetime(year, month, day)
            return lv_timezone.localize(native_dt)
        except Exception as e:
            logging.error(f"Timezone conversion error: {e}")
            return None


    def display_params(self, params, hrows, colwidth_env, colwidth_param, colwidth_value, tablefmt="pretty", floatfmt=".5f", numalign="left", stralign="left"):
        """Displays the current parameters in a tabular format."""
        # Convert dictionary into a tabulated format
        table_data = []
        for name, param_dict in params.items():
            if param_dict is None:  # Handle NoneType
                param_dict = {}  # Set an empty dictionary if None
            for key, value in param_dict.items():
                if value is None:
                    value = "N/A"  # Replace None values with a placeholder
                table_data.append([name, key, value])
        
        # Generate tabulated output
        formatted_table = tabulate(
            table_data,
            headers=["Environment", "Parameter", "Value"],
            tablefmt=tablefmt,
            floatfmt=floatfmt,
            numalign=numalign,
            stralign=stralign,
            maxcolwidths=[colwidth_env, colwidth_param, colwidth_value]  # Set individual column widths
        )

        # Display the formatted table
        print(formatted_table)
        return formatted_table
