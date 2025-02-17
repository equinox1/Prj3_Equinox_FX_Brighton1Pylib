#+------------------------------------------------------------------+
#|                                              tsmqlDataLoader.py
#|                                                    tony shepherd |
#|                                     https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
import logging
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

import tabulate
from tabulate import tabulate
import textwrap



class CDataLoader:
    """Class to store and manage market data parameters with override capability."""

    def __init__(self, all_params, **kwargs):
        # Store environment parameters
        self.all_params = all_params

        # Extracting different parameter sets
        globalenv_params = self.all_params['genparams']['globalenv']
        data_params = self.all_params['dataparams']['dataenv']
        ml_params = self.all_params['mlearnparams']['mlenv']
        tuner_params = self.all_params['tunerparams']['tuneenv']
        model_params = self.all_params['modelparams']['modelenv']

        # Override function: if key is in kwargs, use that; otherwise, use the default from params
        def override(param_key, param_dict, default=None):
            return kwargs.get(param_key, param_dict.get(param_key, default))

        # Overridable parameters
        self.mp_data_path = override('mp_data_path', globalenv_params)
        self.mp_data_file_value1 = override('mp_data_filename1', data_params)
        self.mp_data_file_value2 = override('mp_data_filename2', data_params)

        self.mp_data_loadapiticks = override('mp_data_loadapiticks', data_params, True)
        self.mp_data_loadapirates = override('mp_data_loadapirates', data_params, True)
        self.mp_data_loadfileticks = override('mp_data_loadfileticks', data_params, True)
        self.mp_data_loadfilerates = override('mp_data_loadfilerates', data_params, True)
        self.mp_data_cfg_usedata = override('mp_data_cfg_usedata', data_params, 'loadfilerates')

        self.mp_data_rows = override('mp_data_rows', data_params, 1000)
        self.mp_data_rowcount = override('mp_data_rowcount', data_params, 10000)

        self.mp_data_symbol = override('mp_data_symbol', data_params, 'EURUSD')
        #self.mp_data_utc_from = override('mp_data_utc_from', data_params, self.set_mql_timezone(2021, 1, 1, 'UTC'))
        self.mp_data_utc_to = override('mp_data_utc_to', data_params, None)

        self.mp_data_custom_input_keyfeat = override('mp_data_custom_input_keyfeat', data_params, {'Close'})
        self.mp_data_custom_output_label = override('mp_data_custom_output_label', data_params, self.mp_data_custom_input_keyfeat)

        self.mp_data_timeframe = override('mp_data_timeframe', data_params, None)

        # Logging the overridden values
        logger.info(f"mp_data_path: {self.mp_data_path}")
        logger.info(f"Data file value1: {self.mp_data_file_value1}")
        logger.info(f"Data file value2: {self.mp_data_file_value2}")

        # Store file parameters
        self.files_params = {
            "datapath": self.mp_data_path,
            "filename1": self.mp_data_file_value1,
            "filename2": self.mp_data_file_value2,
        }

    def get_params(self):
        """Returns a dictionary of all set parameters."""
        return {
            "mp_data_path": self.mp_data_path,
            "mp_data_file_value1": self.mp_data_file_value1,
            "mp_data_file_value2": self.mp_data_file_value2,
            "mp_data_symbol": self.mp_data_symbol,
            "mp_data_utc_from": self.mp_data_utc_from,
        }

    def load_data_from_mql(self, **kwargs):
        """Loads market data from API or files."""
        self.mp_data_loadapiticks = kwargs.get("mp_data_loadapiticks")
        self.mp_data_loadapirates = kwargs.get("mp_data_loadapirates")
        self.mp_data_loadfileticks = kwargs.get("mp_data_loadfileticks")
        self.mp_data_loadfilerates = kwargs.get("mp_data_loadfilerates")
        self.mp_data_symbol = kwargs.get("mp_data_symbol")
        self.mp_data_utc_from = kwargs.get("mp_data_utc_from")
        self.mp_data_rows = kwargs.get("mp_data_rows")
        self.mp_data_rowcount = kwargs.get("mp_data_rowcount")
        self.mp_data_timeframe = kwargs.get("mp_data_timeframe")
        self.command_ticks = kwargs.get("command_ticks")
        self.command_rates = kwargs.get("command_rates")
        self.mp_data_path = kwargs.get("mp_data_path")
        self.mp_data_file_value1 = kwargs.get("mp_data_file_value1")
        self.mp_data_file_value2 = kwargs.get("mp_data_file_value2")
        self.mp_data_timeframe = kwargs.get("mp_data_timeframe")
       
        # Reset DataFrames
        rates1, rates2, rates3, rates4 = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # Ensure MetaTrader5 is available if using Windows

        if loadmql:
            if loadapiticks:
                try:
                    logging.info("Fetching tick data from MetaTrader5 API")
                    rates1 = mt5.copy_ticks_from(self.mp_data_symbol, self.mp_data_utc_from, self.mp_data_rows, self.mp_command_ticks)
                   
                    rates1 = pd.DataFrame(rates1)
                    logging.info(f"API tick data received: {len(rates1)} rows" if not rates1.empty else "No tick data found")
                except Exception as e:
                    logging.error(f"MT5 API tick data exception: {e}")

            if loadapirates:
                try:
                    logging.info("Fetching rate data from MetaTrader5 API")
                    self.command_ticksvalid_timeframe = self.validate_timeframe(self.mp_data_timeframe)
                    if valid_timeframe is None:
                        raise ValueError(f"Invalid timeframe: {self.mp_data_timeframe}")
                    rates2 = mt5.copy_rates_from(self.mp_data_symbol, self.valid_timeframe, self.mp_data_utc_from, self.mp_data_rows)
                    rates2 = pd.DataFrame(rates2)
                    logging.info(f"API rate data received: {len(rates2)} rows" if not rates2.empty else "No rate data found")
                except Exception as e:
                    logging.error(f"MT5 API rates exception: {e}")

        if loadfileticks and path and filename1:
            try:
                lpmergepath = f"{self.mp_data_path}/{self.mp_data_file_value1}"
                rates3 = pd.read_csv(lpmergepath, sep=",", nrows=self.mp_data_rowcount, low_memory=False)
                logging.info(f"File tick data received: {len(rates3)} rows" if not rates3.empty else "No tick data found")
            except Exception as e:
                logging.error(f"File load Tick exception: {e}")

        if loadfilerates and path and filename2:
            try:
                lpmergepath = f"{self.mp_data_path}/{self.mp_data_file_value2}"
                rates4 = pd.read_csv(lpmergepath, sep=",", nrows=rowcount, low_memory=False)
                if "vol3" in rates4.columns:
                    rates4.drop("vol3", axis=1, inplace=True)
                logging.info(f"File rate data received: {len(rates4)} rows" if not rates4.empty else "No rate data found")
            except Exception as e:
                logging.error(f"File load Rates exception: {e}")

        return rates1, rates2, rates3, rates4

    def validate_timeframe(self, timeframe):
        """Validates the timeframe string."""
        valid_timeframes = {
            "TIMEFRAME_M1": mt5.TIMEFRAME_M1,
            "TIMEFRAME_M5": mt5.TIMEFRAME_M5,
            "TIMEFRAME_M15": mt5.TIMEFRAME_M15,
            "TIMEFRAME_M30": mt5.TIMEFRAME_M30,
            "TIMEFRAME_H1": mt5.TIMEFRAME_H1,
            "TIMEFRAME_H4": mt5.TIMEFRAME_H4,
            "TIMEFRAME_D1": mt5.TIMEFRAME_D1,
            "TIMEFRAME_W1": mt5.TIMEFRAME_W1,
            "TIMEFRAME_MN1": mt5.TIMEFRAME_MN1,
        }
        return valid_timeframes.get(timeframe)

    def set_mql_timezone(self, year, month, day, timezone):
        """Converts a given date into a timezone-aware datetime object."""
        try:
            lv_timezone = pytz.timezone(timezone)
            native_dt = datetime(year, month, day)
            return lv_timezone.localize(native_dt)
        except Exception as e:
            logging.error(f"Timezone conversion error: {e}")
            return None

    def display_params(self, params, hrows=100, colwidth_env=15, colwidth_param=40, colwidth_value=80, tablefmt="pretty", floatfmt=".5f", numalign="left", stralign="left"):
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