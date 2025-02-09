#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

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

    def __init__(self, datenv, mlenv, globalenv, **kwargs):
        self.dataenv = datenv
        self.mlenv = mlenv
        self.globalenv = globalenv

        # Data Loading Options
        self.api_ticks = kwargs.get("api_ticks", True)
        self.api_rates = kwargs.get("api_rates", True)
        self.file_ticks = kwargs.get("file_ticks", True)
        self.file_rates = kwargs.get("file_rates", True)
        self.dfname1 = kwargs.get("dfname1", "df_rates1")
        self.dfname2 = kwargs.get("dfname2", "df_rates2")
        self.utc_from = kwargs.get("utc_from", None)
        self.symbol_primary = kwargs.get("symbol_primary", None)
        self.rows = kwargs.get("rows", 1000)
        self.rowcount = kwargs.get("rowcount", 10000)

        # File Paths
        self.data_path = kwargs.get("data_path", None)
        self.file_value1 = kwargs.get("file_value1", None)
        self.file_value2 = kwargs.get("file_value2", None)
        self.timeframe = kwargs.get("timeframe", None)
        self.os = kwargs.get("os", "windows")  # 'windows', 'linux', or 'macos'

        # Ensure MetaTrader5 is available if using Windows
        if self.os == "windows":
            try:
                import MetaTrader5 as mt5
                self.command_ticks = kwargs.get("command_ticks", mt5.COPY_TICKS_ALL)
                self.command_rates = kwargs.get("command_rates", None)
            except ImportError:
                logging.error("MetaTrader5 package is missing. Install it with 'pip install MetaTrader5'")
                raise
        else:
            self.command_ticks = kwargs.get("command_ticks", None)
            self.command_rates = kwargs.get("command_rates", None)

        # Store Parameters
        self.api_params = {"ticks": self.api_ticks, "rates": self.api_rates}
        self.file_params = {"ticks": self.file_ticks, "rates": self.file_rates}
        self.data_params = {
            "dfname1": self.dfname1,
            "dfname2": self.dfname2,
            "utc_from": self.utc_from,
            "symbol_primary": self.symbol_primary,
            "rows": self.rows,
            "rowcount": self.rowcount,
        }
        self.command_params = {"ticks": self.command_ticks, "rates": self.command_rates}
        self.file_settings = {"data_path": self.data_path, "file_value1": self.file_value1, "file_value2": self.file_value2}

    def get_params(self):
        """Returns a dictionary of all set parameters."""
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
        if self.os == "windows":
            try:
                import MetaTrader5 as mt5
                self.command_ticks = kwargs.get("command_ticks", mt5.COPY_TICKS_ALL)
                self.command_rates = kwargs.get("command_rates", None)
            except ImportError:
                logging.error("MetaTrader5 package is missing. Install it with 'pip install MetaTrader5'")
                raise
        else:
            self.command_ticks = kwargs.get("command_ticks", None)
            self.command_rates = kwargs.get("command_rates", None)

        if self.os == "windows" :
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
