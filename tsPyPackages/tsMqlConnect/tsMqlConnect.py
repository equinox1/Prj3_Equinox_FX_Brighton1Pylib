#+------------------------------------------------------------------+
#|                                                    tsmqlmod1.pyw
#|                                                    tony shepherd |
#|                                                  https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------

# classes for mql
#+-------------------------------------------------------------------
import MetaTrader5 as mt5
from MetaTrader5 import *
#--------------------------------------------------------------------
# create class  "CMqlinit"
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 
import MetaTrader5 as mt5

import MetaTrader5 as mt5
import keyring as kr

class CMqlinit:
    """
    Initializes and manages the connection to the MT5 terminal.

    Attributes:
        MPPATH (str): Path to the MT5 terminal executable.
        MPLOGIN (int): Login ID for the MT5 account.
        MPPASS (str): Password for the MT5 account.
        MPSERVER (str): Broker's MT5 server.
        MPTIMEOUT (int): Timeout duration for connection.
        MPPORTABLE (bool): Whether MT5 is in portable mode.
        MPENV (str): Environment type (demo or prod).
    """

    def __init__(self, **kwargs):
        self.MPPATH = kwargs.get('MPPATH')
        self.MPLOGIN = kwargs.get('MPLOGIN')
        self.MPPASS = kwargs.get('MPPASS', 'password')
        self.MPSERVER = kwargs.get('MPSERVER')
        self.MPTIMEOUT = kwargs.get('MPTIMEOUT', 60000)
        self.MPPORTABLE = kwargs.get('MPPORTABLE', True)
        self.MPENV = kwargs.get('MPENV', 'demo')

    def run_mql_login(self):
        """Attempts to initialize and login to the MT5 terminal."""
        if mt5.initialize(
            path=self.MPPATH,
            login=self.MPLOGIN,
            password=self.MPPASS,
            server=self.MPSERVER,
            timeout=self.MPTIMEOUT,
            portable=self.MPPORTABLE
        ):
            print("MT5 platform launched successfully:", mt5.version())
            print("Account info:", mt5.account_info())
            print("Environment:", self.MPENV)
            return True
        else:
            print("Initialization failed:", mt5.last_error())
            return mt5.last_error()


class CMqlBrokerConfig:
    """
    Configures broker-specific settings for MT5.

    Attributes:
        lpbroker (str): Broker name.
        mp_symbol_primary (str): Primary symbol for trading.
        MPDATAFILE1 (str): Path to the first data file.
        MPDATAFILE2 (str): Path to the second data file.
    """

    def __init__(self, lpbroker='MetaQuotes', mp_symbol_primary='EURUSD', MPDATAFILE1=None, MPDATAFILE2=None):
        self.lpbroker = lpbroker.upper()
        self.mp_symbol_primary = mp_symbol_primary
        self.MPDATAFILE1 = MPDATAFILE1
        self.MPDATAFILE2 = MPDATAFILE2

    def set_mql_broker(self):
        """Returns broker-specific configurations."""
        base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"
        data_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"
        timeout = 60000
        portable = True
        env = "demo"

        broker_configs = {
            "ICM": {
                "broker": "xerces_icm",
                "broker_path": r"Brokers/ICMarkets/terminal64.exe",
                "files_path": r"Brokers/ICMarkets/MQL5/Files/",
                "server": "ICMarketsSC-Demo",
            },
            "METAQUOTES": {
                "broker": "xerces_meta",
                "broker_path": r"Brokers/Metaquotes/terminal64.exe",
                "files_path": r"Brokers/Metaquotes/MQL5/Files/",
                "server": "MetaQuotes-Demo",
            },
        }

        if self.lpbroker not in broker_configs:
            raise ValueError(f"Unsupported broker: {self.lpbroker}")

        config = broker_configs[self.lpbroker]

        print(f"Configuring Broker: {self.lpbroker}")
        return {
            "BROKER": config["broker"],
            "MPPATH": base_path + config["broker_path"],
            "MPBASEPATH": base_path,
            "MPDATAPATH": data_path,
            "MPFILEVALUE1": f"{self.mp_symbol_primary}_{self.MPDATAFILE1}",
            "MPFILEVALUE2": f"{self.mp_symbol_primary}_{self.MPDATAFILE2}",
            "MKFILES": config["files_path"],
            "MPSERVER": config["server"],
            "MPTIMEOUT": timeout,
            "MPPORTABLE": portable,
            "MPENV": env,
        }


 