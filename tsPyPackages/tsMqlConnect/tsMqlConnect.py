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
class CMqlinit:
    def __init__(self, MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV,**kwargs):
        self.MPPATH = kwargs.get('MPPATH', None)
        self.MPLOGIN = kwargs.get('MPLOGIN', None)
        self.MPPASS = kwargs.get('MPPASS', 'password')
        self.MPSERVER = kwargs.get('MPSERVER', None)
        self.MPTIMEOUT = kwargs.get('MPTIMEOUT', 60000)
        self.MPPORTABLE = kwargs.get('MPPORTABLE', True)
        self.MPENV = kwargs.get('MPENV', 'demo')
        
#--------------------------------------------------------------------
# create method  "run_mql_login()".
# class: cmqlinit      
# usage: login
# /param cmqlinit    var                          
#--------------------------------------------------------------------
    def run_mql_login(self, MPPATH, MPLOGIN, MPPASS, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV):
        if mt5.initialize(path=MPPATH, login=MPLOGIN, password=MPPASS, server=MPSERVER, timeout=MPTIMEOUT, portable=MPPORTABLE):
            print("Platform mt5 launched correctly: Ver:", mt5.version(),"Info: ", mt5.account_info())
            print("Environment:", MPENV)
            return True
        else:
            print(f"there has been a problem with initialization: {mt5.last_error()}")
            return mt5.last_error()


class CMqlBrokerConfig:
    def __init__(self, lpbroker='MetaQuotes', mp_symbol_primary='EURUSD', MPDATAFILE1=None, MPDATAFILE2=None):
        self.lpbroker = lpbroker
        self.mp_symbol_primary = mp_symbol_primary
        self.MPDATAFILE1 = MPDATAFILE1
        self.MPDATAFILE2 = MPDATAFILE2

    def set_mql_broker(self):
        # Common defaults
        base_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"
        data_path = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"
        timeout = 60000
        portable = True
        env = "demo"  # "prod" or "demo"

        # Broker-specific configurations
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

        # Debugging output
        print(f"Broker: {config['broker']}")
        print(f"Broker Path: {base_path + config['broker_path']}")
        print(f"Base Path: {base_path}")
        print(f"Data Path: {data_path}")
        print(f"MPFILEVALUE1: {self.mp_symbol_primary}_{self.MPDATAFILE1}")
        print(f"MPFILEVALUE2: {self.mp_symbol_primary}_{self.MPDATAFILE2}")
        print(f"Files Path: {config['files_path']}")
        print(f"Server: {config['server']}")
        print(f"Timeout: {timeout}")
        print(f"Portable: {portable}")
        print(f"Environment: {env}")

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

