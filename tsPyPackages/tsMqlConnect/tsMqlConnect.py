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


#--------------------------------------------------------------------
# create method  "setbrokers.
# class: cmqlinit      
# usage: login
# /param cmqlinit    var                          
#--------------------------------------------------------------------
    def set_mql_broker(self, lpbroker='MetaQuotes', mp_symbol_primary='EURUSD', MPDATAFILE1=None, MPDATAFILE2=None, **kwargs):
        # Default values for variables
        BROKER = None
        MPPATH = None
        MPDATAPATH = None
        MPFILEVALUE1 = None
        MPFILEVALUE2 = None
        MKFILES = None
        MPSERVER = None
        MPTIMEOUT = None
        MPPORTABLE = None
        MPENV = None

        if lpbroker == "ICM":
            BROKER = "xerces_icm"
            MPBASEPATH = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"
            MPBROKPATH = r"Brokers/ICMarkets/terminal64.exe"
            MKFILES = r"Brokers/ICMarkets/MQL5/Files/"
            MPSERVER = "ICMarketsSC-Demo"
            MPTIMEOUT = 60000
            MPPORTABLE = True
            MPPATH = MPBASEPATH + MPBROKPATH
            MPENV = "demo"  # "prod" or "demo"
            MPDATAPATH = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"
            MPFILEVALUE1 = f"{mp_symbol_primary}_{MPDATAFILE1}"
            MPFILEVALUE2 = f"{mp_symbol_primary}_{MPDATAFILE2}"
            print(f"MPPATH: {MPPATH}")
        elif lpbroker == "MetaQuotes":
            BROKER = "xerces_meta"
            MPBASEPATH = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/mql5/"
            MPBROKPATH = r"Brokers/Metaquotes/terminal64.exe"
            MKFILES = r"Brokers/Metaquotes/MQL5/Files/"
            MPSERVER = "MetaQuotes-Demo"
            MPTIMEOUT = 60000
            MPPORTABLE = True
            MPPATH = MPBASEPATH + MPBROKPATH
            MPENV = "demo"  # "prod" or "demo"
            MPDATAPATH = r"c:/users/shepa/onedrive/8.0 projects/8.3 projectmodelsequinox/equinrun/Mql5Data"
            MPFILEVALUE1 = f"{mp_symbol_primary}_{MPDATAFILE1}"
            MPFILEVALUE2 = f"{mp_symbol_primary}_{MPDATAFILE2}"
            print(f"MPPATH: {MPPATH}")
        else:
            raise ValueError(f"Unsupported broker: {lpbroker}")

    return BROKER, MPPATH, MPDATAPATH, MPFILEVALUE1, MPFILEVALUE2, MKFILES, MPSERVER, MPTIMEOUT, MPPORTABLE, MPENV
