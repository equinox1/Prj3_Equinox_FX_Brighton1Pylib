#+------------------------------------------------------------------+
#|                                                  tsmqlconnect.pyw
#|                                                    tony shepherd |
#|                                    https://www.xercescloud.co.uk |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "https://www.xercescloud.co.uk"
#property version   "1.01"
#+-------------------------------------------------------------------
from tsMqlPlatform import run_platform,platform_checker, PLATFORM_DEPENDENCIES, logger, config
pchk=run_platform.RunPlatform()
os_platform = platform_checker.get_platform()

#--------------------------------------------------------------------
# create class  "CMqlinit"dir
# usage: connect services mql api
#
# section:params
# /param  double svar;              -  value
#-------------------------------------------------------------------- 

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

        # Initialize the MT5 api
        self.os = kwargs.get('os', 'windows')  # windows or linux or macos
        if self.os == 'windows':
            import MetaTrader5 as mt5

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

    def __init__(self, lpbroker='MetaQuotes', mp_symbol_primary='EURUSD', MPDATAFILE1=None, MPDATAFILE2=None, **kwargs):
        self.lpbroker = lpbroker
        self.mp_symbol_primary = mp_symbol_primary
        self.MPDATAFILE1 = MPDATAFILE1
        self.MPDATAFILE2 = MPDATAFILE2
        # Initialize the MT5 api
        self.os = kwargs.get('os', 'windows')  # windows or linux or macos
        if self.os == 'windows':
            import MetaTrader5 as mt5

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


    def initialize_mt5(self,broker, tm):
        mp_symbol_primary = tm.TIME_CONSTANTS['SYMBOLS'][0]
        mp_symbol_secondary = tm.TIME_CONSTANTS['SYMBOLS'][1]
        mp_shiftvalue = tm.TIME_CONSTANTS['DATATYPE']['MINUTES']
        mp_unit = tm.TIME_CONSTANTS['UNIT'][1]
        MPDATAFILE1 = "tickdata1.csv"
        MPDATAFILE2 = "ratesdata1.csv"
        c0 = CMqlBrokerConfig(broker, mp_symbol_primary, MPDATAFILE1, MPDATAFILE2)
        broker_config = c0.set_mql_broker()
        return broker_config, mp_symbol_primary, mp_symbol_secondary, mp_shiftvalue, mp_unit

    def login_mt5(self,broker_config):
        cred = kr.get_credential(broker_config["BROKER"], "")
        if not cred:
            raise ValueError("Credentials not found in keyring")
        try:
            MPLOGIN = int(cred.username)
            MPPASS = str(cred.password)
        except ValueError:
            raise ValueError("Invalid credentials format")
        obj = CMqlinit(
            MPPATH=broker_config["MPPATH"],
            MPLOGIN=MPLOGIN,
            MPPASS=MPPASS,
            MPSERVER=broker_config["MPSERVER"],
            MPTIMEOUT=broker_config["MPTIMEOUT"],
            MPPORTABLE=broker_config["MPPORTABLE"],
            MPENV=broker_config["MPENV"]
        )
        if not obj.run_mql_login():
            raise ConnectionError("Failed to login to MT5 terminal")
        return obj
