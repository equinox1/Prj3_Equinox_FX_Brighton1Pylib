"""
#!/usr/bin/env python3 - uncomment for linux run
# -*- coding: utf-8 -*-  - uncomment for linux run
Filename: tsMqlConnect.py
File: tsPyPackages/tsMqlConnect/tsMqlConnect.py
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.0
License: (Optional) e.g., MIT License
"""

from tsMqlPlatform import run_platform, platform_checker, logger
import keyring as kr

# Initialize platform
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"Running on: {os_platform} and loadmql state is {loadmql}")
if loadmql:
      import MetaTrader5 as mt5

class CMqlinit:
    """
    Initializes and manages the connection to the MT5 terminal.
    """
   

    def __init__(self, **kwargs):
        self.MPPATH = kwargs.get('MPPATH', '')
        self.MPLOGIN = kwargs.get('MPLOGIN', None)
        self.MPPASS = kwargs.get('MPPASS', 'password')
        self.MPSERVER = kwargs.get('MPSERVER', '')
        self.MPTIMEOUT = kwargs.get('MPTIMEOUT', 60000)
        self.MPPORTABLE = kwargs.get('MPPORTABLE', True)
        self.MPENV = kwargs.get('MPENV', 'demo')
       
        
    def run_mql_login(self):
        """Attempts to initialize and login to the MT5 terminal."""
        if loadmql == False:
            raise RuntimeError("MT5 is not available on this OS.")
        if loadmql:
            if self.mt5.initialize(
                path=self.MPPATH,
                login=self.MPLOGIN,
                password=self.MPPASS,
                server=self.MPSERVER,
                timeout=self.MPTIMEOUT,
                portable=self.MPPORTABLE
            ):
                logger.info(f"MT5 platform launched successfully: {self.mt5.version()}")
                return True
            else:
                logger.error(f"Initialization failed: {self.mt5.last_error()}")
                return self.mt5.last_error()

class CMqlBrokerConfig:
    """
    Configures broker-specific settings for MT5.
    """

    DEFAULT_BROKERS = {
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

    def __init__(self, lpbroker='METAQUOTES', mp_symbol_primary='EURUSD', **kwargs):
        self.lpbroker = lpbroker.upper()
        self.mp_symbol_primary = mp_symbol_primary
        self.MPDATAFILE1 = kwargs.get('MPDATAFILE1', 'tickdata1.csv')
        self.MPDATAFILE2 = kwargs.get('MPDATAFILE2', 'ratesdata1.csv')
        self.os = kwargs.get('os', 'windows')
      

    def set_mql_broker(self):
        """Returns broker-specific configurations with override options."""
        mp_mql_basepath = self.all_params['genparams']['globalenv'].get('mp_mql_basepath')
        mp_mql_datapath = self.all_params['genparams']['globalenv'].get('mp_mql_datapath')
         
        if self.lpbroker not in self.DEFAULT_BROKERS:
            raise ValueError(f"Unsupported broker: {self.lpbroker}")

        config = self.DEFAULT_BROKERS[self.lpbroker]

        return {
            "BROKER": config["broker"],
            "MPPATH": base_path + config["broker_path"],
            "MPBASEPATH": base_path,
            "MPDATAPATH": data_path,
            "MPFILEVALUE1": f"{self.mp_symbol_primary}_{self.MPDATAFILE1}",
            "MPFILEVALUE2": f"{self.mp_symbol_primary}_{self.MPDATAFILE2}",
            "MKFILES": config["files_path"],
            "MPSERVER": config["server"],
            "MPTIMEOUT": kwargs.get('MPTIMEOUT', 60000),
            "MPPORTABLE": kwargs.get('MPPORTABLE', True),
            "MPENV": kwargs.get('MPENV', 'demo'),
        }

    def initialize_mt5(self, broker, tm, **kwargs):
        """Initializes MT5 broker settings with overrides."""
        if loadmql:
            broker_config = CMqlBrokerConfig(broker, tm.TIME_CONSTANTS['SYMBOLS'][0], **kwargs).set_mql_broker()
        else:
            broker_config = None
        return broker_config

    def login_mt5(self, broker_config, **kwargs):
        """Logs into MT5 using stored credentials with override options."""
        if loadmql:
            cred = kr.get_credential(broker_config["BROKER"], "")
        else:
            cred = None # Placeholder for non-MT5 platforms
        if loadmql:
            obj = CMqlinit(
                  MPPATH=broker_config.get("MPPATH", kwargs.get("MPPATH")),
                  MPLOGIN=int(kwargs.get("MPLOGIN", cred.username)),
                  MPPASS=str(kwargs.get("MPPASS", cred.password)),
                  MPSERVER=broker_config.get("MPSERVER", kwargs.get("MPSERVER")),
                  MPTIMEOUT=broker_config.get("MPTIMEOUT", kwargs.get("MPTIMEOUT", 60000)),
                  MPPORTABLE=broker_config.get("MPPORTABLE", kwargs.get("MPPORTABLE", True)),
                  MPENV=broker_config.get("MPENV", kwargs.get("MPENV", 'demo')),
            )
            if not obj.run_mql_login():
                  raise ConnectionError("Failed to login to MT5 terminal")
        else:
            obj = None  # Placeholder for non-MT5 platforms
        
        return obj
