"""
Filename: tsMqlConnect.py
File: tsPyPackages/tsMqlConnect/tsMqlConnect.py
Description: Login to Metatrader.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1 (Optimized)
License: (Optional) e.g., MIT License
"""

import logging
import keyring as kr
from tsMqlPlatform import run_platform, platform_checker

# Equinox environment manager
from tsMqlEnvMgr import CMqlEnvMgr

# Configure logging (do this once at the module level)
logger = logging.getLogger(__name__)  # Use module-specific logger
logger.setLevel(logging.INFO)  # Set desired logging level
# ... add handler if needed (e.g. file handler)


# Initialize platform (do this only once)
pchk = run_platform.RunPlatform()
os_platform = platform_checker.get_platform()
loadmql = pchk.check_mql_state()
logger.info(f"TsMqlConnect: Running on: {os_platform} and loadmql state is {loadmql}")

try:
    if loadmql:
        import MetaTrader5 as mt5
    else:
        mt5 = None
except ImportError:
    mt5 = None
    loadmql = False
    logger.error("MetaTrader5 module not found. Ensure it is installed.")

   
class CMqlBrokerConfig:
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

   def __init__(self, lpbroker='METAQUOTES', **kwargs):
      # Initialize the class with the default broker configuration
      self.brokers = self.DEFAULT_BROKERS
      self.lpbroker = lpbroker.upper()
      self.mt5 = mt5 if loadmql else None
      
      # Accessing nested parameters more efficiently
      self.env = CMqlEnvMgr()
      self.params = self.env.all_params()
      self.base_params = self.params.get("base", {})
      self.data_params = self.params.get("data", {})
      self.ml_params = self.params.get("ml", {})
      self.mltune_params = self.params.get("mltune", {})
      self.app_params = self.params.get("app", {})

   
      # Set the path to the broker terminal
       # Initialize parameters, using .get with default values
      self.basepath = self.base_params.get('mp_glob_mql_basepath', '')
      self.datapath = self.base_params.get('mp_glob_mql_data_path', '')
      self.filepath = self.broker_path + self.broker.get('files_path', '')
      self.environment = self.base_params.get('mp_env', 'demo')
      self.mqllogin = self.mqllogin
      self.mqlpass = self.mqlpass
      self.server = self.base_params.get('mp_server', '')
      self.timeout = self.base_params.get('mp_timeout', 60000)
      self.portable = self.base_params.get('mp_portable', True)


      #Run methods to set the broker path and login to the terminal
      self.set_mql_broker_path()
      self.run_mql_login()
      

   def get_login_name(self):
            cred = kr.get_credential(DEFAULT_BROKERS[self.lpbroker], "")
            self.mqllogin = int(cred.username)
            return self.mqllogin

   def get_login_pass(self):
            cred = kr.get_credential(DEFAULT_BROKERS[self.lpbroker], "")
            self.mqlpass = str(cred.password)
            return self.mqlpass


   def set_mql_broker_path(self):
        self.base_path = self.base_params.get('mp_glob_mql_basepath', '')
        self.data_path = self.base_params.get('mp_glob_mql_data_path', '')
        logger.info(f"CMqlBrokerConfig: Base path is {self.base_path} and data path is {self.data_path}")
        self.broker = self.brokers.get(self.lpbroker, {})
        logger.info(f"CMqlBrokerConfig: Broker is {self.broker}")
        self.broker_path = self.base_path + self.broker.get('broker_path', '')
        logger.info(f"CMqlBrokerConfig: Broker path is {self.broker_path}")

   def run_mql_login(self):
         if not self.mt5:  # Simplified check
               raise RuntimeError("CMqlinit:MT5 is not available on this OS.")

         # ... (logging remains the same)

         if self.mt5.initialize(
               path=self.MPPATH,
               login=self.MPLOGIN,
               password=self.MPPASS,
               server=self.MPSERVER,
               timeout=self.MPTIMEOUT,
               portable=self.MPPORTABLE
         ):
               logger.info(f"CMqlinit:MT5 platform launched successfully: {self.mt5.version()}")
               logger.info(f"CMqlinit:MT5 connected to server: {self.MPSERVER}")
               logger.info(f"CMqlinit:Terminal Info: {self.mt5.terminal_info()}")
               return True
         else:
               error_code = self.mt5.last_error() #Store the error code
               logger.error(f"CMqlinit:Initialization failed: {error_code}")
               self.mt5.shutdown()
               return error_code #Return the error code
